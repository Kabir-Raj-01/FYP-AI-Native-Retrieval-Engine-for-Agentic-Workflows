"""Cross-encoder relevance scorer (RERANK_SPEC §2, §4; ADR-005; LIBRARY_NOTES).

``rerank.cross_encoder_model`` at the pinned ``rerank.revision`` on ``rerank.device``,
inputs truncated to ``rerank.max_input_tokens``, batched by ``rerank.batch_size``. The model
is loaded on the first call that needs it, so building an Engine (and direct retrieval
without rerank) does not pay for it. Scores end in [0, 1]: a model whose ``activation_fn``
is ``Identity`` outputs logits and gets a sigmoid; one with a ``Sigmoid`` head
(bge-reranker-base) already returns probabilities. Thread-safe: one lock around load,
cache and predict (eval thread pool, API threadpool).

Score cache (``rerank.score_cache_enabled``, SQLite at ``paths.rerank_cache_db``): a score
is a pure function of (model, revision, max input tokens, query, text), so memoising it
never changes a result — it only removes repeated CPU work across budgets, systems and
calibration runs. Cached pairs cost ~0 ms in the trace's ``rerank`` stage; latency is
reported from ``scripts/latency_profile.py`` (EVAL_PROTOCOL §4), not from quality runs.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from engine.config import RerankConfig
from engine.errors import ConfigError

__all__ = ["CrossEncoderScorer"]

_SCHEMA = "CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)"


class _CrossEncoderModel(Protocol):
    activation_fn: Any

    def predict(self, inputs: list[tuple[str, str]], **kwargs: Any) -> Any: ...


def _load_model(cfg: RerankConfig) -> _CrossEncoderModel:
    from sentence_transformers import CrossEncoder  # heavy import

    model: Any = CrossEncoder(
        cfg.cross_encoder_model,
        device=cfg.device,
        revision=cfg.revision,
        max_length=cfg.max_input_tokens,
    )
    return model  # type: ignore[no-any-return]


class CrossEncoderScorer:
    """``scorer(query, texts) -> relevances`` in [0, 1], one per text, input order."""

    def __init__(
        self,
        cfg: RerankConfig,
        model: _CrossEncoderModel | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._cfg = cfg
        self._model = model
        self._lock = threading.Lock()
        self._cache_path = cache_path
        self._db: sqlite3.Connection | None = None
        self._prefix = f"{cfg.cross_encoder_model}\0{cfg.revision}\0{cfg.max_input_tokens}\0"

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        keys = [self._key(query, t) for t in texts]
        with self._lock:
            known = self._cached(keys)
            todo = [i for i, k in enumerate(keys) if k not in known]
            if todo:
                fresh = self._predict(query, [texts[i] for i in todo])
                new = {keys[i]: s for i, s in zip(todo, fresh, strict=True)}
                self._store(new)
                known |= new
        return [known[k] for k in keys]

    def _key(self, query: str, text: str) -> str:
        return hashlib.sha256(f"{self._prefix}{query}\0{text}".encode()).hexdigest()

    def _predict(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            self._model = _load_model(self._cfg)
        raw = self._model.predict(
            [(query, t) for t in texts],
            batch_size=self._cfg.batch_size,
            show_progress_bar=False,
        )
        head = type(self._model.activation_fn).__name__
        scores = np.asarray(raw, dtype=np.float64).reshape(-1)
        if head == "Identity":  # logits
            scores = 1.0 / (1.0 + np.exp(-scores))
        elif head != "Sigmoid":
            raise ConfigError(
                f"{self._cfg.cross_encoder_model}: unsupported activation {head}; relevance "
                "must map to [0, 1] (RERANK_SPEC §2)"
            )
        return [float(s) for s in np.clip(scores, 0.0, 1.0)]

    def _conn(self) -> sqlite3.Connection | None:
        if self._cache_path is None:
            return None
        if self._db is None:  # used only under self._lock, so one connection is safe
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self._cache_path, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute(_SCHEMA)
        return self._db

    def _cached(self, keys: list[str]) -> dict[str, float]:
        db = self._conn()
        if db is None:
            return {}
        marks = ",".join("?" * len(keys))
        rows = db.execute(f"SELECT key, score FROM scores WHERE key IN ({marks})", keys)
        return {str(k): float(s) for k, s in rows}

    def _store(self, scores: dict[str, float]) -> None:
        db = self._conn()
        if db is None:
            return
        with db:
            db.executemany(
                "INSERT OR REPLACE INTO scores (key, score) VALUES (?, ?)", scores.items()
            )

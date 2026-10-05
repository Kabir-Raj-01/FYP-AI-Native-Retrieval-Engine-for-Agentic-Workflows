"""Cross-encoder relevance scorer (RERANK_SPEC §2, §4; ADR-005; LIBRARY_NOTES).

``rerank.cross_encoder_model`` at the pinned ``rerank.revision`` on ``rerank.device``,
inputs truncated to ``rerank.max_input_tokens``, batched by ``rerank.batch_size``. The model
is loaded on the first call, so building an Engine (and direct retrieval without rerank)
does not pay for it. Scores end in [0, 1]: a model whose ``activation_fn`` is ``Identity``
outputs logits and gets a sigmoid; one with a ``Sigmoid`` head (bge-reranker-base) already
returns probabilities. Thread-safe: one lock around load and predict (eval thread pool, API
threadpool).
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

import numpy as np

from engine.config import RerankConfig
from engine.errors import ConfigError

__all__ = ["CrossEncoderScorer"]


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

    def __init__(self, cfg: RerankConfig, model: _CrossEncoderModel | None = None) -> None:
        self._cfg = cfg
        self._model = model
        self._lock = threading.Lock()

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        with self._lock:
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

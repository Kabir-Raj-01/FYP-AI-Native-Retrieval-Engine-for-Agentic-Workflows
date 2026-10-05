"""Embedder backed by sentence-transformers (RETRIEVAL_SPEC §5; ADR-001).

The model is ``embedding.model_name`` at the pinned ``embedding.revision``. Vectors are
L2-normalised float32, so dot product == cosine. Query texts get ``embedding.query_prefix``
and are cached in an in-process LRU (``embedding.lru_size``); passages get
``embedding.passage_prefix`` and are not cached (bulk ingestion).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Protocol

import numpy as np

from engine.config import EmbeddingConfig
from engine.embeddings.interfaces import Embedder, Vectors
from engine.errors import ConfigError

__all__ = ["SentenceTransformerEmbedder"]


class _EncoderModel(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...

    def get_embedding_dimension(self) -> int | None: ...


def _load_model(cfg: EmbeddingConfig) -> _EncoderModel:
    from sentence_transformers import SentenceTransformer  # heavy import

    # Any: SentenceTransformer.encode's overloads are wider than _EncoderModel needs.
    model: Any = SentenceTransformer(cfg.model_name, device=cfg.device, revision=cfg.revision)
    return model  # type: ignore[no-any-return]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, cfg: EmbeddingConfig, model: _EncoderModel | None = None) -> None:
        self._cfg = cfg
        self._model = model if model is not None else _load_model(cfg)
        actual = self._model.get_embedding_dimension()
        if actual != cfg.dimension:
            raise ConfigError(
                f"{cfg.model_name} produces {actual}-d vectors but embedding.dimension is "
                f"{cfg.dimension}"
            )
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @property
    def dimension(self) -> int:
        return self._cfg.dimension

    def embed_queries(self, texts: list[str]) -> Vectors:
        found: dict[str, np.ndarray] = {}
        for text in dict.fromkeys(texts):
            cached = self._query_cache.get(text)
            if cached is not None:
                self._query_cache.move_to_end(text)
                found[text] = cached
        missing = [t for t in dict.fromkeys(texts) if t not in found]
        if missing:
            vectors = self._encode([self._cfg.query_prefix + t for t in missing])
            for text, vector in zip(missing, vectors, strict=True):
                found[text] = vector
                self._remember(text, vector)
        # Rows come from this call's lookups, not the cache: an LRU smaller than the
        # number of distinct texts may already have evicted some of them.
        return self._stack([found[t] for t in texts])

    def embed_passages(self, texts: list[str]) -> Vectors:
        return self._encode([self._cfg.passage_prefix + t for t in texts])

    def _remember(self, text: str, vector: np.ndarray) -> None:
        if self._cfg.lru_size <= 0:
            return
        self._query_cache[text] = vector
        self._query_cache.move_to_end(text)
        while len(self._query_cache) > self._cfg.lru_size:
            self._query_cache.popitem(last=False)

    def _encode(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self._cfg.dimension), dtype=np.float32)
        encoded = self._model.encode(
            texts,
            batch_size=self._cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(encoded, dtype=np.float32)

    def _stack(self, rows: list[np.ndarray]) -> Vectors:
        if not rows:
            return np.zeros((0, self._cfg.dimension), dtype=np.float32)
        return np.stack(rows).astype(np.float32, copy=False)

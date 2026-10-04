"""FakeEmbedder (TESTING §2): deterministic unit vectors, no model download."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from engine.embeddings.interfaces import Embedder, Vectors

DEFAULT_DIMENSION = 64


class FakeEmbedder(Embedder):
    """Identical text -> identical vector; different texts -> near-orthogonal vectors.

    ``vectors`` pins explicit vectors for texts whose similarities a test controls; they
    are L2-normalised and must have ``dimension`` entries. Queries and passages embed the
    same way (no prefixes), so a text's query and passage vectors are identical.
    """

    def __init__(
        self,
        dimension: int = DEFAULT_DIMENSION,
        vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self._dimension = dimension
        self._pinned = {text: self._normalise(np.asarray(v)) for text, v in (vectors or {}).items()}
        for text, v in self._pinned.items():
            if v.shape != (dimension,):
                raise ValueError(f"pinned vector for {text!r} has shape {v.shape}")
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_queries(self, texts: list[str]) -> Vectors:
        self.calls.append(("queries", list(texts)))
        return self._embed(texts)

    def embed_passages(self, texts: list[str]) -> Vectors:
        self.calls.append(("passages", list(texts)))
        return self._embed(texts)

    def _embed(self, texts: list[str]) -> Vectors:
        rows = [self._pinned[t] if t in self._pinned else self._hashed(t) for t in texts]
        return (
            np.stack(rows).astype(np.float32)
            if rows
            else np.zeros((0, self._dimension), np.float32)
        )

    def _hashed(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        return self._normalise(np.random.default_rng(seed).standard_normal(self._dimension))

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float64)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            raise ValueError("cannot normalise a zero vector")
        return v / norm

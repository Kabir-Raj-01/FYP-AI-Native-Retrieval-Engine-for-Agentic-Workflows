"""Embedder interface (RETRIEVAL_SPEC §5; CLAUDE.md rule 6)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

__all__ = ["Embedder", "Vectors"]

# One row per input text, L2-normalised so that dot product == cosine similarity.
Vectors = NDArray[np.float32]


class Embedder(ABC):
    """The one embedding model (``embedding.model_name``) used for the index, queries,
    cache similarity, gap descriptions and every similarity signal."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector size; equals ``embedding.dimension``."""
        raise NotImplementedError

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> Vectors:
        """Embed queries and gap descriptions; applies ``embedding.query_prefix``.

        Returns an array of shape ``(len(texts), dimension)``.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> Vectors:
        """Embed corpus chunks; applies ``embedding.passage_prefix``.

        Returns an array of shape ``(len(texts), dimension)``.
        """
        raise NotImplementedError

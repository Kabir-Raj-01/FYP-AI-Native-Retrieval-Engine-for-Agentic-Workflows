"""Embedder and SparseEncoder interfaces (RETRIEVAL_SPEC §3, §5; CLAUDE.md rule 6).

``SparseVector`` is a transport type of this interface (like ``Message`` in
``engine/llm/interfaces.py``, ADR-009 §10): it only travels between the sparse encoder, the
index writer and the retriever adapter, never in task state (ADR-001).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

__all__ = ["Embedder", "SparseEncoder", "SparseVector", "Vectors"]

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


class SparseVector(BaseModel):
    """Sparse term weights: parallel lists of term ids and weights."""

    model_config = ConfigDict(extra="forbid")

    indices: list[int]
    values: list[float]


class SparseEncoder(ABC):
    """The one sparse encoder (``sparse.method`` / ``sparse.model_name``) used both at
    ingestion and for queries, so their term ids agree (RETRIEVAL_SPEC §3)."""

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode corpus chunks (document-side term weights)."""
        raise NotImplementedError

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> list[SparseVector]:
        """Encode queries (query-side weights; for BM25 the IDF is applied by Qdrant)."""
        raise NotImplementedError

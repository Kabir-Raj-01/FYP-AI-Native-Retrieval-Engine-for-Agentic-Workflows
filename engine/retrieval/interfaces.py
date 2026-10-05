"""Retrieval backend interfaces (RETRIEVAL_SPEC §1-§4; ADR-009 "Also"; ADR-001).

``RetrieverAdapter`` is the read side used by the engine and the baselines. ``IndexWriter``
is the write side used only by ingestion (``data/``), so that only ``engine/retrieval``
talks to the backend client. ``IndexPoint`` is a transport type of this interface
(ADR-009 §10 precedent); it never appears in task state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from engine.embeddings.interfaces import SparseVector
from engine.state.models import Candidate, RetrievalMode

__all__ = ["IndexPoint", "IndexWriter", "RetrieverAdapter", "SearchFilters"]

# Payload filters (field -> value). RETRIEVAL_SPEC §1 writes a bare `dict`; the value type
# mirrors Candidate.metadata so the signature passes mypy --strict (Week 7 plan, S9).
SearchFilters = Mapping[str, str | int | None]


class RetrieverAdapter(ABC):
    """The only thing the engine knows about the backend. Stateless: no tasks, budgets
    or caches. ``fetch`` is intentionally absent until a caller needs it (ADR-009)."""

    @abstractmethod
    def search(
        self,
        query: str,
        mode: RetrievalMode,
        k: int,
        filters: SearchFilters | None = None,
        with_vectors: bool = True,
    ) -> list[Candidate]:
        """Return at most ``k`` candidates ordered by backend score, ``rank`` 0..k-1.

        ``with_vectors`` fills ``Candidate.vector`` with each chunk's dense vector.
        Scores are not comparable across modes. Never returns partial results.

        Raises:
            BackendError: timeout, connection failure or malformed response.
        """
        raise NotImplementedError

    @abstractmethod
    def health(self) -> bool:
        """True if the backend is reachable and the configured collection exists."""
        raise NotImplementedError


class IndexPoint(BaseModel):
    """One chunk to index: payload fields plus its dense and sparse vectors."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, str | int | None] = {}  # stored flat in the payload
    dense: list[float]
    sparse: SparseVector


class IndexWriter(ABC):
    """Creates and fills the collection ``qdrant.collection`` (RETRIEVAL_SPEC §2)."""

    @abstractmethod
    def ensure_collection(self) -> None:
        """Create the collection if missing (named dense + sparse vectors, payload indexes).

        Raises:
            ConfigError: the collection exists with different vector settings — an index
                change needs a new collection name (ADR-001), never an overwrite.
            BackendError: the backend is unreachable.
        """
        raise NotImplementedError

    @abstractmethod
    def existing_ids(self) -> set[str]:
        """``chunk_id`` of every point already stored (for resumable indexing)."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, points: Sequence[IndexPoint]) -> None:
        """Insert or replace ``points``; idempotent (point id = UUIDv5 of chunk_id)."""
        raise NotImplementedError

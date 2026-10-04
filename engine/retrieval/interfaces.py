"""Retrieval backend interface (RETRIEVAL_SPEC §1, §3, §4; ADR-009 "Also")."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from engine.state.models import Candidate, RetrievalMode

__all__ = ["RetrieverAdapter", "SearchFilters"]

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

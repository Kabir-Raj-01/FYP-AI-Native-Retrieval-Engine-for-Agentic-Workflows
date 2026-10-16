"""Reranker interface (RERANK_SPEC §1-§5; ADR-009 "Also")."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.state.models import Candidate, RetrievalState, ScoredCandidate

__all__ = ["Reranker"]


class Reranker(ABC):
    """Cross-encoder relevance plus state-aware, MMR-style selection."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        state: RetrievalState | None,
        goal: str | None = None,
    ) -> list[ScoredCandidate]:
        """Score and select ``candidates`` for ``query`` (the pending query).

        Relevance is scored against ``query``, the task goal, or the maximum of both, per
        ``rerank.relevance_query`` (ADR-014). The goal is ``goal`` if given, else
        ``state.goal``; with neither (the one-shot paths, where query = goal) it is ``query``.

        With a ``state``: relevance, gap coverage over OPEN gaps, novelty against E_t and
        redundancy against already-selected candidates, combined with
        ``rerank.weights``. With ``state=None`` (direct path, baselines) or
        ``ablation.state_aware_rerank`` false: relevance only, ``final = relevance``, no
        MMR. Returns at most ``rerank.top_n_output`` items sorted by ``final_score``; an
        empty input gives an empty output. One call counts as one reranker call.
        """
        raise NotImplementedError

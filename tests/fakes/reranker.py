"""FakeReranker (TESTING §2): relevance from a {chunk_id: score} map."""

from __future__ import annotations

from collections.abc import Mapping

from engine.rerank.interfaces import Reranker
from engine.state.models import Candidate, RetrievalState, ScoredCandidate

DEFAULT_TOP_N = 10


class FakeReranker(Reranker):
    """Relevance-only scoring (``final = relevance``) in descending order, truncated to
    ``top_n``. State-aware logic is tested with the real Reranker, not this fake."""

    def __init__(
        self,
        relevance: Mapping[str, float] | None = None,
        *,
        default: float = 0.0,
        top_n: int = DEFAULT_TOP_N,
    ) -> None:
        self._relevance = dict(relevance or {})
        self._default = default
        self._top_n = top_n
        self.calls: list[tuple[str, list[str], bool]] = []
        self.goals: list[str | None] = []  # the ``goal`` argument of each call

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        state: RetrievalState | None,
        goal: str | None = None,
    ) -> list[ScoredCandidate]:
        self.calls.append((query, [c.chunk_id for c in candidates], state is not None))
        self.goals.append(goal)
        scored = [
            ScoredCandidate(
                candidate=c,
                relevance=(rel := self._relevance.get(c.chunk_id, self._default)),
                gap_coverage=0.0,
                novelty=1.0,
                redundancy=0.0,
                final_score=rel,
            )
            for c in candidates
        ]
        scored.sort(key=lambda s: (-s.final_score, s.candidate.rank))
        return scored[: self._top_n]

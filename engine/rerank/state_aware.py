"""State-aware reranking (RERANK_SPEC §1-§5; ADR-009 "Also").

``base = w_rel·relevance + w_gap·gap_coverage + w_nov·novelty``, then greedy MMR selection
``final = base - w_red·redundancy(c | selected)`` up to ``rerank.top_n_output``.

- ``state=None`` (direct path, baselines): relevance only — gap coverage 0, novelty 1,
  redundancy 0, ``final = relevance``, cross-encoder order.
- ``ablation.state_aware_rerank`` false (with a state): ``final = relevance`` and
  cross-encoder order, no MMR; gap coverage and novelty are still computed, because the
  EvidenceManager's acceptance checks read them (the ablation isolates the ordering).
- ``ablation.gap_tracking`` false → gap coverage 0; ``ablation.redundancy_signal`` false →
  redundancy 0.
Ties keep backend order. Cosines are clipped to [0, 1].
Relevance is scored against the subquery, the task goal or the maximum of both
(``rerank.relevance_query``, ADR-014); the one-shot paths have query = goal.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from engine.config import EngineConfig
from engine.embeddings.interfaces import Embedder
from engine.monitor.logging import get_logger
from engine.rerank.interfaces import Reranker
from engine.state.models import Candidate, GapStatus, RetrievalState, ScoredCandidate

__all__ = ["Scorer", "StateAwareReranker"]

# (query, texts) -> cross-encoder relevance in [0, 1] per text (CrossEncoderScorer)
Scorer = Callable[[str, list[str]], list[float]]


def _unit(rows: list[list[float]]) -> np.ndarray:
    m = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0.0, 1.0, norms)


class StateAwareReranker(Reranker):
    def __init__(self, cfg: EngineConfig, embedder: Embedder, scorer: Scorer) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._scorer = scorer
        self._log = get_logger("engine.rerank")

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        state: RetrievalState | None,
        goal: str | None = None,
    ) -> list[ScoredCandidate]:
        if not candidates:
            return []
        goal = goal if goal is not None else (state.goal if state is not None else None)
        relevance = self._relevance(query, goal, [c.text for c in candidates])
        n = len(candidates)
        gap = np.zeros(n)
        covered: list[str | None] = [None] * n
        novelty = np.ones(n)
        vectors = None
        if state is not None:
            vectors = self._vectors(candidates)
            gap, covered = self._gap_coverage(vectors, state)
            novelty = self._novelty(candidates, vectors, state)
        if vectors is None or not self._cfg.ablation.state_aware_rerank:
            order = sorted(range(n), key=lambda i: -relevance[i])  # stable sort
            picked = [(i, float(relevance[i]), 0.0) for i in order]
        else:
            picked = self._mmr(relevance, gap, novelty, vectors)
        out = [
            ScoredCandidate(
                candidate=candidates[i],
                relevance=float(relevance[i]),
                gap_coverage=float(gap[i]),
                novelty=float(novelty[i]),
                redundancy=red,
                final_score=final,
                covered_gap_id=covered[i],
            )
            for i, final, red in picked[: self._cfg.rerank.top_n_output]
        ]
        self._log.info(
            "rerank.done",
            query=query,
            relevance_query=self._cfg.rerank.relevance_query,
            state_aware=state is not None and self._cfg.ablation.state_aware_rerank,
            candidates=[
                {
                    "chunk_id": s.candidate.chunk_id,
                    "relevance": round(s.relevance, 4),
                    "gap": round(s.gap_coverage, 4),
                    "novelty": round(s.novelty, 4),
                    "redundancy": round(s.redundancy, 4),
                    "final": round(s.final_score, 4),
                }
                for s in out
            ],
        )
        return out

    def _relevance(self, query: str, goal: str | None, texts: list[str]) -> np.ndarray:
        """Cross-encoder relevance against the query, the goal or both (ADR-014)."""
        mode = self._cfg.rerank.relevance_query
        if mode == "subquery" or goal is None or goal == query:
            return np.asarray(self._scorer(query, texts))
        if mode == "goal":
            return np.asarray(self._scorer(goal, texts))
        return np.maximum(self._scorer(query, texts), self._scorer(goal, texts))

    @staticmethod
    def _vectors(candidates: list[Candidate]) -> np.ndarray:
        missing = [c.chunk_id for c in candidates if c.vector is None]
        if missing:  # Qdrant (with_vectors=True) and cache entries always carry them
            raise ValueError(f"state-aware reranking needs candidate vectors; missing {missing}")
        return _unit([c.vector for c in candidates if c.vector is not None])

    def _gap_coverage(
        self, vectors: np.ndarray, state: RetrievalState
    ) -> tuple[np.ndarray, list[str | None]]:
        gaps = [g for g in state.gaps if g.status == GapStatus.OPEN]
        n = len(vectors)
        if not self._cfg.ablation.gap_tracking or not gaps:
            return np.zeros(n), [None] * n
        g = _unit(self._embedder.embed_queries([x.description for x in gaps]).tolist())
        cos = np.clip(vectors @ g.T, 0.0, 1.0)  # (candidates, gaps)
        best = cos.argmax(axis=1)
        return cos.max(axis=1), [gaps[j].gap_id for j in best]

    @staticmethod
    def _novelty(
        candidates: list[Candidate], vectors: np.ndarray, state: RetrievalState
    ) -> np.ndarray:
        evidence = [e.vector for e in state.evidence if e.vector is not None]
        novelty = np.ones(len(candidates))
        if evidence:
            sim = np.clip(vectors @ _unit(evidence).T, 0.0, 1.0).max(axis=1)
            novelty = 1.0 - sim
        accepted = {e.chunk_id for e in state.evidence}
        for i, c in enumerate(candidates):
            if c.chunk_id in accepted:
                novelty[i] = 0.0
        return novelty

    def _mmr(
        self, relevance: np.ndarray, gap: np.ndarray, novelty: np.ndarray, vectors: np.ndarray
    ) -> list[tuple[int, float, float]]:
        w = self._cfg.rerank.weights
        base = w.relevance * relevance + w.gap_coverage * gap + w.novelty * novelty
        use_redundancy = self._cfg.ablation.redundancy_signal
        remaining = list(range(len(base)))
        picked: list[tuple[int, float, float]] = []
        while remaining and len(picked) < self._cfg.rerank.top_n_output:
            options = []
            for i in remaining:
                red = 0.0
                if use_redundancy and picked:
                    sel = vectors[[p[0] for p in picked]]
                    red = float(np.clip(sel @ vectors[i], 0.0, 1.0).max())
                options.append((i, float(base[i] - w.redundancy * red), red))
            best = max(options, key=lambda o: (o[1], -o[0]))  # ties keep backend order
            picked.append(best)
            remaining.remove(best[0])
        return picked

"""engine/rerank/state_aware.py with a fake cross-encoder scorer (RERANK_SPEC §2-§3, §6)."""

from __future__ import annotations

import math

import pytest

from engine.config import EngineConfig
from engine.rerank.state_aware import StateAwareReranker
from engine.state.models import Candidate, GapStatus, RetrievalState
from tests.builders import make_candidate, make_evidence, make_gap, make_state
from tests.fakes import FakeEmbedder

# FR7 constructed case in 4-d: the open gap's description embeds to GAP; E_t holds one item
# with vector EV. A: cos(A, GAP) 0.5, cos(A, EV) 0.98; B: cos(B, GAP) 0.8, cos(B, EV) 0.1.
GAP = [1.0, 0.0, 0.0, 0.0]
EV = [0.6, 0.8, 0.0, 0.0]
A = [0.5, 0.85, math.sqrt(1 - 0.25 - 0.85**2), 0.0]
B = [0.8, -0.475, 0.0, math.sqrt(1 - 0.64 - 0.475**2)]


def scorer(relevance: dict[str, float]):  # type: ignore[no-untyped-def]
    calls: list[tuple[str, list[str]]] = []

    def score(query: str, texts: list[str]) -> list[float]:
        calls.append((query, texts))
        return [relevance[t] for t in texts]

    score.calls = calls  # type: ignore[attr-defined]
    return score


def cand(chunk_id: str, vector: list[float], rank: int = 0) -> Candidate:
    return make_candidate(chunk_id, text=chunk_id, vector=vector, rank=rank)


def fr7_state() -> RetrievalState:
    return make_state(
        gaps=[make_gap("g0", description="the gap")],
        current_gap_id="g0",
        evidence=[make_evidence("e1", vector=EV)],
    )


def reranker(cfg: EngineConfig, relevance: dict[str, float]) -> StateAwareReranker:
    embedder = FakeEmbedder(dimension=4, vectors={"the gap": GAP})
    return StateAwareReranker(cfg, embedder, scorer(relevance))


def test_fr7_state_aware_prefers_gap_covering_novel_candidate(cfg: EngineConfig) -> None:
    rr = reranker(cfg, {"A": 0.9, "B": 0.6})
    out = rr.rerank("q", [cand("A", A), cand("B", B, rank=1)], fr7_state())
    by_id = {s.candidate.chunk_id: s for s in out}
    assert [s.candidate.chunk_id for s in out] == ["B", "A"]
    assert by_id["B"].final_score == pytest.approx(0.70)
    assert by_id["A"].final_score == pytest.approx(0.644)  # cos(A, B) < 0 -> redundancy 0
    assert (by_id["A"].gap_coverage, by_id["A"].novelty) == (
        pytest.approx(0.5),
        pytest.approx(0.02),
    )
    assert (by_id["B"].gap_coverage, by_id["B"].novelty) == (pytest.approx(0.8), pytest.approx(0.9))
    assert by_id["B"].covered_gap_id == "g0"


def test_fr7_with_state_aware_off_ranks_by_relevance(cfg: EngineConfig) -> None:
    cfg.ablation.state_aware_rerank = False
    out = reranker(cfg, {"A": 0.9, "B": 0.6}).rerank(
        "q", [cand("A", A), cand("B", B, rank=1)], fr7_state()
    )
    assert [(s.candidate.chunk_id, s.final_score) for s in out] == [("A", 0.9), ("B", 0.6)]
    # signals are still computed for the EvidenceManager's checks
    assert out[0].novelty == pytest.approx(0.02) and out[0].redundancy == 0.0


def test_relevance_only_without_state(cfg: EngineConfig) -> None:
    rr = reranker(cfg, {"A": 0.3, "B": 0.6, "C": 0.6})
    out = rr.rerank("q", [cand("A", A), cand("B", B, 1), cand("C", B, 2)], None)
    assert [s.candidate.chunk_id for s in out] == ["B", "C", "A"]  # ties keep backend order
    assert all(
        (s.final_score, s.gap_coverage, s.novelty, s.redundancy, s.covered_gap_id)
        == (s.relevance, 0.0, 1.0, 0.0, None)
        for s in out
    )


def test_near_duplicate_is_pushed_down_by_redundancy(cfg: EngineConfig) -> None:
    state = make_state(gaps=[make_gap("g0", description="the gap")], current_gap_id="g0")
    far = [0.0, 0.0, 1.0, 0.0]
    rr = reranker(cfg, {"A": 0.9, "A2": 0.88, "C": 0.8})
    out = rr.rerank("q", [cand("A", A), cand("A2", A, 1), cand("C", far, 2)], state)
    assert [s.candidate.chunk_id for s in out] == ["A", "C", "A2"]
    assert out[2].redundancy == pytest.approx(1.0)


def test_redundancy_ablation_keeps_the_duplicate_in_place(cfg: EngineConfig) -> None:
    cfg.ablation.redundancy_signal = False
    state = make_state(gaps=[make_gap("g0", description="the gap")], current_gap_id="g0")
    far = [0.0, 0.0, 1.0, 0.0]
    rr = reranker(cfg, {"A": 0.9, "A2": 0.88, "C": 0.8})
    out = rr.rerank("q", [cand("A", A), cand("A2", A, 1), cand("C", far, 2)], state)
    assert [s.candidate.chunk_id for s in out] == ["A", "A2", "C"]


def test_empty_evidence_gives_novelty_one_and_accepted_chunks_zero(cfg: EngineConfig) -> None:
    state = make_state(
        gaps=[make_gap("g0", description="the gap")],
        current_gap_id="g0",
        evidence=[make_evidence("A", vector=[0.0, 0.0, 0.0, 1.0])],  # chunk A already accepted
    )
    out = reranker(cfg, {"A": 0.5, "B": 0.5}).rerank("q", [cand("A", A), cand("B", B, 1)], state)
    assert {s.candidate.chunk_id: s.novelty for s in out}["A"] == 0.0
    fresh = reranker(cfg, {"A": 0.5}).rerank("q", [cand("A", A)], make_state())
    assert fresh[0].novelty == 1.0


def test_gap_coverage_uses_only_open_gaps_and_the_gap_tracking_ablation(cfg: EngineConfig) -> None:
    closed = make_state(gaps=[make_gap("g0", description="the gap", status=GapStatus.EXHAUSTED)])
    out = reranker(cfg, {"B": 0.5}).rerank("q", [cand("B", B)], closed)
    assert (out[0].gap_coverage, out[0].covered_gap_id) == (0.0, None)
    cfg.ablation.gap_tracking = False
    out = reranker(cfg, {"B": 0.5}).rerank("q", [cand("B", B)], fr7_state())
    assert out[0].gap_coverage == 0.0


def test_output_is_truncated_to_top_n(cfg: EngineConfig) -> None:
    cfg.rerank.top_n_output = 2
    rel = {f"c{i}": 0.1 * i for i in range(5)}
    cands = [cand(f"c{i}", [1.0, float(i), 0.0, 0.0], i) for i in range(5)]
    assert len(reranker(cfg, rel).rerank("q", cands, None)) == 2
    assert len(reranker(cfg, rel).rerank("q", cands, make_state())) == 2


def test_empty_input_gives_empty_output_without_scoring(cfg: EngineConfig) -> None:
    rr = reranker(cfg, {})
    assert rr.rerank("q", [], make_state()) == []
    assert rr._scorer.calls == []  # type: ignore[attr-defined]


def test_state_aware_mode_needs_candidate_vectors(cfg: EngineConfig) -> None:
    with pytest.raises(ValueError, match="needs candidate vectors"):
        reranker(cfg, {"A": 0.5}).rerank(
            "q", [make_candidate("A", text="A", vector=None)], make_state()
        )


def test_query_is_passed_to_the_scorer(cfg: EngineConfig) -> None:
    rr = reranker(cfg, {"A": 0.5})
    rr.rerank("the pending query", [cand("A", A)], None)
    assert rr._scorer.calls == [("the pending query", ["A"])]  # type: ignore[attr-defined]

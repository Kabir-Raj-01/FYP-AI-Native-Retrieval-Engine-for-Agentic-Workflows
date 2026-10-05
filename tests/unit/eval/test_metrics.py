"""eval/metrics.py on hand-built cases with known answers (TESTING §4.8)."""

from __future__ import annotations

import math

import pytest

from engine.state.models import Action, EvidenceBundle, StopReason, TraceStep, TraceSummary
from eval.metrics import (
    TaskMetrics,
    aggregate,
    groups_hit,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    task_metrics,
)
from tests.builders import NOW, make_budget, make_evidence

# Two gold groups; group 0 can be satisfied by either "a1" or "a2".
GROUPS = [["a1", "a2"], ["b"]]
RANKED = ["x", "a2", "y", "a1", "b"]  # first hit at rank 2; group 1 hit at rank 5


def test_groups_hit_counts_each_group_once() -> None:
    assert groups_hit(["a1", "a2"], GROUPS) == 1
    assert groups_hit(["a1", "b"], GROUPS) == 2
    assert groups_hit([], GROUPS) == 0


@pytest.mark.parametrize(
    ("k", "hit", "recall"), [(1, 0.0, 0.0), (2, 1.0, 0.5), (4, 1.0, 0.5), (5, 1.0, 1.0)]
)
def test_hit_and_recall_at_k(k: int, hit: float, recall: float) -> None:
    assert hit_at_k(RANKED, GROUPS, k) == hit
    assert recall_at_k(RANKED, GROUPS, k) == recall


def test_mrr_is_reciprocal_rank_of_the_first_gold_member() -> None:
    assert mrr(RANKED, GROUPS) == 0.5
    assert mrr(["x", "y"], GROUPS) == 0.0


def test_ndcg_counts_a_group_only_at_its_first_hit() -> None:
    # gains at ranks 2 (group 0) and 5 (group 1); rank 4's "a1" re-hits group 0: no gain
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, GROUPS, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a1", "b"], GROUPS, 5) == pytest.approx(1.0)  # ideal order
    assert ndcg_at_k(RANKED, GROUPS, 1) == 0.0
    # IDCG@1 places min(|G|, 1) = 1 gain
    assert ndcg_at_k(["b", "x"], GROUPS, 1) == pytest.approx(1.0)


def bundle(ids: list[str], latency: float = 120.0, calls: int = 1) -> EvidenceBundle:
    return EvidenceBundle(
        task_id="t",
        system="s",
        goal="g",
        evidence=[make_evidence(i) for i in ids],
        gaps=[],
        stop_reason=StopReason.ONE_SHOT_COMPLETE,
        trace_summary=TraceSummary(
            steps=1,
            retrieval_calls=calls,
            cache_hits=0,
            llm_calls=0,
            tokens=0,
            reranker_calls=0,
            total_latency_ms=latency,
            actions=[Action.DENSE],
        ),
    )


def step(candidate_ids: list[str]) -> TraceStep:
    return TraceStep(
        run_id="r",
        task_id="t",
        system="s",
        step=1,
        action=Action.DENSE,
        rule="one_shot",
        eligible_actions=[Action.DENSE],
        utility_scores={},
        cache_hit=False,
        candidate_ids=candidate_ids,
        new_candidate_ids=candidate_ids,
        accepted_ids=[],
        rejected={},
        evicted_ids=[],
        resolved_gap_ids=[],
        evidence_gain=0.0,
        plan_events=[],
        stagnation_count=0,
        latency_ms={"total": 1.0},
        llm_calls=0,
        tokens=0,
        budget_after=make_budget(steps_used=1),
        timestamp=NOW,
    )


def test_task_metrics_bundle_vs_trajectory_coverage() -> None:
    # the bundle holds only group 0; group 1 ("b") was retrieved but not kept
    m = task_metrics(
        "t", "inference_query", GROUPS, bundle(["a1", "x"]), [step(["a1", "x", "b"])], [1, 2]
    )
    assert (m.has_gold, m.n_groups) == (True, 2)
    assert m.recall == {1: 0.5, 2: 0.5}
    assert (m.cer, m.partial_coverage, m.trajectory_coverage) == (0.0, 0.5, 1.0)
    assert m.mrr == 1.0
    assert (m.retrieval_calls, m.steps, m.latency_ms) == (1, 1, 120.0)


def test_null_task_has_only_efficiency_metrics() -> None:
    m = task_metrics("t", "null_query", [], bundle(["x"]), [step(["x"])], [1, 5])
    assert m.has_gold is False
    assert m.recall == {} and m.cer is None and m.mrr is None
    assert m.retrieval_calls == 1


def test_aggregate_excludes_null_tasks_from_quality_but_not_efficiency() -> None:
    rows: list[TaskMetrics] = [
        task_metrics(
            "t1",
            "comparison_query",
            GROUPS,
            bundle(["a1", "b"], latency=100.0),
            [step(["a1", "b"])],
            [1, 2],
        ),
        task_metrics(
            "t2",
            "comparison_query",
            GROUPS,
            bundle(["x", "y"], latency=300.0),
            [step(["x", "y"])],
            [1, 2],
        ),
        task_metrics("t3", "null_query", [], bundle(["z"], latency=200.0), [step(["z"])], [1, 2]),
    ]
    out = aggregate(rows)
    assert (out["tasks"], out["tasks_with_gold"]) == (3, 2)
    quality = out["quality"]
    assert quality["recall@2"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
    assert quality["cer"] == pytest.approx(0.5)
    assert out["quality_by_type"]["comparison_query"]["tasks"] == 2
    assert out["efficiency"]["latency_ms_median"] == 200.0  # over all 3 tasks
    assert out["efficiency"]["retrieval_calls_mean"] == 1


def test_row_flattens_per_k_values() -> None:
    m = task_metrics("t", "inference_query", GROUPS, bundle(["a1"]), [step(["a1"])], [1, 3])
    row = m.row()
    assert (row["recall@1"], row["recall@3"], row["hit@1"], row["ndcg@3"]) == (
        0.5,
        0.5,
        1.0,
        row["ndcg@3"],
    )
    assert row["has_gold"] == 1

"""Retrieval / evidence / efficiency metrics (EVAL_PROTOCOL §4), v0.

Computed only from bundles, traces and gold groups — no system-specific branches (FR10).
Gold groups G: a group is *hit* if any of its chunk ids is in the considered set. Tasks with
no gold groups (null queries) are excluded from retrieval and evidence metrics and kept in
efficiency metrics. The bundle is ranked in its stored order (sorted by final_score).
Answer accuracy needs the answer generator (ADR-004) and is not computed yet.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from engine.state.models import EvidenceBundle, TraceStep

__all__ = [
    "TaskMetrics",
    "aggregate",
    "groups_hit",
    "hit_at_k",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "task_metrics",
]

Groups = Sequence[Sequence[str]]


def groups_hit(ids: Sequence[str] | set[str], groups: Groups) -> int:
    present = set(ids)
    return sum(1 for g in groups if present & set(g))


def hit_at_k(ranked: Sequence[str], groups: Groups, k: int) -> float:
    return 1.0 if groups_hit(ranked[:k], groups) else 0.0


def recall_at_k(ranked: Sequence[str], groups: Groups, k: int) -> float:
    return groups_hit(ranked[:k], groups) / len(groups)


def mrr(ranked: Sequence[str], groups: Groups) -> float:
    members = {c for g in groups for c in g}
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in members:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], groups: Groups, k: int) -> float:
    """Binary gain: a rank gains 1 if it holds a member of a gold group not already hit at a
    higher rank; IDCG@k places min(|G|, k) gains at the top ranks (EVAL_PROTOCOL §4)."""
    hit: set[int] = set()
    dcg = 0.0
    for i, chunk_id in enumerate(ranked[:k], start=1):
        new = [j for j, g in enumerate(groups) if j not in hit and chunk_id in g]
        if new:
            hit.add(new[0])
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(groups), k) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass(frozen=True)
class TaskMetrics:
    task_id: str
    question_type: str | None
    has_gold: bool
    n_groups: int
    recall: dict[int, float] = field(default_factory=dict)
    hit: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float | None = None
    cer: float | None = None  # complete-evidence recall: all groups hit in the bundle
    partial_coverage: float | None = None  # hit groups / |G| in the bundle
    trajectory_coverage: float | None = None  # same, over everything retrieved
    retrieval_calls: int = 0
    steps: int = 0
    latency_ms: float = 0.0

    def row(self) -> dict[str, str | float | int | None]:
        """Flat CSV row (``recall@5``, ``hit@5``, ``ndcg@5``, ...)."""
        out: dict[str, str | float | int | None] = {
            "task_id": self.task_id,
            "question_type": self.question_type,
            "has_gold": int(self.has_gold),
            "n_groups": self.n_groups,
        }
        for name, values in (("recall", self.recall), ("hit", self.hit), ("ndcg", self.ndcg)):
            for k, v in sorted(values.items()):
                out[f"{name}@{k}"] = v
        out.update(
            mrr=self.mrr,
            cer=self.cer,
            partial_coverage=self.partial_coverage,
            trajectory_coverage=self.trajectory_coverage,
            retrieval_calls=self.retrieval_calls,
            steps=self.steps,
            latency_ms=round(self.latency_ms, 3),
        )
        return out


def task_metrics(
    task_id: str,
    question_type: str | None,
    gold_groups: Groups,
    bundle: EvidenceBundle,
    steps: Sequence[TraceStep],
    recall_ks: Sequence[int],
) -> TaskMetrics:
    summary = bundle.trace_summary
    if not gold_groups:
        return TaskMetrics(
            task_id=task_id,
            question_type=question_type,
            has_gold=False,
            n_groups=0,
            retrieval_calls=summary.retrieval_calls,
            steps=summary.steps,
            latency_ms=summary.total_latency_ms,
        )
    ranked = [e.chunk_id for e in bundle.evidence]
    retrieved = {c for s in steps for c in s.candidate_ids}
    n = len(gold_groups)
    in_bundle = groups_hit(ranked, gold_groups)
    return TaskMetrics(
        task_id=task_id,
        question_type=question_type,
        has_gold=True,
        n_groups=n,
        recall={k: recall_at_k(ranked, gold_groups, k) for k in recall_ks},
        hit={k: hit_at_k(ranked, gold_groups, k) for k in recall_ks},
        ndcg={k: ndcg_at_k(ranked, gold_groups, k) for k in recall_ks},
        mrr=mrr(ranked, gold_groups),
        cer=1.0 if in_bundle == n else 0.0,
        partial_coverage=in_bundle / n,
        trajectory_coverage=groups_hit(retrieved, gold_groups) / n,
        retrieval_calls=summary.retrieval_calls,
        steps=summary.steps,
        latency_ms=summary.total_latency_ms,
    )


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def aggregate(rows: Sequence[TaskMetrics]) -> dict[str, object]:
    """Means over tasks with gold (quality) and over all tasks (efficiency), plus per type."""
    gold = [r for r in rows if r.has_gold]
    ks = sorted(gold[0].recall) if gold else []

    def quality(subset: Sequence[TaskMetrics]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for k in ks:
            out[f"recall@{k}"] = _mean([r.recall[k] for r in subset])
            out[f"hit@{k}"] = _mean([r.hit[k] for r in subset])
            out[f"ndcg@{k}"] = _mean([r.ndcg[k] for r in subset])
        for name in ("mrr", "cer", "partial_coverage", "trajectory_coverage"):
            out[name] = _mean([getattr(r, name) for r in subset])
        return out

    latencies = sorted(r.latency_ms for r in rows)
    by_type: dict[str, list[TaskMetrics]] = {}
    for r in gold:
        by_type.setdefault(r.question_type or "unknown", []).append(r)
    return {
        "tasks": len(rows),
        "tasks_with_gold": len(gold),
        "quality": quality(gold),
        "quality_by_type": {t: {"tasks": len(v), **quality(v)} for t, v in sorted(by_type.items())},
        "efficiency": {
            "retrieval_calls_mean": _mean([r.retrieval_calls for r in rows]),
            "steps_mean": _mean([r.steps for r in rows]),
            "latency_ms_median": statistics.median(latencies) if latencies else None,
            "latency_ms_p90": (
                latencies[min(len(latencies) - 1, int(len(latencies) * 0.9))] if latencies else None
            ),
        },
    }

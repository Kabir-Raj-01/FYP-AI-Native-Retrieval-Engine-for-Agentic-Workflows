"""dense_oneshot baseline (BASELINES_SPEC §2).

1. ``retrieve_direct(goal, mode=dense, k=retrieval.default_k, rerank=True)``.
2. Bundle = top ``rerank.top_n_output`` by relevance; one TraceStep (action DENSE);
   ``stop_reason = ONE_SHOT_COMPLETE``; retrieval calls = 1, LLM calls = 0.

Interim (until the Reranker exists, ADR-005): without a reranker the bundle is the top
``rerank.top_n_output`` in retrieval order, with ``final_score`` = the raw retrieval score
and ``relevance`` = 0.0; ``eval/run.py`` marks such runs non-reportable.
"""

from __future__ import annotations

from baselines.base import BaselineContext
from engine.monitor.interfaces import TraceWriter
from engine.state.models import (
    Action,
    Budget,
    Candidate,
    EvidenceBundle,
    EvidenceItem,
    RetrievalMode,
    StopReason,
)
from engine.text import summary_of
from eval.datasets.base import Task

__all__ = ["SYSTEM", "run_task"]

SYSTEM = "dense_oneshot"
_ACTION = Action.DENSE
_MODE = RetrievalMode.DENSE


def _item(
    c: Candidate, relevance: float, final: float, task: Task, ctx: BaselineContext
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=c.chunk_id,
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        text=c.text,
        summary=summary_of(c.text, ctx.cfg.planner.evidence_summary_chars),
        relevance=relevance,
        novelty=1.0,  # nothing accepted before this single step
        final_score=final,
        source_action=_ACTION,
        query=task.question,
        metadata=dict(c.metadata),
        added_at_step=1,
        vector=c.vector,
    )


def run_task(task: Task, ctx: BaselineContext) -> EvidenceBundle:
    cfg = ctx.cfg
    k = cfg.retrieval.default_k
    reranked = ctx.engine.has_reranker
    result = ctx.engine.retrieve_direct(task.question, _MODE, k, rerank=reranked)
    if result.scored is not None:
        ranked = [(s.candidate, s.relevance, s.final_score) for s in result.scored]
    else:
        ranked = [(c, 0.0, c.retrieval_score) for c in result.candidates]
    top = ranked[: cfg.rerank.top_n_output]
    evidence = sorted(
        (_item(c, rel, fin, task, ctx) for c, rel, fin in top), key=lambda e: -e.final_score
    )  # stable: ties keep backend order

    budget = Budget(
        max_steps=1,
        max_retrieval_calls=ctx.retrieval_budget,
        max_llm_calls=0,
        max_tokens=cfg.budget_defaults.max_tokens,
        steps_used=1,
        retrieval_calls_used=1,
        reranker_calls_used=1 if reranked else 0,
    )
    candidate_ids = [c.chunk_id for c in result.candidates]
    step = ctx.traces.new_step(
        task_id=task.task_id,
        step=1,
        action=_ACTION,
        rule="one_shot",
        eligible_actions=[_ACTION],
        utility_scores={},  # baselines have no utility (OBSERVABILITY §2)
        query=task.question,
        mode=_MODE,
        k=k,
        cache_hit=False,
        candidate_ids=candidate_ids,
        new_candidate_ids=candidate_ids,
        accepted_ids=[e.evidence_id for e in evidence],
        rejected={},
        evicted_ids=[],
        resolved_gap_ids=[],
        evidence_gain=0.0,
        plan_events=[],
        stagnation_count=0,
        latency_ms={"execute": result.latency_ms, "total": result.latency_ms},
        llm_calls=0,
        tokens=0,
        budget_after=budget,
        stop_reason=StopReason.ONE_SHOT_COMPLETE,
        timestamp=ctx.now(),
    )
    ctx.traces.write_step(step)
    bundle = EvidenceBundle(
        task_id=task.task_id,
        system=ctx.traces.context.system,
        goal=task.question,
        evidence=evidence,
        gaps=[],  # a one-shot baseline plans no gaps
        stop_reason=StopReason.ONE_SHOT_COMPLETE,
        trace_summary=TraceWriter.summarize([step]),
    )
    ctx.traces.write_bundle(bundle)
    return bundle

"""fixed_iteration: agentic RAG with a fixed hop budget (BASELINES_SPEC §2).

1. ``Planner.plan(INIT)`` on the goal — the engine's own planner, prompts and validation.
2. Hop i = 1..N (N = the sub-run's retrieval budget B): query = subgoal i's query while
   there are subgoals left, otherwise a planner REWRITE of the last subgoal given the
   accepted-evidence summaries; when the LLM budget is spent, the last query again with k
   increased by ``retrieval.expand_step_k`` (capped at ``retrieval.max_k``). Retrieve with
   ``fixed_iteration.mode`` and rerank relevance-only (scored per ``rerank.relevance_query``
   against the hop query and/or the goal, ADR-014); accept candidates with relevance ≥
   ``evidence.relevance_threshold`` that are not yet accepted (no novelty or gap logic).
3. No cache, no utility, no early stop: exactly N retrieval calls; the last step has
   ``stop_reason = FIXED_ITERATIONS_COMPLETE``.
4. Bundle = the top ``rerank.top_n_output`` accepted items by relevance.
Without a reranker every relevance is 0, so nothing is accepted (such runs are
non-reportable anyway, ``eval/run.py``).
"""

from __future__ import annotations

from baselines.base import BaselineContext, step_latency
from engine.monitor.interfaces import TraceWriter
from engine.state.models import (
    Action,
    Budget,
    EvidenceBundle,
    EvidenceGap,
    EvidenceItem,
    NeedType,
    PlanEvent,
    PlanMode,
    QueryRecord,
    RetrievalMode,
    RetrievalState,
    StopReason,
)
from engine.text import summary_of
from eval.datasets.base import Task

__all__ = ["SYSTEM", "run_task"]

SYSTEM = "fixed_iteration"


def run_task(task: Task, ctx: BaselineContext) -> EvidenceBundle:
    cfg = ctx.cfg
    n_hops = ctx.retrieval_budget
    mode = RetrievalMode(cfg.fixed_iteration.mode)
    action = Action.DENSE if mode == RetrievalMode.DENSE else Action.SPARSE_HYBRID
    planner = ctx.engine.planner
    now = ctx.now()
    budget = Budget(
        max_steps=n_hops,
        max_retrieval_calls=n_hops,
        max_llm_calls=cfg.budget_defaults.max_llm_calls,
        max_tokens=cfg.budget_defaults.max_tokens,
    )
    state = RetrievalState(
        task_id=task.task_id, goal=task.question, budget=budget, created_at=now, updated_at=now
    )
    init = planner.plan(state, PlanMode.INIT)
    budget = budget.consume(llm_calls=init.llm_calls, tokens=init.tokens)
    subgoals = init.result.subgoals
    accepted: dict[str, EvidenceItem] = {}
    seen: set[str] = set()
    used: list[str] = []  # queries issued for the last subgoal (REWRITE history)
    steps = []
    k = cfg.retrieval.default_k
    llm_before = tokens_before = 0  # hop 1 also carries the INIT planning call
    for hop in range(1, n_hops + 1):
        events: list[PlanEvent] = list(init.events) if hop == 1 else []
        if hop <= len(subgoals):
            query = subgoals[hop - 1].query
        elif budget.llm_calls_used < budget.max_llm_calls:
            last = subgoals[-1]
            gap = EvidenceGap(
                gap_id="g0",
                description=last.description,
                query=used[-1],
                need_type=last.need_type,
                created_at_step=1,
            )
            rewrite_state = state.model_copy(
                update={
                    "gaps": [gap],
                    "current_gap_id": "g0",
                    "needs_initial_plan": False,
                    "evidence": list(accepted.values()),
                    "query_history": [
                        QueryRecord(
                            step=1, gap_id="g0", query=q, need_type=NeedType.MIXED, source="planner"
                        )
                        for q in used
                    ],
                    "budget": budget.model_copy(update={"steps_used": hop - 1}),
                }
            )
            outcome = planner.plan(rewrite_state, PlanMode.REWRITE)
            budget = budget.consume(llm_calls=outcome.llm_calls, tokens=outcome.tokens)
            events += outcome.events
            query = outcome.result.subgoals[0].query
        else:  # LLM budget spent: the same query, deeper
            query = used[-1]
            k = min(k + cfg.retrieval.expand_step_k, cfg.retrieval.max_k)
        if hop >= len(subgoals):
            used.append(query)

        result = ctx.engine.retrieve_direct(
            query, mode, k, rerank=ctx.engine.has_reranker, goal=task.question
        )
        if result.scored is not None:
            ranked = [(s.candidate, s.relevance) for s in result.scored]
        else:
            ranked = [(c, 0.0) for c in result.candidates]
        new_ids = [c.chunk_id for c in result.candidates if c.chunk_id not in seen]
        seen.update(new_ids)
        threshold = cfg.evidence.relevance_threshold
        taken = [
            EvidenceItem(
                evidence_id=c.chunk_id,
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                summary=summary_of(c.text, cfg.planner.evidence_summary_chars),
                relevance=rel,
                novelty=1.0,  # no novelty logic in this baseline
                final_score=rel,
                source_action=action,
                query=query,
                metadata=dict(c.metadata),
                added_at_step=hop,
                vector=c.vector,
            )
            for c, rel in ranked
            if rel >= threshold and c.chunk_id not in accepted
        ]
        accepted.update({e.evidence_id: e for e in taken})
        budget = budget.consume(
            steps=1, retrieval_calls=1, reranker_calls=1 if result.scored is not None else 0
        )
        last_hop = hop == n_hops
        step = ctx.traces.new_step(
            task_id=task.task_id,
            step=hop,
            action=action,
            rule="fixed_iteration",
            eligible_actions=[action],
            utility_scores={},
            query=query,
            mode=mode,
            k=k,
            cache_hit=False,
            candidate_ids=[c.chunk_id for c in result.candidates],
            new_candidate_ids=new_ids,
            accepted_ids=[e.evidence_id for e in taken],
            rejected={},
            evicted_ids=[],
            resolved_gap_ids=[],
            evidence_gain=0.0,
            plan_events=events,
            stagnation_count=0,
            latency_ms=step_latency(result),
            llm_calls=budget.llm_calls_used - llm_before,
            tokens=budget.tokens_used - tokens_before,
            budget_after=budget,
            stop_reason=StopReason.FIXED_ITERATIONS_COMPLETE if last_hop else None,
            timestamp=ctx.now(),
        )
        ctx.traces.write_step(step)
        steps.append(step)
        llm_before, tokens_before = budget.llm_calls_used, budget.tokens_used

    top = sorted(accepted.values(), key=lambda e: -e.relevance)[: cfg.rerank.top_n_output]
    bundle = EvidenceBundle(
        task_id=task.task_id,
        system=ctx.traces.context.system,
        goal=task.question,
        evidence=top,
        gaps=[],
        stop_reason=StopReason.FIXED_ITERATIONS_COMPLETE,
        trace_summary=TraceWriter.summarize(steps),
    )
    ctx.traces.write_bundle(bundle)
    return bundle

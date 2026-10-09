"""Node functions of the controlled loop (ARCHITECTURE §4-§5; STATE_SPEC §4; ADR-009).

Each node is ``node(gs: GraphState, ctx: Context) -> dict`` and returns full replacement
values for ``state`` and/or ``scratch``: LangGraph replaces a returned field wholesale and
silently drops unknown keys (LIBRARY_NOTES "LangGraph"). Nodes never mutate their input
(STATE_SPEC §5 inv 9) — every change is a ``model_copy``. Only the fields the ownership
matrix gives a node are written by it.

Interpretation (spec gap, ARCHITECTURE §5 vs CACHE_SPEC §4): ``scratch.candidates`` holds
everything an action returned, so the queued cache entry stores all candidates; for
RERANK_EXPAND, ``score_and_update`` passes only the unseen ones to the reranker.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from engine.cache.interfaces import Cache
from engine.config import EngineConfig
from engine.controller.interfaces import Controller
from engine.embeddings.interfaces import Embedder
from engine.evidence.interfaces import EvidenceManager
from engine.monitor.interfaces import TraceWriter, TrajectoryMonitor
from engine.monitor.logging import get_logger
from engine.planner.interfaces import Planner
from engine.rerank.interfaces import Reranker
from engine.retrieval.interfaces import RetrieverAdapter
from engine.state.interfaces import TaskStore
from engine.state.invariants import check_invariants
from engine.state.models import (
    Action,
    ActionDecision,
    CacheEntry,
    EvidenceGap,
    EvidenceUpdate,
    GapStatus,
    GraphState,
    IterationScratch,
    LastResult,
    NeedType,
    PlanMode,
    PlanOutcome,
    PlanStatus,
    QueryRecord,
    RetrievalMode,
    RetrievalRecord,
    RetrievalState,
    TaskStatus,
    TraceStep,
)
from engine.text import normalize

__all__ = [
    "CANDIDATE_ACTIONS",
    "Components",
    "Context",
    "execute",
    "maybe_plan",
    "monitor",
    "persist",
    "score_and_update",
    "select_action",
]

# Actions whose step produces candidates (rerank -> evidence -> monitor gain).
CANDIDATE_ACTIONS = frozenset(
    {Action.REUSE, Action.DENSE, Action.SPARSE_HYBRID, Action.RERANK_EXPAND}
)
_FRESH = frozenset({Action.DENSE, Action.SPARSE_HYBRID, Action.RERANK_EXPAND})
_CLOSED = frozenset({GapStatus.RESOLVED, GapStatus.EXHAUSTED, GapStatus.SUPERSEDED})

_log = get_logger("engine.graph")


@dataclass(frozen=True)
class Components:
    """Component instances the nodes call (built by ``engine/factory.py``)."""

    store: TaskStore
    cache: Cache
    planner: Planner
    controller: Controller
    retriever: RetrieverAdapter
    reranker: Reranker
    evidence: EvidenceManager
    monitor: TrajectoryMonitor
    embedder: Embedder
    traces: TraceWriter


@dataclass
class Context:
    """Per-iteration call context — not state, never persisted."""

    cfg: EngineConfig
    parts: Components
    prev: RetrievalState  # deep copy of the loaded state (check_invariants' ``prev``)
    clock: Callable[[], float]
    now: Callable[[], datetime]
    started: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)
    acted_gap_id: str | None = None  # the gap this iteration worked on (TraceStep)
    trace: TraceStep | None = None  # written by persist

    @property
    def n(self) -> int:
        """The step number this iteration becomes (ADR-009 "Also": steps_used + 1)."""
        return self.prev.budget.steps_used + 1


@contextmanager
def _timed(ctx: Context, stage: str) -> Iterator[None]:
    start = ctx.clock()
    try:
        yield
    finally:
        ms = (ctx.clock() - start) * 1000.0
        ctx.latency_ms[stage] = ctx.latency_ms.get(stage, 0.0) + ms


def _decision(state: RetrievalState) -> ActionDecision:
    if state.last_decision is None:  # select_action always runs first
        raise RuntimeError(f"task {state.task_id}: no decision in this iteration")
    return state.last_decision


def _query_records(outcome: PlanOutcome, gaps: list[EvidenceGap], n: int) -> list[QueryRecord]:
    fallback = any(e.status == PlanStatus.FALLBACK for e in outcome.events)
    source = "fallback" if fallback else "planner"
    return [
        QueryRecord(step=n, gap_id=g.gap_id, query=g.query, need_type=g.need_type, source=source)
        for g in gaps
    ]


def _new_gaps(
    state: RetrievalState, outcome: PlanOutcome, n: int, parent: str | None = None
) -> list[EvidenceGap]:
    first = len(state.gaps)  # ids g0, g1, ... unique within the task
    gaps = [
        EvidenceGap(
            gap_id=f"g{first + i}",
            description=sg.description,
            query=sg.query,
            need_type=sg.need_type,
            parent_gap_id=parent,
            created_at_step=n,
        )
        for i, sg in enumerate(outcome.result.subgoals)
    ]
    for g in gaps:
        _log.info("gap.created", gap_id=g.gap_id, parent=parent, need_type=g.need_type)
    return gaps


def _after_plan(state: RetrievalState, outcome: PlanOutcome, **update: Any) -> RetrievalState:
    return state.model_copy(
        update={
            "plan_events": [*state.plan_events, *outcome.events],
            "budget": state.budget.consume(llm_calls=outcome.llm_calls, tokens=outcome.tokens),
            **update,
        }
    )


# ------------------------------------------------------------------------------- nodes
def maybe_plan(gs: GraphState, ctx: Context) -> dict[str, Any]:
    """INIT planning on the first iteration; otherwise a no-op."""
    s = gs.state
    if not s.needs_initial_plan:
        return {}
    with _timed(ctx, "plan"):
        outcome = ctx.parts.planner.plan(s, PlanMode.INIT)
    gaps = _new_gaps(s, outcome, ctx.n)
    s = _after_plan(
        s,
        outcome,
        gaps=[*s.gaps, *gaps],
        query_history=[*s.query_history, *_query_records(outcome, gaps, ctx.n)],
        needs_initial_plan=False,
    )
    s = s.model_copy(update={"current_gap_id": ctx.parts.monitor.advance(s).current_gap_id})
    return {"state": s, "scratch": gs.scratch.model_copy(update={"plan_outcome": outcome})}


def select_action(gs: GraphState, ctx: Context) -> dict[str, Any]:
    """Cache lookup for the pending query, then the controller's decision."""
    s = gs.state
    ctx.acted_gap_id = s.current_gap_id
    with _timed(ctx, "select"):
        hit = (
            ctx.parts.cache.lookup(s.task_id, s.pending_query, s.current_need_type)
            if s.pending_query
            else None
        )
        decision = ctx.parts.controller.select_action(s, hit)
    _log.info(
        "controller.decision",
        eligible=[a.value for a in decision.eligible],
        scores={a.value: sc.model_dump() for a, sc in decision.scores.items()},
        chosen=decision.action.value,
        rule=decision.rule,
    )
    return {
        "state": s.model_copy(update={"last_decision": decision}),
        "scratch": gs.scratch.model_copy(update={"cache_hit": hit}),
    }


def _search_args(
    s: RetrievalState, action: Action, cfg: EngineConfig
) -> tuple[str, RetrievalMode, int]:
    r = cfg.retrieval
    if action == Action.DENSE:
        return s.pending_query, RetrievalMode.DENSE, r.default_k
    if action == Action.SPARSE_HYBRID:
        lexical = s.current_need_type == NeedType.LEXICAL
        return (
            s.pending_query,
            RetrievalMode.SPARSE if lexical else RetrievalMode.HYBRID,
            r.default_k,
        )
    last = s.last_result  # RERANK_EXPAND: precondition guarantees a last result
    if last is None or last.mode is None:
        raise RuntimeError(f"task {s.task_id}: RERANK_EXPAND without a last result")
    return last.query, last.mode, min(last.depth + r.expand_step_k, r.max_k)


def execute(gs: GraphState, ctx: Context) -> dict[str, Any]:
    """Run the chosen action (ARCHITECTURE §5)."""
    s = gs.state
    decision = _decision(s)
    action = decision.action
    if action == Action.STOP:
        return {}
    if action == Action.TRANSFORM:
        return _transform(gs, ctx, decision)
    with _timed(ctx, "execute"):
        if action == Action.REUSE:
            hit = gs.scratch.cache_hit
            if hit is None:
                raise RuntimeError(f"task {s.task_id}: REUSE without a cache hit")
            query, mode = s.pending_query, hit.entry.mode
            candidates, k, latency_ms, calls = (
                hit.entry.candidates,
                len(hit.entry.candidates),
                0.0,
                0,
            )
        else:
            query, mode, k = _search_args(s, action, ctx.cfg)
            start = ctx.clock()
            candidates = ctx.parts.retriever.search(query, mode, k)
            latency_ms, calls = (ctx.clock() - start) * 1000.0, 1
            _log.info(
                "retrieval.call",
                path="controlled",
                mode=mode.value,
                k=k,
                query=query,
                n_returned=len(candidates),
                latency_ms=round(latency_ms, 2),
            )
    seen = set(s.seen_candidate_ids)
    new_ids = [c.chunk_id for c in candidates if c.chunk_id not in seen]
    record = RetrievalRecord(
        step=ctx.n,
        action=action,
        mode=mode,
        query=query,
        k=k,
        candidate_ids=[c.chunk_id for c in candidates],
        new_candidate_ids=new_ids,
        cache_hit=action == Action.REUSE,
        latency_ms=latency_ms,
    )
    s = s.model_copy(
        update={
            "retrieval_history": [*s.retrieval_history, record],
            "seen_candidate_ids": [*s.seen_candidate_ids, *new_ids],
            "budget": s.budget.consume(retrieval_calls=calls),
        }
    )
    return {"state": s, "scratch": gs.scratch.model_copy(update={"candidates": candidates})}


def _transform(gs: GraphState, ctx: Context, decision: ActionDecision) -> dict[str, Any]:
    """TRANSFORM: REWRITE the current gap's query, or SPLIT it into children."""
    s = gs.state
    current = s.current_gap
    if decision.plan_mode is None or current is None:
        raise RuntimeError(f"task {s.task_id}: TRANSFORM without plan_mode or current gap")
    with _timed(ctx, "plan"):
        outcome = ctx.parts.planner.plan(s, decision.plan_mode)
    subgoals = outcome.result.subgoals
    if outcome.result.mode == PlanMode.SPLIT and len(subgoals) >= 2:
        children = _new_gaps(s, outcome, ctx.n, parent=current.gap_id)
        _log.info("gap.superseded", gap_id=current.gap_id, by=[c.gap_id for c in children])
        gaps = [
            g.model_copy(update={"status": GapStatus.SUPERSEDED, "closed_at_step": ctx.n})
            if g.gap_id == current.gap_id
            else g
            for g in s.gaps
        ]
        s = _after_plan(
            s,
            outcome,
            gaps=[*gaps, *children],
            query_history=[*s.query_history, *_query_records(outcome, children, ctx.n)],
        )
        s = s.model_copy(update={"current_gap_id": ctx.parts.monitor.advance(s).current_gap_id})
    else:  # REWRITE, and the deterministic fallback of either mode (PLANNER_SPEC §6)
        sg = subgoals[0]
        rewritten = current.model_copy(update={"query": sg.query, "need_type": sg.need_type})
        gaps = [rewritten if g.gap_id == current.gap_id else g for g in s.gaps]
        s = _after_plan(
            s,
            outcome,
            gaps=gaps,
            query_history=[*s.query_history, *_query_records(outcome, [rewritten], ctx.n)],
        )
    return {"state": s, "scratch": gs.scratch.model_copy(update={"plan_outcome": outcome})}


def _apply_update(s: RetrievalState, u: EvidenceUpdate, n: int) -> RetrievalState:
    evicted = set(u.evicted)
    items = []
    for item in s.evidence:
        if item.evidence_id in evicted:
            continue
        change = u.item_changes.get(item.evidence_id)
        if change is not None:
            item = item.model_copy(update=change.model_dump(exclude_none=True))
        items.append(item)
    gaps = [
        g.model_copy(
            update={
                "status": GapStatus.RESOLVED,
                "resolved_by": u.gap_resolutions[g.gap_id],
                "closed_at_step": n,
            }
        )
        if g.gap_id in u.gap_resolutions
        else g
        for g in s.gaps
    ]
    return s.model_copy(update={"evidence": [*items, *u.accepted], "gaps": gaps, "last_update": u})


def score_and_update(gs: GraphState, ctx: Context) -> dict[str, Any]:
    """Rerank this step's candidates, update evidence, queue the fresh cache entry."""
    s = gs.state
    action = _decision(s).action
    if action not in CANDIDATE_ACTIONS:
        return {"state": s.model_copy(update={"last_update": None})}  # ADR-009 §9
    record = s.retrieval_history[-1]  # written by execute in this iteration
    pool = gs.scratch.candidates
    if action == Action.RERANK_EXPAND:  # only unseen candidates are passed on
        fresh = set(record.new_candidate_ids)
        pool = [c for c in pool if c.chunk_id in fresh]
    scored = []
    if pool:
        with _timed(ctx, "rerank"):
            scored = ctx.parts.reranker.rerank(record.query, pool, s)
        s = s.model_copy(update={"budget": s.budget.consume(reranker_calls=1)})
    with _timed(ctx, "evidence"):
        update = ctx.parts.evidence.update(s, scored, ctx.n)
    top_n = ctx.cfg.rerank.top_n_output
    s = _apply_update(s, update, ctx.n).model_copy(
        update={
            "last_result": LastResult(
                step=ctx.n,
                query=record.query,
                mode=record.mode,
                depth=record.k,
                candidate_ids=record.candidate_ids,
                top_relevances=sorted((x.relevance for x in scored[:top_n]), reverse=True),
            )
        }
    )
    queued = list(gs.scratch.queued_cache_entries)
    if action in _FRESH and record.mode is not None:
        vector = ctx.parts.embedder.embed_queries([record.query])[0]
        queued.append(
            CacheEntry(
                task_id=s.task_id,
                entry_id=f"step-{ctx.n}",  # one fresh retrieval per step
                normalized_query=normalize(record.query),
                query_vector=[float(x) for x in vector],
                need_type=s.current_need_type,
                mode=record.mode,
                candidates=gs.scratch.candidates,
                accepted_ids=[e.evidence_id for e in update.accepted],
                created_at_step=ctx.n,
            )
        )
    scratch = gs.scratch.model_copy(
        update={"scored": scored, "evidence_update": update, "queued_cache_entries": queued}
    )
    return {"state": s, "scratch": scratch}


def monitor(gs: GraphState, ctx: Context) -> dict[str, Any]:
    """Apply the TrajectoryMonitor's verdict verbatim (ADR-009 §4)."""
    s = gs.state
    decision = _decision(s)
    update = s.last_update if decision.action in CANDIDATE_ACTIONS else None
    with _timed(ctx, "monitor"):
        verdict = ctx.parts.monitor.observe(s, decision, update)
    changes = verdict.gap_status_changes
    for gap_id, status in changes.items():
        if status == GapStatus.EXHAUSTED:
            _log.info("gap.exhausted", gap_id=gap_id, stagnation_count=verdict.stagnation_count)
    gaps = [
        g.model_copy(
            update={
                "status": changes[g.gap_id],
                "closed_at_step": g.closed_at_step
                if g.closed_at_step is not None or changes[g.gap_id] not in _CLOSED
                else ctx.n,
            }
        )
        if g.gap_id in changes
        else g
        for g in s.gaps
    ]
    upd: dict[str, Any] = {
        "gaps": gaps,
        "stagnation_count": verdict.stagnation_count,
        "transforms_in_streak": verdict.transforms_in_streak,
        "action_gain_ema": verdict.action_gain_ema,
        "current_gap_id": verdict.current_gap_id,
    }
    if not verdict.should_continue:
        upd |= {"status": TaskStatus.STOPPED, "stop_reason": verdict.stop_reason}
    _log.info(
        "monitor.verdict",
        should_continue=verdict.should_continue,
        stop_reason=verdict.stop_reason,
        stagnation_count=verdict.stagnation_count,
        advanced_to=verdict.advanced_to_gap,
    )
    return {"state": s.model_copy(update=upd)}


def persist(gs: GraphState, ctx: Context) -> dict[str, Any]:
    """Count the step, check invariants, save state + queued cache entries, write the trace."""
    s = gs.state
    new = s.model_copy(
        update={
            "budget": s.budget.consume(steps=1),
            "version": s.version + 1,
            "updated_at": ctx.now(),
        }
    )
    check_invariants(ctx.prev, new, ctx.cfg)
    ctx.parts.store.save(new, s.version, gs.scratch.queued_cache_entries)
    ctx.trace = _trace_step(gs, new, ctx)
    ctx.parts.traces.write_step(ctx.trace)
    return {"state": new, "scratch": IterationScratch()}


def _trace_step(gs: GraphState, new: RetrievalState, ctx: Context) -> TraceStep:
    decision = _decision(new)
    fields: dict[str, Any] = {
        "query": None,
        "mode": None,
        "k": None,
        "cache_hit": False,
        "candidate_ids": [],
        "new_candidate_ids": [],
        "accepted_ids": [],
        "rejected": {},
        "evicted_ids": [],
        "resolved_gap_ids": [],
        "evidence_gain": 0.0,
    }
    if decision.action in CANDIDATE_ACTIONS:
        r = new.retrieval_history[-1]
        fields |= {
            "query": r.query,
            "mode": r.mode,
            "k": r.k,
            "cache_hit": r.cache_hit,
            # post-filter for RERANK_EXPAND (OBSERVABILITY §2)
            "candidate_ids": r.new_candidate_ids
            if decision.action == Action.RERANK_EXPAND
            else r.candidate_ids,
            "new_candidate_ids": r.new_candidate_ids,
        }
    u = gs.scratch.evidence_update
    if u is not None:
        fields |= {
            "accepted_ids": [e.evidence_id for e in u.accepted],
            "rejected": u.rejected,
            "evicted_ids": u.evicted,
            "resolved_gap_ids": u.resolved_gap_ids,
            "evidence_gain": u.evidence_gain,
        }
    prev = ctx.prev.budget
    latency = dict(ctx.latency_ms)
    latency["total"] = (ctx.clock() - ctx.started) * 1000.0
    return ctx.parts.traces.new_step(
        task_id=new.task_id,
        step=ctx.n,
        action=decision.action,
        rule=decision.rule,
        eligible_actions=decision.eligible,
        utility_scores=decision.scores,
        current_gap_id=ctx.acted_gap_id,
        plan_events=[e for e in new.plan_events if e.step == ctx.n],
        stagnation_count=new.stagnation_count,
        latency_ms=latency,
        llm_calls=new.budget.llm_calls_used - prev.llm_calls_used,
        tokens=new.budget.tokens_used - prev.tokens_used,
        budget_after=new.budget,
        stop_reason=new.stop_reason,
        timestamp=ctx.now(),
        **fields,
    )

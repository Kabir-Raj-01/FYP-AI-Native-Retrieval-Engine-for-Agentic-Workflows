"""Plain constructors for shared models, used by unit tests and the conftest fixtures.

Literal values here are test data, not engine configuration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from engine.state.models import (
    Action,
    ActionDecision,
    ActionScores,
    AdvanceResult,
    Budget,
    CacheEntry,
    CacheHit,
    Candidate,
    EvidenceBundle,
    EvidenceGap,
    EvidenceItem,
    EvidenceUpdate,
    GapStatus,
    GraphState,
    ItemChange,
    IterationScratch,
    LastResult,
    MonitorVerdict,
    NeedType,
    PlanEvent,
    PlanLLMOutput,
    PlanMode,
    PlanOutcome,
    PlanResult,
    PlanStatus,
    QueryRecord,
    RetrievalMode,
    RetrievalRecord,
    RetrievalState,
    RunContext,
    ScoredCandidate,
    StopReason,
    SubgoalPlan,
    TraceStep,
    TraceSummary,
)

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def vec(*values: float) -> list[float]:
    return list(values)


def make_budget(**overrides: Any) -> Budget:
    fields: dict[str, Any] = {
        "max_steps": 8,
        "max_retrieval_calls": 4,
        "max_llm_calls": 6,
        "max_tokens": 20000,
    }
    fields.update(overrides)
    return Budget(**fields)


def make_candidate(chunk_id: str = "c1", **overrides: Any) -> Candidate:
    fields: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": f"d-{chunk_id}",
        "text": f"text of {chunk_id}",
        "retrieval_score": 0.8,
        "rank": 0,
        "mode": RetrievalMode.DENSE,
        "metadata": {"title": "A title", "published_at": "2026-01-01", "source": None},
        "vector": vec(0.6, 0.8, 0.0),
    }
    fields.update(overrides)
    return Candidate(**fields)


def make_scored(chunk_id: str = "c1", **overrides: Any) -> ScoredCandidate:
    fields: dict[str, Any] = {
        "candidate": make_candidate(chunk_id),
        "relevance": 0.9,
        "gap_coverage": 0.7,
        "novelty": 0.8,
        "redundancy": 0.0,
        "final_score": 0.85,
        "covered_gap_id": "g0",
    }
    fields.update(overrides)
    return ScoredCandidate(**fields)


def make_evidence(evidence_id: str = "c1", **overrides: Any) -> EvidenceItem:
    fields: dict[str, Any] = {
        "evidence_id": evidence_id,
        "chunk_id": evidence_id,
        "doc_id": f"d-{evidence_id}",
        "text": f"text of {evidence_id}",
        "summary": f"text of {evidence_id}",
        "relevance": 0.9,
        "novelty": 0.8,
        "final_score": 0.85,
        "source_action": Action.DENSE,
        "query": "a query",
        "metadata": {"title": "A title"},
        "added_at_step": 1,
        "vector": vec(0.0, 0.6, 0.8),
    }
    fields.update(overrides)
    return EvidenceItem(**fields)


def make_gap(gap_id: str = "g0", **overrides: Any) -> EvidenceGap:
    fields: dict[str, Any] = {
        "gap_id": gap_id,
        "description": f"description of {gap_id}",
        "query": f"query for {gap_id}",
        "need_type": NeedType.SEMANTIC,
        "created_at_step": 1,
    }
    fields.update(overrides)
    return EvidenceGap(**fields)


def make_state(**overrides: Any) -> RetrievalState:
    fields: dict[str, Any] = {
        "task_id": "t-1",
        "goal": "Which company acquired the startup first?",
        "budget": make_budget(),
        "created_at": NOW,
        "updated_at": NOW,
    }
    fields.update(overrides)
    return RetrievalState(**fields)


def make_cache_entry(**overrides: Any) -> CacheEntry:
    fields: dict[str, Any] = {
        "task_id": "t-1",
        "entry_id": "e1",
        "normalized_query": "query for g0",
        "query_vector": vec(1.0, 0.0, 0.0),
        "need_type": NeedType.SEMANTIC,
        "mode": RetrievalMode.DENSE,
        "candidates": [make_candidate("c1"), make_candidate("c2", rank=1)],
        "accepted_ids": ["c1"],
        "created_at_step": 1,
    }
    fields.update(overrides)
    return CacheEntry(**fields)


def make_scores() -> ActionScores:
    return ActionScores(G=0.71, C=0.375, R=0.0, L=0.06, U=0.069)


def make_decision(**overrides: Any) -> ActionDecision:
    fields: dict[str, Any] = {
        "action": Action.SPARSE_HYBRID,
        "eligible": [Action.DENSE, Action.SPARSE_HYBRID],
        "scores": {Action.DENSE: make_scores(), Action.SPARSE_HYBRID: make_scores()},
        "rule": "utility_argmax",
    }
    fields.update(overrides)
    return ActionDecision(**fields)


def make_update(**overrides: Any) -> EvidenceUpdate:
    fields: dict[str, Any] = {
        "accepted": [make_evidence("c3", added_at_step=2)],
        "rejected": {"c4": "LOW_RELEVANCE"},
        "evicted": [],
        "resolved_gap_ids": ["g0"],
        "gap_resolutions": {"g0": "c1"},
        "item_changes": {"c1": ItemChange(pinned=True, supports_gap_ids=["g0"])},
        "evidence_gain": 0.6,
    }
    fields.update(overrides)
    return EvidenceUpdate(**fields)


def make_plan_result(**overrides: Any) -> PlanResult:
    fields: dict[str, Any] = {
        "mode": PlanMode.INIT,
        "subgoals": [SubgoalPlan(description="d", query="q", need_type=NeedType.MIXED)],
        "referenced_entities": ["Acme"],
        "rationale": "because",
    }
    fields.update(overrides)
    return PlanResult(**fields)


def make_plan_event(step: int = 1, **overrides: Any) -> PlanEvent:
    fields: dict[str, Any] = {
        "step": step,
        "mode": PlanMode.INIT,
        "attempt": 1,
        "status": PlanStatus.ACCEPTED,
    }
    fields.update(overrides)
    return PlanEvent(**fields)


def populated_state() -> RetrievalState:
    """A mid-task state touching every field, with vectors and a non-null current gap."""
    return make_state(
        version=3,
        gaps=[
            make_gap("g0", status=GapStatus.RESOLVED, resolved_by="c1", closed_at_step=2),
            make_gap(
                "g1",
                query="exact entity query",
                need_type=NeedType.LEXICAL,
                created_at_step=1,
            ),
        ],
        current_gap_id="g1",
        needs_initial_plan=False,
        evidence=[
            make_evidence("c1", pinned=True, supports_gap_ids=["g0"]),
            make_evidence("c3", added_at_step=2, source_action=Action.REUSE),
        ],
        query_history=[
            QueryRecord(
                step=1,
                gap_id="g0",
                query="query for g0",
                need_type=NeedType.SEMANTIC,
                source="planner",
            )
        ],
        retrieval_history=[
            RetrievalRecord(
                step=1,
                action=Action.DENSE,
                mode=RetrievalMode.DENSE,
                query="query for g0",
                k=20,
                candidate_ids=["c1", "c2"],
                new_candidate_ids=["c1", "c2"],
                cache_hit=False,
                latency_ms=12.5,
            )
        ],
        budget=make_budget(
            steps_used=3,
            retrieval_calls_used=2,
            llm_calls_used=1,
            tokens_used=900,
            reranker_calls_used=2,
        ),
        seen_candidate_ids=["c1", "c2", "c3"],
        last_result=LastResult(
            step=2,
            query="query for g0",
            mode=RetrievalMode.DENSE,
            depth=20,
            candidate_ids=["c1", "c2"],
            top_relevances=[0.9, 0.45],
        ),
        stagnation_count=1,
        transforms_in_streak=0,
        action_gain_ema={Action.DENSE: 0.2, Action.REUSE: 0.4},
        last_decision=make_decision(),
        last_update=make_update(),
        plan_events=[make_plan_event()],
    )


def sample_instances() -> dict[type[BaseModel], BaseModel]:
    """One valid instance of every shared model (used by the extra='forbid' tests)."""
    state = populated_state()
    entry = make_cache_entry()
    scored = make_scored()
    summary = TraceSummary(
        steps=3,
        retrieval_calls=2,
        cache_hits=1,
        llm_calls=1,
        tokens=900,
        reranker_calls=2,
        total_latency_ms=120.0,
        actions=[Action.DENSE, Action.REUSE, Action.STOP],
    )
    instances: list[BaseModel] = [
        make_candidate(),
        scored,
        make_evidence(),
        make_gap(),
        state.query_history[0],
        state.retrieval_history[0],
        state.last_result,  # type: ignore[list-item]
        make_budget(),
        make_plan_event(),
        SubgoalPlan(description="d", query="q", need_type=NeedType.MIXED),
        PlanLLMOutput(subgoals=[SubgoalPlan(description="d", query="q", need_type="mixed")]),
        make_plan_result(),
        PlanOutcome(result=make_plan_result(), events=[make_plan_event()], llm_calls=1, tokens=50),
        make_scores(),
        make_decision(),
        entry,
        CacheHit(entry=entry, similarity=1.0, match="exact"),
        ItemChange(pinned=True),
        make_update(),
        MonitorVerdict(
            should_continue=True,
            stagnation_count=0,
            transforms_in_streak=0,
            action_gain_ema={Action.DENSE: 0.5},
        ),
        AdvanceResult(current_gap_id="g1", reset_stagnation=True),
        state,
        IterationScratch(
            cache_hit=CacheHit(entry=entry, similarity=0.95, match="semantic"),
            candidates=[make_candidate()],
            scored=[scored],
            evidence_update=make_update(),
            queued_cache_entries=[entry],
        ),
        GraphState(state=state),
        RunContext(run_id="r1", system="controller_on"),
        summary,
        EvidenceBundle(
            task_id="t-1",
            system="controller_on",
            goal=state.goal,
            evidence=state.evidence,
            gaps=state.gaps,
            stop_reason=StopReason.ALL_GAPS_RESOLVED,
            trace_summary=summary,
        ),
        TraceStep(
            run_id="r1",
            task_id="t-1",
            system="controller_on",
            step=1,
            action=Action.DENSE,
            rule="utility_argmax",
            eligible_actions=[Action.DENSE],
            utility_scores={Action.DENSE: make_scores()},
            query="q",
            mode=RetrievalMode.DENSE,
            k=20,
            cache_hit=False,
            candidate_ids=["c1"],
            new_candidate_ids=["c1"],
            accepted_ids=["c1"],
            rejected={},
            evicted_ids=[],
            resolved_gap_ids=[],
            evidence_gain=0.3,
            plan_events=[],
            stagnation_count=0,
            latency_ms={"total": 10.0},
            llm_calls=0,
            tokens=0,
            budget_after=make_budget(),
            timestamp=NOW,
        ),
    ]
    return {type(instance): instance for instance in instances}

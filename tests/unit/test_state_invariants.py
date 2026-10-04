"""check_invariants: one passing and one violating case per STATE_SPEC §5 invariant.

The engine-driven property tests TESTING §4.2 also asks for (many synthetic tasks, the
max_steps exact-iteration test, concurrent steps, ...) need the graph and TaskStore
(Weeks 8-9) and will be added here then.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from engine.config import EngineConfig
from engine.errors import StateInvariantError
from engine.state.invariants import check_invariants
from engine.state.models import (
    Action,
    GapStatus,
    PlanMode,
    QueryRecord,
    RetrievalRecord,
    RetrievalState,
    StopReason,
    TaskStatus,
)
from tests.builders import make_budget, make_evidence, make_gap, make_plan_event, make_state


def active_prev(**overrides: Any) -> RetrievalState:
    fields: dict[str, Any] = {
        "gaps": [make_gap("g0"), make_gap("g1")],
        "current_gap_id": "g0",
        "needs_initial_plan": False,
        "version": 2,
        "budget": make_budget(steps_used=2, retrieval_calls_used=1, tokens_used=500),
    }
    fields.update(overrides)
    return make_state(**fields)


def next_state(prev: RetrievalState, **changes: Any) -> RetrievalState:
    """What persist would save: version + 1, steps_used + 1, then the given changes."""
    data = prev.model_dump(exclude={"current_subgoal", "pending_query", "current_need_type"})
    data.update(version=prev.version + 1, budget=prev.budget.consume(steps=1))
    data.update(changes)
    return RetrievalState.model_validate(data)


def assert_violates(prev: RetrievalState, new: RetrievalState, cfg: EngineConfig, tag: str) -> None:
    with pytest.raises(StateInvariantError, match=tag.replace("[", r"\[").replace("]", r"\]")):
        check_invariants(prev, new, cfg)


def stopped(reason: StopReason) -> dict[str, Any]:
    return {"status": TaskStatus.STOPPED, "stop_reason": reason}


def record(step: int, query: str = "q") -> RetrievalRecord:
    return RetrievalRecord(
        step=step,
        action=Action.DENSE,
        mode="dense",
        query=query,
        k=20,
        candidate_ids=[],
        new_candidate_ids=[],
        cache_hit=False,
        latency_ms=1.0,
    )


def test_valid_iteration_passes(cfg: EngineConfig) -> None:
    prev = active_prev()
    check_invariants(prev, next_state(prev), cfg)


def test_error_carries_task_id_and_lists_all_violations(cfg: EngineConfig) -> None:
    prev = active_prev()
    new = next_state(prev, version=prev.version + 5, current_gap_id=None)
    with pytest.raises(StateInvariantError) as info:
        check_invariants(prev, new, cfg)
    assert info.value.task_id == prev.task_id
    assert "[inv 6]" in str(info.value)
    assert "[inv 8]" in str(info.value)


# ---------------------------------------------------------------------------- inv 1
@pytest.mark.parametrize(
    ("used", "cap"),
    [("retrieval_calls_used", "max_retrieval_calls"), ("llm_calls_used", "max_llm_calls")],
)
def test_inv1_counter_at_cap_passes_and_above_cap_fails(
    cfg: EngineConfig, used: str, cap: str
) -> None:
    prev = active_prev()
    limit = getattr(prev.budget, cap)
    at_cap = prev.budget.consume(steps=1).model_copy(update={used: limit})
    over = prev.budget.consume(steps=1).model_copy(update={used: limit + 1})
    check_invariants(prev, next_state(prev, budget=at_cap), cfg)
    assert_violates(prev, next_state(prev, budget=over), cfg, "[inv 1]")


def test_inv1_steps_above_max_steps_fails(cfg: EngineConfig) -> None:
    # Reaching max_steps exactly is covered by the inv 10 cases below.
    prev = active_prev(budget=make_budget(max_steps=3, steps_used=3))
    assert_violates(prev, next_state(prev), cfg, "[inv 1]")


def test_inv1_token_overshoot_by_stopping_iteration_passes(cfg: EngineConfig) -> None:
    prev = active_prev(budget=make_budget(steps_used=2, tokens_used=19_900))
    budget = prev.budget.consume(steps=1, tokens=400)
    check_invariants(prev, next_state(prev, budget=budget, **stopped(StopReason.MAX_TOKENS)), cfg)


def test_inv1_token_overshoot_while_still_active_fails(cfg: EngineConfig) -> None:
    prev = active_prev(budget=make_budget(steps_used=2, tokens_used=19_900))
    budget = prev.budget.consume(steps=1, tokens=400)
    assert_violates(prev, next_state(prev, budget=budget), cfg, "[inv 1]")


def test_inv1_token_overshoot_already_before_iteration_fails(cfg: EngineConfig) -> None:
    prev = active_prev(budget=make_budget(steps_used=2, tokens_used=20_000))
    budget = prev.budget.consume(steps=1, tokens=400)
    new = next_state(prev, budget=budget, **stopped(StopReason.MAX_TOKENS))
    assert_violates(prev, new, cfg, "more than one iteration")


# ---------------------------------------------------------------------------- inv 2
def test_inv2_unique_evidence_within_cap_passes(cfg: EngineConfig) -> None:
    prev = active_prev()
    evidence = [make_evidence(f"c{i}") for i in range(cfg.evidence.max_items)]
    check_invariants(prev, next_state(prev, evidence=evidence), cfg)


def test_inv2_duplicate_evidence_id_fails(cfg: EngineConfig) -> None:
    prev = active_prev()
    new = next_state(prev, evidence=[make_evidence("c1"), make_evidence("c1")])
    assert_violates(prev, new, cfg, "[inv 2]")


def test_inv2_too_many_evidence_items_fails(cfg: EngineConfig) -> None:
    prev = active_prev()
    evidence = [make_evidence(f"c{i}") for i in range(cfg.evidence.max_items + 1)]
    assert_violates(prev, next_state(prev, evidence=evidence), cfg, "[inv 2]")


# ---------------------------------------------------------------------------- inv 3
HISTORIES = {
    "retrieval_history": lambda i: record(i, query=f"q{i}"),
    "query_history": lambda i: QueryRecord(
        step=i, gap_id="g0", query=f"q{i}", need_type="semantic", source="planner"
    ),
    "plan_events": lambda i: make_plan_event(step=i, mode=PlanMode.REWRITE),
}


@pytest.mark.parametrize("name", list(HISTORIES))
def test_inv3_appending_passes(cfg: EngineConfig, name: str) -> None:
    make = HISTORIES[name]
    prev = active_prev(**{name: [make(1)]})
    check_invariants(prev, next_state(prev, **{name: [make(1), make(2)]}), cfg)


@pytest.mark.parametrize("name", list(HISTORIES))
@pytest.mark.parametrize("change", ["truncate", "rewrite"])
def test_inv3_removing_or_rewriting_fails(cfg: EngineConfig, name: str, change: str) -> None:
    make = HISTORIES[name]
    prev = active_prev(**{name: [make(1), make(2)]})
    after = [make(1)] if change == "truncate" else [make(1), make(3), make(4)]
    assert_violates(prev, next_state(prev, **{name: after}), cfg, "[inv 3]")


# ---------------------------------------------------------------------------- inv 4
def test_inv4_resolved_gap_backed_by_pinned_item_passes(cfg: EngineConfig) -> None:
    prev = active_prev()
    new = next_state(
        prev,
        gaps=[make_gap("g0", status=GapStatus.RESOLVED, resolved_by="c1"), make_gap("g1")],
        current_gap_id="g1",
        evidence=[make_evidence("c1", pinned=True)],
    )
    check_invariants(prev, new, cfg)


@pytest.mark.parametrize(("resolved_by", "pinned"), [("c1", False), ("c9", True), (None, True)])
def test_inv4_resolved_gap_without_pinned_item_fails(
    cfg: EngineConfig, resolved_by: str | None, pinned: bool
) -> None:
    prev = active_prev()
    new = next_state(
        prev,
        gaps=[make_gap("g0", status=GapStatus.RESOLVED, resolved_by=resolved_by), make_gap("g1")],
        current_gap_id="g1",
        evidence=[make_evidence("c1", pinned=pinned)],
    )
    assert_violates(prev, new, cfg, "[inv 4]")


# ---------------------------------------------------------------------------- inv 5
def test_inv5_stopping_with_a_reason_passes(cfg: EngineConfig) -> None:
    prev = active_prev()
    check_invariants(prev, next_state(prev, **stopped(StopReason.LOW_MARGINAL_UTILITY)), cfg)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": TaskStatus.STOPPED, "stop_reason": None},
        {"status": TaskStatus.ACTIVE, "stop_reason": StopReason.STAGNATION},
    ],
)
def test_inv5_status_and_stop_reason_must_agree(cfg: EngineConfig, changes: dict[str, Any]) -> None:
    prev = active_prev()
    assert_violates(prev, next_state(prev, **changes), cfg, "[inv 5]")


def test_inv5_stop_reason_set_only_once(cfg: EngineConfig) -> None:
    prev = active_prev(**stopped(StopReason.STAGNATION))
    new = next_state(prev, **stopped(StopReason.LOW_MARGINAL_UTILITY))
    assert_violates(prev, new, cfg, "stop_reason changed")


# ---------------------------------------------------------------------------- inv 6
def test_inv6_needs_initial_plan_without_gap_passes(cfg: EngineConfig) -> None:
    prev = make_state(version=0)
    check_invariants(prev, next_state(prev), cfg)


def test_inv6_stopped_task_may_have_no_current_gap(cfg: EngineConfig) -> None:
    prev = active_prev()
    new = next_state(prev, current_gap_id=None, **stopped(StopReason.STAGNATION))
    check_invariants(prev, new, cfg)


@pytest.mark.parametrize(
    "changes",
    [
        {"current_gap_id": None},
        {"current_gap_id": "missing"},
        {"gaps": [make_gap("g0", status=GapStatus.EXHAUSTED), make_gap("g1")]},
    ],
)
def test_inv6_active_task_needs_an_open_current_gap(
    cfg: EngineConfig, changes: dict[str, Any]
) -> None:
    prev = active_prev()
    assert_violates(prev, next_state(prev, **changes), cfg, "[inv 6]")


# ---------------------------------------------------------------------------- inv 7
def test_inv7_live_gaps_at_cap_passes(cfg: EngineConfig) -> None:
    prev = active_prev()
    gaps = [make_gap(f"g{i}") for i in range(cfg.planner.max_gaps)]
    gaps.append(make_gap("gs", status=GapStatus.SUPERSEDED))  # not live: not counted
    check_invariants(prev, next_state(prev, gaps=gaps), cfg)


def test_inv7_too_many_live_gaps_fails(cfg: EngineConfig) -> None:
    prev = active_prev()
    gaps = [make_gap(f"g{i}") for i in range(cfg.planner.max_gaps + 1)]
    assert_violates(prev, next_state(prev, gaps=gaps), cfg, "[inv 7]")


# ---------------------------------------------------------------------------- inv 8
@pytest.mark.parametrize("delta", [0, 2, -1])
def test_inv8_version_must_increase_by_exactly_one(cfg: EngineConfig, delta: int) -> None:
    prev = active_prev()
    assert_violates(prev, next_state(prev, version=prev.version + delta), cfg, "[inv 8]")


# --------------------------------------------------------------------------- inv 10
@pytest.mark.parametrize(
    "reason", [StopReason.MAX_STEPS, StopReason.ALL_GAPS_RESOLVED, StopReason.STAGNATION]
)
def test_inv10_reaching_max_steps_with_monitor_stop_passes(
    cfg: EngineConfig, reason: StopReason
) -> None:
    prev = active_prev(budget=make_budget(max_steps=3, steps_used=2))
    new = next_state(prev, current_gap_id=None, **stopped(reason))
    assert new.budget.steps_used == new.budget.max_steps
    check_invariants(prev, new, cfg)


@pytest.mark.parametrize(
    "changes",
    [
        {},  # still active
        stopped(StopReason.LOW_MARGINAL_UTILITY),  # MAX_STEPS takes precedence (§6.4)
        stopped(StopReason.MAX_TOKENS),
    ],
)
def test_inv10_reaching_max_steps_otherwise_fails(
    cfg: EngineConfig, changes: dict[str, Any]
) -> None:
    prev = active_prev(budget=make_budget(max_steps=3, steps_used=2))
    assert_violates(prev, next_state(prev, **changes), cfg, "[inv 10]")


# ---------------------------------------------------------------------- immutable fields
@pytest.mark.parametrize(
    "changes",
    [
        {"task_id": "t-other"},
        {"goal": "another goal"},
        {"created_at": make_state().created_at + timedelta(seconds=1)},
    ],
)
def test_immutable_fields_must_not_change(cfg: EngineConfig, changes: dict[str, Any]) -> None:
    prev = active_prev()
    assert_violates(prev, next_state(prev, **changes), cfg, "[immutable]")


def test_budget_caps_must_not_change(cfg: EngineConfig) -> None:
    prev = active_prev()
    budget = prev.budget.consume(steps=1).model_copy(update={"max_retrieval_calls": 9})
    assert_violates(prev, next_state(prev, budget=budget), cfg, "[immutable]")

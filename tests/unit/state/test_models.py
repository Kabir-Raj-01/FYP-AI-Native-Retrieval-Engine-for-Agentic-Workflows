"""engine/state/models.py: shape rules from STATE_SPEC.md (as amended by ADR-009)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import pytest
from pydantic import BaseModel, ValidationError

from engine.state import models
from engine.state.models import (
    Action,
    GraphState,
    MonitorVerdict,
    NeedType,
    PlanLLMOutput,
    PlanResult,
    RetrievalState,
    SubgoalPlan,
)
from engine.state.serialization import dump_with_vectors
from tests.builders import NOW, make_gap, make_state, populated_state, sample_instances

MODEL_CLASSES = [
    obj
    for name in models.__all__
    if isinstance(obj := getattr(models, name), type) and issubclass(obj, BaseModel)
]
ENUM_CLASSES = [
    obj
    for name in models.__all__
    if isinstance(obj := getattr(models, name), type) and issubclass(obj, StrEnum)
]
COMPUTED = {"current_subgoal", "pending_query", "current_need_type"}


def test_every_model_has_a_sample_instance() -> None:
    assert set(sample_instances()) == set(MODEL_CLASSES)


def test_every_enum_is_a_str_enum() -> None:
    assert len(ENUM_CLASSES) == 8
    for enum_cls in ENUM_CLASSES:
        for member in enum_cls:
            assert isinstance(member, str)
            assert str(member) == member.value


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_extra_forbid_rejects_unknown_field(cls: type[BaseModel]) -> None:
    payload = dump_with_vectors(sample_instances()[cls])
    assert cls.model_validate(payload) is not None  # the valid payload is accepted ...
    with pytest.raises(ValidationError, match="unexpected_field"):
        cls.model_validate({**payload, "unexpected_field": 1})  # ... an extra key is not


def test_retrieval_state_has_no_step_or_current_depth_field() -> None:
    assert "step" not in RetrievalState.model_fields
    assert "current_depth" not in RetrievalState.model_fields
    assert COMPUTED.isdisjoint(RetrievalState.model_fields)
    assert set(RetrievalState.model_computed_fields) == COMPUTED


def test_monitor_verdict_uses_should_continue() -> None:
    assert "should_continue" in MonitorVerdict.model_fields
    assert "continue_" not in MonitorVerdict.model_fields


def test_plan_llm_output_has_no_mode_and_needs_one_subgoal() -> None:
    assert "mode" not in PlanLLMOutput.model_fields
    with pytest.raises(ValidationError):
        PlanLLMOutput(subgoals=[])


def test_plan_models_carry_no_length_limits() -> None:
    # Count/length limits are config rules in the planner validator (ADR-009 §10).
    long = "x" * 10_000
    subgoals = [SubgoalPlan(description=long, query=long, need_type=NeedType.MIXED)] * 50
    result = PlanResult(mode="split", subgoals=subgoals, rationale=long)
    assert len(result.subgoals) == 50


# ---------------------------------------------------------------------- computed fields
def test_computed_fields_follow_current_gap_id() -> None:
    state = make_state(
        gaps=[
            make_gap("g0", description="first", query="q0", need_type=NeedType.SEMANTIC),
            make_gap("g1", description="second", query="q1", need_type=NeedType.LEXICAL),
        ],
        current_gap_id="g0",
    )
    assert (state.current_subgoal, state.pending_query, state.current_need_type) == (
        "first",
        "q0",
        NeedType.SEMANTIC,
    )
    state.current_gap_id = "g1"
    assert (state.current_subgoal, state.pending_query, state.current_need_type) == (
        "second",
        "q1",
        NeedType.LEXICAL,
    )


@pytest.mark.parametrize("gap_id", [None, "missing"])
def test_computed_fields_fall_back_without_a_current_gap(gap_id: str | None) -> None:
    state = make_state(gaps=[make_gap("g0")], current_gap_id=gap_id)
    assert state.current_gap is None
    assert (state.current_subgoal, state.pending_query, state.current_need_type) == (
        "",
        "",
        NeedType.MIXED,
    )


def test_rewrite_of_gap_query_is_seen_through_pending_query() -> None:
    state = make_state(gaps=[make_gap("g0", query="old")], current_gap_id="g0")
    gaps = [state.gaps[0].model_copy(update={"query": "new", "need_type": NeedType.LEXICAL})]
    state.gaps = gaps
    assert state.pending_query == "new"
    assert state.current_need_type == NeedType.LEXICAL


@pytest.mark.parametrize("name", sorted(COMPUTED))
def test_computed_fields_cannot_be_assigned(name: str) -> None:
    state = make_state(gaps=[make_gap("g0")], current_gap_id="g0")
    with pytest.raises(AttributeError, match="has no setter"):  # verified on pydantic 2.13
        setattr(state, name, "x")


@pytest.mark.parametrize("name", sorted(COMPUTED))
def test_computed_fields_cannot_be_passed_to_the_constructor(name: str) -> None:
    with pytest.raises(ValidationError, match=name):
        RetrievalState.model_validate({**make_state().model_dump(exclude=COMPUTED), name: "x"})


def test_plain_dump_includes_computed_fields() -> None:
    dumped = populated_state().model_dump()
    assert dumped.keys() >= COMPUTED
    assert dumped["pending_query"] == "exact entity query"


# ------------------------------------------------------------------- assignment / dates
def test_validate_assignment_rejects_wrong_type() -> None:
    state = make_state()
    with pytest.raises(ValidationError):
        state.version = "not-an-int"  # type: ignore[assignment]


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        make_state(created_at=datetime(2026, 10, 4, 12, 0))


def test_aware_datetime_accepted() -> None:
    assert make_state(created_at=NOW).created_at.tzinfo is not None


def test_graph_state_default_scratch_is_not_shared() -> None:
    a = GraphState(state=make_state())
    b = GraphState(state=make_state())
    a.scratch.candidates.append(sample_instances()[models.Candidate])  # type: ignore[arg-type]
    assert b.scratch.candidates == []


def test_mutable_defaults_are_not_shared_between_instances() -> None:
    a = make_state()
    b = make_state()
    a.seen_candidate_ids.append("c1")
    a.action_gain_ema[Action.DENSE] = 0.1
    assert b.seen_candidate_ids == []
    assert b.action_gain_ema == {}

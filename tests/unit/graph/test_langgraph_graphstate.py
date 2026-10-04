"""Characterisation of LangGraph with the nested GraphState (ADR-009 §2 reversal condition).

These tests pin LangGraph behaviour the graph design relies on, verified on langgraph
1.2.12 (2026-10-05); a library upgrade that changes any of it fails here first.
Findings (LIBRARY_NOTES "LangGraph"; ADR-009 addendum A7):
- each top-level GraphState field (`state`, `scratch`) is one channel; a node receives
  `GraphState(**channels)` with typed nested models, computed fields included;
- a returned field REPLACES that field wholesale (no deep merge) -> return full values,
  e.g. `scratch.model_copy(update=...)`;
- returned values are validated only when the next node's input is built; a bad value
  returned by the last node reaches invoke()'s output unvalidated;
- returned keys that are not GraphState fields are silently dropped;
- objects pass between nodes by reference, without copy or revalidation — starting with
  the caller's own input objects — so in-place mutation leaks to later nodes and to the
  caller (STATE_SPEC §5 inv 9 is a process rule LangGraph does not enforce);
- invoke() returns a dict of field values, not a GraphState.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from engine.state.models import (
    GraphState,
    IterationScratch,
    NeedType,
    RetrievalState,
    StopReason,
    TaskStatus,
)
from tests.builders import make_candidate, make_gap, make_scored, make_state

Node = Callable[[GraphState], dict[str, Any]]


def linear_graph(nodes: Sequence[tuple[str, Node]]) -> Any:
    g = StateGraph(GraphState)
    for name, fn in nodes:
        g.add_node(name, fn)
    for a, b in itertools.pairwise([START, *(name for name, _ in nodes), END]):
        g.add_edge(a, b)
    return g.compile()


def initial() -> GraphState:
    gap = make_gap("g0", query="KX-41 drone", need_type=NeedType.LEXICAL)
    return GraphState(state=make_state(gaps=[gap], current_gap_id="g0"))


def test_nodes_receive_typed_nested_models_with_computed_fields() -> None:
    seen: dict[str, Any] = {}

    def node(gs: GraphState) -> dict[str, Any]:
        seen.update(
            types=(type(gs), type(gs.state), type(gs.scratch)),
            pending=gs.state.pending_query,
            need=gs.state.current_need_type,
        )
        return {}

    linear_graph([("n", node)]).invoke(initial())
    assert seen["types"] == (GraphState, RetrievalState, IterationScratch)
    assert (seen["pending"], seen["need"]) == ("KX-41 drone", NeedType.LEXICAL)


def test_invoke_returns_a_dict_of_field_values() -> None:
    out = linear_graph([("n", lambda gs: {})]).invoke(initial())
    assert isinstance(out, dict)
    assert set(out) == {"state", "scratch"}
    assert GraphState.model_validate(out).state.task_id == initial().state.task_id


def test_returned_field_replaces_the_whole_value_without_merging() -> None:
    seen: dict[str, Any] = {}

    def execute(gs: GraphState) -> dict[str, Any]:
        return {"scratch": gs.scratch.model_copy(update={"candidates": [make_candidate("c9")]})}

    def partial_dict(gs: GraphState) -> dict[str, Any]:
        return {"scratch": {"scored": [make_scored("c9")]}}  # only one scratch field

    def check(gs: GraphState) -> dict[str, Any]:
        seen["candidates"] = gs.scratch.candidates
        seen["scored"] = [s.candidate.chunk_id for s in gs.scratch.scored]
        return {}

    linear_graph([("execute", execute), ("partial", partial_dict), ("check", check)]).invoke(
        initial()
    )
    assert seen["candidates"] == []  # lost: the dict replaced scratch, it was not merged
    assert seen["scored"] == ["c9"]


def test_model_copy_update_keeps_the_other_scratch_fields() -> None:
    seen: dict[str, Any] = {}

    def execute(gs: GraphState) -> dict[str, Any]:
        return {"scratch": gs.scratch.model_copy(update={"candidates": [make_candidate("c9")]})}

    def score(gs: GraphState) -> dict[str, Any]:
        return {"scratch": gs.scratch.model_copy(update={"scored": [make_scored("c9")]})}

    def check(gs: GraphState) -> dict[str, Any]:
        seen["candidates"] = [c.chunk_id for c in gs.scratch.candidates]
        return {}

    linear_graph([("execute", execute), ("score", score), ("check", check)]).invoke(initial())
    assert seen["candidates"] == ["c9"]


def test_loop_with_conditional_edge_and_scratch_reset_in_persist() -> None:
    seen: dict[str, list[int]] = {"scratch_at_plan": [], "iterations": []}

    def plan(gs: GraphState) -> dict[str, Any]:
        seen["scratch_at_plan"].append(len(gs.scratch.candidates))
        return {"scratch": gs.scratch.model_copy(update={"candidates": [make_candidate()]})}

    def monitor(gs: GraphState) -> dict[str, Any]:
        seen["iterations"].append(1)
        if len(seen["iterations"]) < 3:
            return {}
        stopped = gs.state.model_copy(
            update={"status": TaskStatus.STOPPED, "stop_reason": StopReason.MAX_STEPS}
        )
        return {"state": stopped}

    def persist(gs: GraphState) -> dict[str, Any]:
        return {"scratch": IterationScratch()}  # scratch is discarded every iteration

    g = StateGraph(GraphState)
    g.add_node("plan", plan)
    g.add_node("monitor", monitor)
    g.add_node("persist", persist)
    g.add_edge(START, "plan")
    g.add_edge("plan", "monitor")
    g.add_edge("monitor", "persist")
    g.add_conditional_edges(
        "persist", lambda gs: END if gs.state.status != TaskStatus.ACTIVE else "plan"
    )
    out = g.compile().invoke(initial(), config={"recursion_limit": 20})
    assert len(seen["iterations"]) == 3
    assert seen["scratch_at_plan"] == [0, 0, 0]  # every iteration starts with fresh scratch
    assert out["state"].stop_reason == StopReason.MAX_STEPS


def test_unknown_update_keys_are_silently_dropped() -> None:
    out = linear_graph([("n", lambda gs: {"scrach": IterationScratch(), "typo": 1})]).invoke(
        initial()
    )
    assert set(out) == {"state", "scratch"}  # no error: the typo is simply ignored


def test_invalid_update_is_rejected_when_the_next_node_reads_it() -> None:
    nodes: list[tuple[str, Node]] = [("bad", lambda gs: {"state": "x"}), ("next", lambda gs: {})]
    with pytest.raises(ValidationError, match="state"):
        linear_graph(nodes).invoke(initial())


def test_invalid_update_from_the_last_node_reaches_the_output_unvalidated() -> None:
    out = linear_graph([("bad", lambda gs: {"state": "not a state"})]).invoke(initial())
    assert out["state"] == "not a state"
    with pytest.raises(ValidationError):
        GraphState.model_validate(out)  # so the Engine must validate invoke() output


def test_objects_pass_by_reference_and_in_place_mutation_leaks() -> None:
    seen: dict[str, Any] = {}

    def a(gs: GraphState) -> dict[str, Any]:
        new = gs.state.model_copy(update={"stagnation_count": 1})
        seen["returned"] = new
        return {"state": new}

    def b(gs: GraphState) -> dict[str, Any]:
        seen["received"] = gs.state
        gs.state.transforms_in_streak = 7  # violates STATE_SPEC inv 9; nothing returned
        return {}

    def c(gs: GraphState) -> dict[str, Any]:
        seen["c_sees"] = gs.state.transforms_in_streak
        return {}

    out = linear_graph([("a", a), ("b", b), ("c", c)]).invoke(initial())
    assert seen["received"] is seen["returned"]  # no copy, no revalidation
    assert seen["c_sees"] == 7
    assert out["state"].transforms_in_streak == 7


def test_nodes_receive_the_callers_own_objects_so_mutation_reaches_the_input() -> None:
    # GraphState(state=s) keeps `s` itself (pydantic revalidate_instances="never"), and
    # invoke() hands that object to the first node. The `prev` state given to
    # check_invariants must therefore be an independent deep copy, or a node's in-place
    # mutation would also change `prev` and hide the violation.
    loaded = make_state(gaps=[make_gap("g0")], current_gap_id="g0")
    start = GraphState(state=loaded)
    assert start.state is loaded
    seen: dict[str, bool] = {}

    def a(gs: GraphState) -> dict[str, Any]:
        seen["same_object"] = gs.state is loaded
        gs.state.stagnation_count = 5
        return {}

    linear_graph([("a", a)]).invoke(start)
    assert seen["same_object"]
    assert loaded.stagnation_count == 5


def test_a_deep_copy_taken_before_invoke_is_isolated() -> None:
    loaded = make_state(gaps=[make_gap("g0")], current_gap_id="g0")
    prev = loaded.model_copy(deep=True)

    def a(gs: GraphState) -> dict[str, Any]:
        gs.state.stagnation_count = 5
        gs.state.gaps.append(make_gap("g1"))
        return {}

    linear_graph([("a", a)]).invoke(GraphState(state=loaded))
    assert prev.stagnation_count == 0
    assert [g.gap_id for g in prev.gaps] == ["g0"]

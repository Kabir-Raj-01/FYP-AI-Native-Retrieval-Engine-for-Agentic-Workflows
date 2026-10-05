"""engine/state/serialization.py round trips and vector visibility (TESTING §4.2)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from engine.state.models import (
    CacheEntry,
    CacheHit,
    Candidate,
    DirectResult,
    EvidenceBundle,
    EvidenceItem,
    EvidenceUpdate,
    GraphState,
    IterationScratch,
    NeedType,
    RetrievalState,
    ScoredCandidate,
)
from engine.state.serialization import dump_with_vectors, load_with_vectors
from tests.builders import make_state, populated_state, sample_instances

VECTOR_KEYS = {"vector", "query_vector"}
COMPUTED = {"current_subgoal", "pending_query", "current_need_type"}
VECTOR_MODELS = [
    Candidate,
    ScoredCandidate,
    EvidenceItem,
    CacheEntry,
    CacheHit,
    EvidenceUpdate,
    IterationScratch,
    RetrievalState,
    GraphState,
    EvidenceBundle,
    DirectResult,
]


def vector_values(node: Any) -> Iterator[Any]:
    """Every value stored under a vector key, anywhere in a dumped structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in VECTOR_KEYS:
                yield value
            else:
                yield from vector_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from vector_values(item)


def all_keys(node: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(key)
            keys |= all_keys(value)
    elif isinstance(node, list):
        for item in node:
            keys |= all_keys(item)
    return keys


@pytest.mark.parametrize("cls", VECTOR_MODELS, ids=lambda c: c.__name__)
def test_round_trip_preserves_every_model_with_vectors(cls: type[BaseModel]) -> None:
    model = sample_instances()[cls]
    restored = load_with_vectors(cls, dump_with_vectors(model))
    assert restored == model
    assert any(v is not None for v in vector_values(dump_with_vectors(model)))


def test_populated_state_round_trips_field_for_field() -> None:
    state = populated_state()
    restored = load_with_vectors(RetrievalState, dump_with_vectors(state))
    assert restored == state
    assert restored.model_dump() == state.model_dump()
    assert restored.evidence[0].vector == state.evidence[0].vector
    assert (restored.current_subgoal, restored.pending_query, restored.current_need_type) == (
        state.current_subgoal,
        state.pending_query,
        state.current_need_type,
    )


def test_round_trip_without_current_gap_falls_back() -> None:
    state = make_state(current_gap_id=None)
    restored = load_with_vectors(RetrievalState, dump_with_vectors(state))
    assert restored == state
    assert (restored.current_subgoal, restored.pending_query, restored.current_need_type) == (
        "",
        "",
        NeedType.MIXED,
    )


def test_default_dumps_contain_no_vector_data() -> None:
    state = populated_state()
    python_dump = state.model_dump()
    json_dump = json.loads(state.model_dump_json())
    for dumped in (python_dump, json_dump):
        values = list(vector_values(dumped))
        assert values  # the keys are present (serialized as null) ...
        assert all(v is None for v in values)  # ... but carry no data


def test_default_dump_of_cache_entry_hides_query_vector() -> None:
    entry = sample_instances()[CacheEntry]
    assert entry.model_dump()["query_vector"] is None
    assert json.loads(entry.model_dump_json())["query_vector"] is None


def test_persisted_dump_keeps_vectors() -> None:
    dumped = dump_with_vectors(populated_state())
    values = list(vector_values(dumped))
    assert values
    assert all(isinstance(v, list) and v for v in values)


def test_persisted_state_dump_excludes_computed_fields() -> None:
    assert COMPUTED.isdisjoint(dump_with_vectors(populated_state()))
    assert populated_state().model_dump().keys() >= COMPUTED


def test_nested_state_dump_excludes_computed_fields() -> None:
    dumped = dump_with_vectors(GraphState(state=populated_state()))
    assert COMPUTED.isdisjoint(all_keys(dumped))


def test_plain_dump_cannot_be_reloaded_which_is_why_the_helper_exists() -> None:
    with pytest.raises(ValueError, match="pending_query"):
        RetrievalState.model_validate(populated_state().model_dump())


def test_persisted_dump_is_json_serialisable() -> None:
    text = json.dumps(dump_with_vectors(populated_state()))
    restored = load_with_vectors(RetrievalState, json.loads(text))
    assert restored == populated_state()

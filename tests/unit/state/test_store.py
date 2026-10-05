"""engine/state/store.py: SqliteTaskStore (ARCHITECTURE §3, §7; ADR-009 §9; TESTING §4.2)."""

from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path

import pytest

from engine.config import EngineConfig
from engine.errors import TaskConflict, TaskExists, TaskNotFound
from engine.state.models import RetrievalState, StopReason, TaskStatus
from engine.state.sqlite import connect, read_cache_entries
from engine.state.store import SqliteTaskStore
from tests.builders import NOW, make_budget, make_cache_entry, populated_state


@pytest.fixture
def store(cfg: EngineConfig, tmp_state_db: Path) -> SqliteTaskStore:
    return SqliteTaskStore(cfg, tmp_state_db, now=lambda: NOW)


def next_version(state: RetrievalState, **updates: object) -> RetrievalState:
    """What ``persist`` hands to ``save``: version + 1 and a later ``updated_at``."""
    return state.model_copy(
        update={"version": state.version + 1, "updated_at": NOW + timedelta(seconds=1), **updates}
    )


def test_create_persists_s0_with_fresh_counters(store: SqliteTaskStore) -> None:
    budget = make_budget(max_steps=5, steps_used=2, retrieval_calls_used=1)
    state = store.create("goal text", budget, task_id="multihop:0001")
    assert (state.task_id, state.goal, state.version) == ("multihop:0001", "goal text", 0)
    assert (state.budget.max_steps, state.budget.steps_used) == (5, 0)
    assert state.budget.retrieval_calls_used == 0
    assert (state.created_at, state.updated_at, state.status) == (NOW, NOW, TaskStatus.ACTIVE)
    assert state.needs_initial_plan is True
    assert store.load("multihop:0001") == state


def test_generated_ids_are_unique(store: SqliteTaskStore) -> None:
    ids = {store.create("g", make_budget()).task_id for _ in range(3)}
    assert len(ids) == 3


def test_duplicate_task_id_raises_task_exists(store: SqliteTaskStore) -> None:
    store.create("first", make_budget(), task_id="t-1")
    with pytest.raises(TaskExists) as exc:
        store.create("second", make_budget(), task_id="t-1")
    assert exc.value.task_id == "t-1"
    assert store.load("t-1").goal == "first"


def test_load_unknown_task_raises(store: SqliteTaskStore) -> None:
    with pytest.raises(TaskNotFound):
        store.load("nope")


def test_save_round_trips_a_populated_state_with_vectors(store: SqliteTaskStore) -> None:
    store.create("Which company acquired the startup first?", make_budget(), task_id="t-1")
    rich = populated_state()  # task t-1, version 3, gaps/evidence/vectors/current gap
    rich = rich.model_copy(update={"version": 1})
    store.save(rich, expected_version=0)
    loaded = store.load("t-1")
    assert loaded == rich
    assert loaded.evidence[0].vector == rich.evidence[0].vector
    assert loaded.pending_query == rich.pending_query  # computed property recomputed


def test_save_with_stale_version_writes_nothing(store: SqliteTaskStore, tmp_state_db: Path) -> None:
    s0 = store.create("g", make_budget(), task_id="t-1")
    s1 = next_version(s0, stagnation_count=1)
    store.save(s1, expected_version=0)
    stale = next_version(s0, stagnation_count=7)  # built from v0 by a second writer
    with pytest.raises(TaskConflict, match="stored version 1, expected 0"):
        store.save(stale, expected_version=0, cache_entries=[make_cache_entry(task_id="t-1")])
    assert store.load("t-1") == s1
    assert read_cache_entries(connect(tmp_state_db), "t-1") == []  # rolled back with the state


def test_save_unknown_task_raises_not_found(store: SqliteTaskStore) -> None:
    with pytest.raises(TaskNotFound):
        store.save(next_version(populated_state()), expected_version=3)


def test_save_commits_cache_entries_with_the_state(
    store: SqliteTaskStore, tmp_state_db: Path
) -> None:
    s0 = store.create("g", make_budget(), task_id="t-1")
    entries = [make_cache_entry(task_id="t-1", entry_id="e1")]
    store.save(next_version(s0), expected_version=0, cache_entries=entries)
    assert read_cache_entries(connect(tmp_state_db), "t-1") == entries


def test_save_applies_the_cache_eviction_rule(
    cfg: EngineConfig, store: SqliteTaskStore, tmp_state_db: Path
) -> None:
    cfg.cache.max_entries_per_task = 2
    s0 = store.create("g", make_budget(), task_id="t-1")
    entries = [
        make_cache_entry(
            task_id="t-1", entry_id=f"e{i}", normalized_query=f"q{i}", created_at_step=i
        )
        for i in range(3)
    ]
    store.save(next_version(s0), expected_version=0, cache_entries=entries)
    kept = [e.entry_id for e in read_cache_entries(connect(tmp_state_db), "t-1")]
    assert kept == ["e1", "e2"]  # oldest dropped (CACHE_SPEC §4)


def test_save_refuses_cache_entries_of_another_task(
    store: SqliteTaskStore, tmp_state_db: Path
) -> None:
    s0 = store.create("g", make_budget(), task_id="t-1")
    store.create("g", make_budget(), task_id="t-2")
    with pytest.raises(ValueError, match="other tasks"):
        store.save(next_version(s0), 0, cache_entries=[make_cache_entry(task_id="t-2")])
    assert store.load("t-1") == s0
    assert read_cache_entries(connect(tmp_state_db), "t-2") == []


def test_status_column_follows_the_state(store: SqliteTaskStore, tmp_state_db: Path) -> None:
    s0 = store.create("g", make_budget(), task_id="t-1")
    stopped = next_version(s0, status=TaskStatus.STOPPED, stop_reason=StopReason.MAX_STEPS)
    store.save(stopped, expected_version=0)
    row = connect(tmp_state_db).execute("SELECT version, status FROM tasks").fetchone()
    assert row == (1, "stopped")


def test_concurrent_writers_from_the_same_version_one_wins(store: SqliteTaskStore) -> None:
    s0 = store.create("g", make_budget(), task_id="t-1")
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def writer(n: int) -> None:
        state = next_version(store.load("t-1"), stagnation_count=n)
        barrier.wait()
        try:
            store.save(state, expected_version=s0.version)
            outcomes.append("saved")
        except TaskConflict:
            outcomes.append("conflict")

    threads = [threading.Thread(target=writer, args=(n,)) for n in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["conflict", "saved"]
    assert store.load("t-1").version == 1


def test_unknown_schema_version_is_rejected(store: SqliteTaskStore) -> None:
    s0 = store.create("g", make_budget(), task_id="t-1")
    store.save(next_version(s0, schema_version=2), expected_version=0)
    with pytest.raises(ValueError, match="schema_version 2"):
        store.load("t-1")

"""engine/state/sqlite.py: schema, WAL, foreign keys, cache-entry write rules."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from engine.state.models import CacheEntry, RetrievalMode
from engine.state.serialization import dump_with_vectors
from engine.state.sqlite import connect, read_cache_entries, write_cache_entries
from tests.builders import make_cache_entry, make_candidate


def add_task(conn: sqlite3.Connection, task_id: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO tasks (task_id, version, status, state_json, created_at, updated_at) "
            "VALUES (?, 0, 'active', '{}', '2026-10-05T00:00:00Z', '2026-10-05T00:00:00Z')",
            (task_id,),
        )


@pytest.fixture
def conn(tmp_state_db: Path) -> sqlite3.Connection:
    c = connect(tmp_state_db)
    add_task(c, "t-1")
    add_task(c, "t-2")
    return c


def entry(
    entry_id: str,
    *,
    task: str = "t-1",
    query: str = "q",
    step: int = 1,
    n: int = 2,
    mode: RetrievalMode = RetrievalMode.DENSE,
) -> CacheEntry:
    return make_cache_entry(
        task_id=task,
        entry_id=entry_id,
        normalized_query=query,
        created_at_step=step,
        mode=mode,
        candidates=[make_candidate(f"c{i}", rank=i) for i in range(n)],
    )


def write(conn: sqlite3.Connection, entries: list[CacheEntry], cap: int = 10) -> list[str]:
    with conn:
        return write_cache_entries(conn, entries, cap)


def test_connect_creates_the_sds_schema_with_wal_and_foreign_keys(tmp_state_db: Path) -> None:
    c = connect(tmp_state_db)
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tasks", "cache_entries"} <= tables
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    connect(tmp_state_db)  # idempotent on an existing database


def test_round_trip_keeps_vectors(conn: sqlite3.Connection) -> None:
    e = entry("e1")
    write(conn, [e])
    (stored,) = read_cache_entries(conn, "t-1")
    assert stored == e
    assert stored.query_vector == e.query_vector
    assert stored.candidates[0].vector == e.candidates[0].vector


def test_reads_only_the_requested_task(conn: sqlite3.Connection) -> None:
    write(conn, [entry("e1", task="t-1"), entry("e2", task="t-2")])
    assert [e.entry_id for e in read_cache_entries(conn, "t-1")] == ["e1"]
    assert [e.entry_id for e in read_cache_entries(conn, "t-2")] == ["e2"]


def test_entry_for_unknown_task_is_rejected(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        write(conn, [entry("e1", task="no-such-task")])


def test_deeper_result_replaces_and_shallower_is_ignored(conn: sqlite3.Connection) -> None:
    write(conn, [entry("e1", n=2)])
    write(conn, [entry("e2", n=1)])  # same query + mode, fewer candidates: kept e1
    assert [e.entry_id for e in read_cache_entries(conn, "t-1")] == ["e1"]
    write(conn, [entry("e3", n=3)])  # deeper: replaces
    assert [(e.entry_id, len(e.candidates)) for e in read_cache_entries(conn, "t-1")] == [("e3", 3)]
    write(conn, [entry("e4", n=3)])  # equally deep: the newer one wins
    assert [e.entry_id for e in read_cache_entries(conn, "t-1")] == ["e4"]


def test_same_query_in_another_mode_is_a_separate_entry(conn: sqlite3.Connection) -> None:
    write(conn, [entry("e1", mode=RetrievalMode.DENSE), entry("e2", mode=RetrievalMode.SPARSE)])
    assert len(read_cache_entries(conn, "t-1")) == 2


def test_oldest_entries_are_evicted_beyond_the_cap(conn: sqlite3.Connection) -> None:
    evicted = write(conn, [entry(f"e{i}", query=f"q{i}", step=i) for i in range(1, 6)], cap=3)
    assert evicted == ["e1", "e2"]
    assert [e.entry_id for e in read_cache_entries(conn, "t-1")] == ["e3", "e4", "e5"]
    write(conn, [entry("x", task="t-2")], cap=3)  # caps are per task
    assert len(read_cache_entries(conn, "t-1")) == 3


def test_writes_roll_back_with_the_callers_transaction(conn: sqlite3.Connection) -> None:
    with pytest.raises(RuntimeError), conn:
        write_cache_entries(conn, [entry("e1")], 10)
        raise RuntimeError("iteration failed before commit")
    assert read_cache_entries(conn, "t-1") == []  # no orphan entry (ADR-009 §9)


def test_unknown_schema_version_is_rejected(conn: sqlite3.Connection) -> None:
    payload = dump_with_vectors(entry("e1"))
    payload["schema_version"] = 99
    with conn:
        conn.execute(
            "INSERT INTO cache_entries VALUES ('t-1', 'e1', 'q', 'dense', ?, 1)",
            (json.dumps(payload),),
        )
    with pytest.raises(ValueError, match="schema_version 99"):
        read_cache_entries(conn, "t-1")


def _first_connections_at_once(path: Path, n: int = 6) -> list[Exception]:
    import threading

    barrier = threading.Barrier(n)
    errors: list[Exception] = []

    def worker() -> None:
        barrier.wait()
        try:
            conn = connect(path)
            with conn:
                conn.execute(
                    "INSERT INTO tasks VALUES (?, 0, 'active', '{}', 'x', 'x')",
                    (str(threading.get_ident()),),
                )
        except sqlite3.OperationalError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_first_connections_to_a_fresh_file_do_not_lock(tmp_path: Path) -> None:
    """Regression: threads opening their first connections at once got "database is
    locked" from the WAL switch (seen in eval runs with max_workers > 1; 20/40 trials
    failed before the setup lock in ``connect``)."""
    for trial in range(10):
        assert _first_connections_at_once(tmp_path / f"t{trial}.sqlite3") == []

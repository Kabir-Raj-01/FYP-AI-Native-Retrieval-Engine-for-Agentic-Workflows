"""SQLite TaskStore: the single persisted copy of S_t (ARCHITECTURE §3, §7; ADR-002, ADR-009 §9).

Table ``tasks`` (schema in ``engine/state/sqlite.py``). ``save`` is an optimistic
compare-and-swap on ``version`` and writes the iteration's queued cache entries in the same
transaction, so a conflict or error leaves both the state and the cache untouched.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from engine.config import EngineConfig
from engine.errors import TaskConflict, TaskExists, TaskNotFound
from engine.monitor.logging import get_logger
from engine.state.interfaces import TaskStore
from engine.state.models import Budget, CacheEntry, RetrievalState
from engine.state.serialization import dump_with_vectors, load_with_vectors
from engine.state.sqlite import connect, write_cache_entries

__all__ = ["STATE_SCHEMA_VERSION", "SqliteTaskStore"]

STATE_SCHEMA_VERSION = 1  # RetrievalState.schema_version this code reads and writes


class SqliteTaskStore(TaskStore):
    def __init__(
        self,
        cfg: EngineConfig,
        db_path: Path | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._cfg = cfg
        self._path = db_path if db_path is not None else cfg.paths.state_db
        self._now = now
        self._local = threading.local()  # one connection per thread (SDS §5.9)
        self._log = get_logger("engine.state")

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self._path)
            self._local.conn = conn
        return conn

    def create(self, goal: str, budget: Budget, task_id: str | None = None) -> RetrievalState:
        now = self._now()
        state = RetrievalState(
            task_id=task_id if task_id is not None else str(uuid4()),
            goal=goal,
            # only the caps are taken from the caller; every counter starts at 0
            budget=Budget(
                max_steps=budget.max_steps,
                max_retrieval_calls=budget.max_retrieval_calls,
                max_llm_calls=budget.max_llm_calls,
                max_tokens=budget.max_tokens,
            ),
            created_at=now,
            updated_at=now,
        )
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO tasks (task_id, version, status, state_json, created_at, "
                    "updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        state.task_id,
                        state.version,
                        state.status.value,
                        json.dumps(dump_with_vectors(state)),
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise TaskExists(f"task {state.task_id} already exists", task_id=state.task_id) from exc
        self._log.info("task.created", task_id=state.task_id, budget=state.budget.model_dump())
        return state

    def load(self, task_id: str) -> RetrievalState:
        row = (
            self._conn()
            .execute("SELECT state_json FROM tasks WHERE task_id = ?", (task_id,))
            .fetchone()
        )
        if row is None:
            raise TaskNotFound(f"no task {task_id}", task_id=task_id)
        state = load_with_vectors(RetrievalState, json.loads(row[0]))
        if state.schema_version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"task {task_id} has schema_version {state.schema_version}; this code reads "
                f"{STATE_SCHEMA_VERSION} (migrate or reject, ADR-009 §6)"
            )
        return state

    def save(
        self,
        state: RetrievalState,
        expected_version: int,
        cache_entries: Sequence[CacheEntry] = (),
    ) -> None:
        foreign = sorted({e.task_id for e in cache_entries} - {state.task_id})
        if foreign:  # the cache is task-local (CLAUDE.md rule 8)
            raise ValueError(f"cache entries of other tasks {foreign} in save of {state.task_id}")
        conn = self._conn()
        with conn:  # one transaction: state and cache entries commit or roll back together
            cur = conn.execute(
                "UPDATE tasks SET version = ?, status = ?, state_json = ?, updated_at = ? "
                "WHERE task_id = ? AND version = ?",
                (
                    state.version,
                    state.status.value,
                    json.dumps(dump_with_vectors(state)),
                    state.updated_at.isoformat(),
                    state.task_id,
                    expected_version,
                ),
            )
            if cur.rowcount == 0:
                found = conn.execute(
                    "SELECT version FROM tasks WHERE task_id = ?", (state.task_id,)
                ).fetchone()
                if found is None:
                    raise TaskNotFound(f"no task {state.task_id}", task_id=state.task_id)
                raise TaskConflict(
                    f"task {state.task_id}: stored version {found[0]}, expected {expected_version}",
                    task_id=state.task_id,
                )
            write_cache_entries(
                conn, cache_entries, self._cfg.cache.max_entries_per_task, log=self._log.info
            )

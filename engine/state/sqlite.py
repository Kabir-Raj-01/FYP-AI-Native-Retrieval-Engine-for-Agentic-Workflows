"""Shared SQLite access for the task state DB (SDS §4.6; ARCHITECTURE §3, §7; ADR-009 §9).

Used by the task-local cache now and by the TaskStore (``TaskStore.save`` must write the
iteration's cache entries in the same transaction as the state, so the cache-table rules
live here rather than in either component). Callers own transactions: the write helpers
never commit; wrap them in ``with conn:`` (commit on success, rollback on error).

Connections: one per thread (SDS §5.9), WAL journal, foreign keys enforced (a cache entry
must belong to an existing task).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

from engine.state.models import CacheEntry
from engine.state.serialization import dump_with_vectors, load_with_vectors

__all__ = [
    "CACHE_ENTRY_SCHEMA_VERSION",
    "connect",
    "ensure_schema",
    "read_cache_entries",
    "write_cache_entries",
]

CACHE_ENTRY_SCHEMA_VERSION = 1  # CacheEntry.schema_version this code reads and writes

# SDS §4.6, with IF NOT EXISTS so every component can ensure it.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
  task_id     TEXT PRIMARY KEY,
  version     INTEGER NOT NULL,
  status      TEXT NOT NULL CHECK (status IN ('active','stopped','failed')),
  state_json  TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cache_entries (
  task_id     TEXT NOT NULL REFERENCES tasks(task_id),
  entry_id    TEXT NOT NULL,
  normalized_query TEXT NOT NULL,
  mode        TEXT NOT NULL,
  entry_json  TEXT NOT NULL,
  created_at_step INTEGER NOT NULL,
  PRIMARY KEY (task_id, entry_id)
);
CREATE INDEX IF NOT EXISTS idx_cache_task_query ON cache_entries(task_id, normalized_query);
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open ``path`` (creating parent folders and the schema) for use by one thread."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def _load(entry_json: str) -> CacheEntry:
    entry = load_with_vectors(CacheEntry, json.loads(entry_json))
    if entry.schema_version != CACHE_ENTRY_SCHEMA_VERSION:
        raise ValueError(
            f"cache entry {entry.entry_id} has schema_version {entry.schema_version}; this code "
            f"reads {CACHE_ENTRY_SCHEMA_VERSION} (migrate or reject, ADR-009 §6)"
        )
    return entry


def read_cache_entries(
    conn: sqlite3.Connection, task_id: str, normalized_query: str | None = None
) -> list[CacheEntry]:
    """Entries of ``task_id`` (oldest first), optionally only those with
    ``normalized_query`` (uses the (task_id, normalized_query) index). Never reads another
    task's rows."""
    if normalized_query is None:
        rows = conn.execute(
            "SELECT entry_json FROM cache_entries WHERE task_id = ? "
            "ORDER BY created_at_step, entry_id",
            (task_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT entry_json FROM cache_entries WHERE task_id = ? AND normalized_query = ? "
            "ORDER BY created_at_step, entry_id",
            (task_id, normalized_query),
        ).fetchall()
    return [_load(row[0]) for row in rows]


def write_cache_entries(
    conn: sqlite3.Connection,
    entries: Sequence[CacheEntry],
    max_entries_per_task: int,
    log: Callable[..., object] | None = None,
) -> list[str]:
    """Insert ``entries`` under CACHE_SPEC §4's rules; return the evicted entry ids.

    - An entry with the same task, normalised query and mode as a stored one replaces it
      only if it has at least as many candidates ("the deeper result wins"); otherwise the
      stored entry is kept.
    - Above ``max_entries_per_task`` the oldest entries (lowest ``created_at_step``, then
      ``entry_id``) are dropped.
    Does not commit. Raises ``sqlite3.IntegrityError`` if a task does not exist.
    """
    tasks: set[str] = set()
    for entry in entries:
        tasks.add(entry.task_id)
        stored = conn.execute(
            "SELECT entry_id, entry_json FROM cache_entries "
            "WHERE task_id = ? AND normalized_query = ? AND mode = ?",
            (entry.task_id, entry.normalized_query, entry.mode.value),
        ).fetchone()
        if stored is not None:
            if len(entry.candidates) < len(_load(stored[1]).candidates):
                if log:
                    log(
                        "cache.put",
                        entry_id=entry.entry_id,
                        task_id=entry.task_id,
                        kept=stored[0],
                        reason="existing entry is deeper",
                    )
                continue
            conn.execute(
                "DELETE FROM cache_entries WHERE task_id = ? AND entry_id = ?",
                (entry.task_id, stored[0]),
            )
        conn.execute(
            "INSERT OR REPLACE INTO cache_entries (task_id, entry_id, normalized_query, mode, "
            "entry_json, created_at_step) VALUES (?, ?, ?, ?, ?, ?)",
            (
                entry.task_id,
                entry.entry_id,
                entry.normalized_query,
                entry.mode.value,
                json.dumps(dump_with_vectors(entry)),
                entry.created_at_step,
            ),
        )
        if log:
            log(
                "cache.put",
                entry_id=entry.entry_id,
                task_id=entry.task_id,
                query=entry.normalized_query,
                mode=entry.mode.value,
                replaced=stored[0] if stored else None,
            )

    evicted: list[str] = []
    for task_id in sorted(tasks):
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM cache_entries WHERE task_id = ?", (task_id,)
        ).fetchone()
        excess = count - max_entries_per_task
        if excess <= 0:
            continue
        oldest = [
            r[0]
            for r in conn.execute(
                "SELECT entry_id FROM cache_entries WHERE task_id = ? "
                "ORDER BY created_at_step, entry_id LIMIT ?",
                (task_id, excess),
            )
        ]
        conn.executemany(
            "DELETE FROM cache_entries WHERE task_id = ? AND entry_id = ?",
            [(task_id, e) for e in oldest],
        )
        evicted.extend(oldest)
        if log:
            for entry_id in oldest:
                log(
                    "cache.evict", entry_id=entry_id, task_id=task_id, reason="max_entries_per_task"
                )
    return evicted

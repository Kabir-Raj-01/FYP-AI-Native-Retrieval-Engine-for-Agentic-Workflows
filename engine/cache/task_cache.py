"""Task-local cache (CACHE_SPEC §2-§6; ADR-009 §9).

Lookup for (task_id, query, need_type), in order:
1. ``ablation.cache`` false -> None, without touching the database.
2. Only this task's entries are read (SQL ``WHERE task_id = ?``) — never another task's.
3. Only entries whose mode can serve the need: dense -> semantic/mixed, sparse -> lexical,
   hybrid -> all.
4. Exact: an entry whose ``normalized_query`` equals ``normalize(query)`` -> similarity 1.0
   (an indexed SQL lookup; no other entry is loaded and nothing is embedded).
5. Semantic: cosine of ``Embedder.embed_queries([query])`` with each ``query_vector``; the
   best one at or above ``cache.similarity_threshold``.
Ties prefer more candidates, then the newest entry, then ``entry_id``.

``put`` is for standalone use and tests; in the controlled loop, ``score_and_update`` queues
entries and ``TaskStore.save`` writes them in its transaction with the same rules
(``engine/state/sqlite.write_cache_entries``). Producers must build ``query_vector`` with
``Embedder.embed_queries`` so the vectors are comparable.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

from engine.cache.interfaces import Cache
from engine.config import EngineConfig
from engine.embeddings.interfaces import Embedder
from engine.errors import ConfigError, TaskNotFound
from engine.monitor.logging import get_logger
from engine.state.models import CacheEntry, CacheHit, NeedType, RetrievalMode
from engine.state.sqlite import connect, read_cache_entries, write_cache_entries
from engine.text import normalize

__all__ = ["SERVES", "TaskCache"]

# CACHE_SPEC §3 step 5: which stored retrieval mode can serve which information need.
SERVES: dict[RetrievalMode, frozenset[NeedType]] = {
    RetrievalMode.DENSE: frozenset({NeedType.SEMANTIC, NeedType.MIXED}),
    RetrievalMode.SPARSE: frozenset({NeedType.LEXICAL}),
    RetrievalMode.HYBRID: frozenset(NeedType),
}


def _preference(entry: CacheEntry) -> tuple[int, int, str]:
    """Sort key: more candidates, then newer, then entry_id (smaller key = preferred)."""
    return (-len(entry.candidates), -entry.created_at_step, entry.entry_id)


class TaskCache(Cache):
    def __init__(self, cfg: EngineConfig, embedder: Embedder, db_path: Path | None = None) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._path = db_path if db_path is not None else cfg.paths.state_db
        self._local = threading.local()  # one connection per thread (SDS §5.9)
        self._log = get_logger("engine.cache")

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self._path)
            self._local.conn = conn
        return conn

    def lookup(self, task_id: str, query: str, need_type: NeedType) -> CacheHit | None:
        if not self._cfg.ablation.cache:
            self._log.info("cache.miss", task_id=task_id, query=query, reason="ablation")
            return None
        conn = self._conn()
        normalized = normalize(query)
        exact = [
            e for e in read_cache_entries(conn, task_id, normalized) if need_type in SERVES[e.mode]
        ]
        hit = self._exact(exact, normalized)
        compatible: list[CacheEntry] = []
        if hit is None:  # only now load every entry of the task (semantic match)
            compatible = [
                e for e in read_cache_entries(conn, task_id) if need_type in SERVES[e.mode]
            ]
            hit = self._semantic(compatible, query)
        if hit is None:
            self._log.info(
                "cache.miss",
                task_id=task_id,
                query=query,
                need_type=need_type.value,
                candidates=len(compatible),
            )
        else:
            self._log.info(
                "cache.hit",
                task_id=task_id,
                query=query,
                similarity=round(hit.similarity, 4),
                match=hit.match,
                entry_id=hit.entry.entry_id,
            )
        return hit

    def _exact(self, entries: list[CacheEntry], normalized: str) -> CacheHit | None:
        matches = sorted((e for e in entries if e.normalized_query == normalized), key=_preference)
        return CacheHit(entry=matches[0], similarity=1.0, match="exact") if matches else None

    def _semantic(self, entries: list[CacheEntry], query: str) -> CacheHit | None:
        if not entries:
            return None
        q = np.asarray(self._embedder.embed_queries([query])[0], dtype=np.float64)
        best: tuple[float, CacheEntry] | None = None
        for entry in sorted(entries, key=_preference):
            v = np.asarray(entry.query_vector, dtype=np.float64)
            if v.shape != q.shape:
                raise ConfigError(
                    f"cache entry {entry.entry_id} has a {v.shape[0]}-d query_vector; the "
                    f"embedder produces {q.shape[0]}-d (embedding model changed?)"
                )
            denom = float(np.linalg.norm(q) * np.linalg.norm(v))
            similarity = float(q @ v) / denom if denom else 0.0
            if best is None or similarity > best[0]:  # strict: ties keep the preferred entry
                best = (similarity, entry)
        if best is None or best[0] < self._cfg.cache.similarity_threshold:
            return None
        return CacheHit(entry=best[1], similarity=best[0], match="semantic")

    def put(self, entry: CacheEntry) -> None:
        conn = self._conn()
        try:
            with conn:
                write_cache_entries(
                    conn, [entry], self._cfg.cache.max_entries_per_task, log=self._log.info
                )
        except sqlite3.IntegrityError as exc:
            raise TaskNotFound(
                f"cannot cache for unknown task {entry.task_id!r}: {exc}", task_id=entry.task_id
            ) from exc

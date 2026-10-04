"""Task-local cache interface (CACHE_SPEC §3-§6; ADR-009 §9)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.state.models import CacheEntry, CacheHit, NeedType

__all__ = ["Cache"]


class Cache(ABC):
    """Task-local retrieval cache. Every lookup is scoped to one ``task_id``."""

    @abstractmethod
    def lookup(self, task_id: str, query: str, need_type: NeedType) -> CacheHit | None:
        """Best compatible entry of ``task_id`` for ``query``, or None.

        Exact normalised match first (similarity 1.0), then semantic match by cosine of
        query embeddings at ``cache.similarity_threshold``; only entries whose mode can
        serve ``need_type`` qualify. Always None when ``ablation.cache`` is false. Never
        reads another task's entries. Emits ``cache.hit`` / ``cache.miss``.
        """
        raise NotImplementedError

    @abstractmethod
    def put(self, entry: CacheEntry) -> None:
        """Insert ``entry``, or replace the entry with the same normalised query and mode.

        In the controlled loop entries are not written through this method directly:
        ``score_and_update`` queues them and ``TaskStore.save`` commits them atomically
        with the state (CACHE_SPEC §4). Enforces ``cache.max_entries_per_task``.
        """
        raise NotImplementedError

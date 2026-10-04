"""TaskStore interface (ARCHITECTURE §2, §3, §7; STATE_SPEC §4; ADR-002, ADR-009 §9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from engine.state.models import Budget, CacheEntry, RetrievalState

__all__ = ["TaskStore"]


class TaskStore(ABC):
    """The single persisted copy of every task's state (SQLite, WAL mode; ADR-002).

    Also owns the ``cache_entries`` table writes, so a state save and the iteration's
    queued cache entries commit atomically (ADR-009 §9).
    """

    @abstractmethod
    def create(self, goal: str, budget: Budget, task_id: str | None = None) -> RetrievalState:
        """Create and persist S_0 for a new task and return it.

        ``task_id`` may be supplied by the caller (API ``CreateTaskRequest.task_id``; the
        eval harness uses dataset ids); otherwise the store generates one. Sets
        ``task_id``, ``goal``, ``budget.max_*`` and ``created_at``/``updated_at`` (UTC),
        which are immutable afterwards (STATE_SPEC §4 ownership matrix).

        Raises:
            TaskExists: a task with ``task_id`` already exists.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, task_id: str) -> RetrievalState:
        """Load the last saved state of ``task_id``, vectors included.

        Raises:
            TaskNotFound: no task with this id.
        """
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        state: RetrievalState,
        expected_version: int,
        cache_entries: Sequence[CacheEntry] = (),
    ) -> None:
        """Atomically save ``state`` and insert/replace every entry in ``cache_entries``.

        The caller (``persist``) has already set ``state.version = expected_version + 1``.
        Both writes happen in one transaction; if the stored version differs from
        ``expected_version`` nothing is written.

        Raises:
            TaskConflict: the stored version is not ``expected_version``.
            TaskNotFound: no task with ``state.task_id``.
        """
        raise NotImplementedError

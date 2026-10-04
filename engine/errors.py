"""Typed exceptions raised by engine components (ARCHITECTURE.md §8).

Every exception derives from :class:`EngineError` and carries, as class-level data, the
API error ``code`` and HTTP status that API_CONTRACT.md "Errors" assigns to it. The API
layer (Week 8) maps exceptions to responses by reading these attributes; nothing here
depends on FastAPI. Exceptions with no API_CONTRACT row map to 500 ``INTERNAL_ERROR``.

The hierarchy is deliberately flat: ``LLMOutputError`` and ``LLMBudgetExceeded`` are
siblings of ``LLMError``, not subclasses, so that ``except LLMError`` (transport failure,
planner reason ``LLM_ERROR``) can never swallow an output-validation failure (planner
reason ``SCHEMA``) or the per-run cap (which must abort the run).
"""

from __future__ import annotations

from http import HTTPStatus
from typing import ClassVar

__all__ = [
    "BackendError",
    "ConfigError",
    "EngineError",
    "LLMBudgetExceeded",
    "LLMError",
    "LLMOutputError",
    "PlannerValidationError",
    "RunTimeout",
    "StateInvariantError",
    "TaskConflict",
    "TaskExists",
    "TaskNotActive",
    "TaskNotFound",
]


class EngineError(Exception):
    """Base class of every engine exception.

    ``task_id`` is optional context for the API error body
    (``{"error": {"code", "message", "task_id"}}``).
    """

    code: ClassVar[str] = "INTERNAL_ERROR"
    http_status: ClassVar[HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str = "", *, task_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.task_id = task_id


class ConfigError(EngineError):
    """Invalid configuration at start-up; the service does not start (SDS §8.3)."""


class BackendError(EngineError):
    """Retrieval backend (Qdrant) timeout, connection failure or malformed response."""

    code = "BACKEND_ERROR"
    http_status = HTTPStatus.SERVICE_UNAVAILABLE


class LLMError(EngineError):
    """LLM transport failure after the client's own retries (LLM_CLIENT_SPEC §6)."""

    code = "LLM_ERROR"
    http_status = HTTPStatus.BAD_GATEWAY


class LLMOutputError(EngineError):
    """LLM response failed JSON parsing or schema validation (LLM_CLIENT_SPEC §5).

    The planner counts it as a rejected attempt with reason ``SCHEMA``; it is not
    expected to reach the API.
    """


class LLMBudgetExceeded(EngineError):  # noqa: N818 — name fixed by ARCHITECTURE §8
    """Per-run LLM call or cost cap reached (LLM_CLIENT_SPEC §8); aborts the run."""

    code = "RUN_BUDGET_EXCEEDED"
    http_status = HTTPStatus.TOO_MANY_REQUESTS


class PlannerValidationError(EngineError):
    """Planner output rejected by the validator. Internal only: the planner falls back."""


class TaskNotFound(EngineError):  # noqa: N818 — name fixed by ARCHITECTURE §8
    """No task with the given ``task_id``."""

    code = "TASK_NOT_FOUND"
    http_status = HTTPStatus.NOT_FOUND


class TaskExists(EngineError):  # noqa: N818 — name approved for API_CONTRACT TASK_EXISTS
    """A task with the caller-supplied ``task_id`` already exists."""

    code = "TASK_EXISTS"
    http_status = HTTPStatus.CONFLICT


class TaskConflict(EngineError):  # noqa: N818 — name fixed by ARCHITECTURE §8
    """Stored version differs from ``expected_version`` in ``TaskStore.save``."""

    code = "TASK_CONFLICT"
    http_status = HTTPStatus.CONFLICT


class TaskNotActive(EngineError):  # noqa: N818 — name fixed by ARCHITECTURE §8
    """step/run on a stopped task, or result on an active task."""

    code = "TASK_NOT_ACTIVE"
    http_status = HTTPStatus.CONFLICT


class RunTimeout(EngineError):  # noqa: N818 — name approved for API_CONTRACT RUN_TIMEOUT
    """``run`` exceeded ``api.run_timeout_s``; state reflects the last completed step."""

    code = "RUN_TIMEOUT"
    http_status = HTTPStatus.GATEWAY_TIMEOUT


class StateInvariantError(EngineError):
    """A STATE_SPEC §5 invariant was violated (ADR-009 §5). Signals a programming error."""

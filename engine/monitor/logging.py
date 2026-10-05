"""Structured event logging (OBSERVABILITY.md §3), minimal version.

Events are JSON lines (``logging.json``) at ``logging.level``, each with ``ts``, ``level``,
``event``, ``run_id``, ``task_id``, ``step`` (null when unbound) and the event's own fields
under ``data``. Producers bind the ids with ``bound_contextvars(run_id=..., task_id=...)``;
the binding is per thread, so a worker thread must bind inside its own work item (a
``ThreadPoolExecutor`` worker does not inherit the submitting thread's context).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import structlog
from structlog.contextvars import bound_contextvars, merge_contextvars

from engine.config import LoggingConfig

__all__ = ["bound_contextvars", "configure_logging", "get_logger"]

# Top-level keys of an event line (OBSERVABILITY §3); every other field goes under "data".
_TOP_LEVEL = ("ts", "level", "event", "run_id", "task_id", "step")


def _nest_data(_logger: Any, _method: str, event_dict: Any) -> dict[str, Any]:
    line = {key: event_dict.pop(key, None) for key in _TOP_LEVEL}
    line["data"] = event_dict
    return line


def configure_logging(cfg: LoggingConfig, events_file: Path | None = None) -> None:
    """Configure structlog once per process from the ``logging`` config section.

    With ``events_file`` (a run's ``events.jsonl``, OBSERVABILITY §4) events are appended
    there as JSON lines (UTF-8, LF) instead of being printed, whatever ``logging.json`` says.
    """
    level = logging.getLevelNamesMapping().get(cfg.level.upper(), logging.INFO)
    as_json = cfg.json_ or events_file is not None
    renderer: Any = (
        structlog.processors.JSONRenderer() if as_json else structlog.dev.ConsoleRenderer()
    )
    factory: Any = structlog.PrintLoggerFactory()
    if events_file is not None:
        events_file.parent.mkdir(parents=True, exist_ok=True)
        # Kept open for the process lifetime; structlog writes one line per event.
        stream = events_file.open("a", encoding="utf-8", newline="\n")
        factory = structlog.WriteLoggerFactory(file=stream)
    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            _nest_data,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=factory,
        cache_logger_on_first_use=False,
    )


def get_logger(component: str) -> Any:
    """A structlog logger carrying ``component`` (e.g. ``"engine.core"``).

    Lazy: it resolves the configuration on every call, so components built before
    ``configure_logging`` (the Engine, the cache) still log to the configured destination,
    e.g. a run's ``events.jsonl``. (``structlog.get_logger().bind(...)`` would freeze the
    configuration at construction time.)
    """
    return structlog.get_logger(component=component)

"""Structured event logging (OBSERVABILITY.md §3), minimal version.

Events are JSON lines (``logging.json``) at ``logging.level``, each with ``ts``, ``level``,
``event`` and the event's fields. This first version covers the direct path's
``retrieval.call``; run/task/step binding and the events of the controlled loop are added
with the trace writer (TASKS Weeks 8-9).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import structlog

from engine.config import LoggingConfig

__all__ = ["configure_logging", "get_logger"]


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
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
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

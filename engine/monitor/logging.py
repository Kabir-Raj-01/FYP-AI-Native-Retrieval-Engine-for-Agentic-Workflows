"""Structured event logging (OBSERVABILITY.md §3), minimal version.

Events are JSON lines (``logging.json``) at ``logging.level``, each with ``ts``, ``level``,
``event`` and the event's fields. This first version covers the direct path's
``retrieval.call``; run/task/step binding and the events of the controlled loop are added
with the trace writer (TASKS Weeks 8-9).
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

from engine.config import LoggingConfig

__all__ = ["configure_logging", "get_logger"]


def configure_logging(cfg: LoggingConfig) -> None:
    """Configure structlog once per process from the ``logging`` config section."""
    level = logging.getLevelNamesMapping().get(cfg.level.upper(), logging.INFO)
    renderer: Any = (
        structlog.processors.JSONRenderer() if cfg.json_ else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=False,
    )


def get_logger(component: str) -> Any:
    """A structlog logger bound to ``component`` (e.g. ``"engine.core"``)."""
    return structlog.get_logger().bind(component=component)

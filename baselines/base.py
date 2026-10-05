"""What every baseline receives (BASELINES_SPEC §1: ``run_task(task, ctx) -> EvidenceBundle``).

BASELINES_SPEC leaves ``ctx`` undefined; it is this context: the shared Engine (same
adapter, index and models as controller-on), the run's TraceWriter (same trace schema for
every system), the resolved config and the sub-run's retrieval budget B.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from engine.config import EngineConfig
from engine.core import Engine
from engine.monitor.interfaces import TraceWriter

__all__ = ["BaselineContext"]


@dataclass(frozen=True)
class BaselineContext:
    cfg: EngineConfig
    engine: Engine
    traces: TraceWriter
    retrieval_budget: int  # B of this sub-run (EVAL_PROTOCOL §1, BASELINES_SPEC §3)
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

"""TrajectoryMonitor and TraceWriter interfaces (ACTION_POLICY §6; OBSERVABILITY §1-§4;
ADR-009 §4, §6, §12)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from engine.state.models import (
    ActionDecision,
    AdvanceResult,
    EvidenceBundle,
    EvidenceUpdate,
    MonitorVerdict,
    RetrievalState,
    RunContext,
    TraceStep,
    TraceSummary,
)

__all__ = ["TraceWriter", "TrajectoryMonitor"]


class TrajectoryMonitor(ABC):
    """Progress bookkeeping, gap advancement and stop checks after every step."""

    @abstractmethod
    def observe(
        self,
        state: RetrievalState,
        decision: ActionDecision,
        update: EvidenceUpdate | None,
    ) -> MonitorVerdict:
        """Evaluate the step just executed (ACTION_POLICY §6, in that order).

        ``update`` is None for steps that produced no candidates (TRANSFORM, STOP). The
        verdict carries full new values (stagnation_count, transforms_in_streak,
        action_gain_ema, gap status changes, current gap) that the ``monitor`` node
        applies verbatim, and a stop reason when the task must stop — including
        ``steps_used + 1 >= max_steps`` -> MAX_STEPS (ADR-009 §12).
        """
        raise NotImplementedError

    @abstractmethod
    def advance(self, state: RetrievalState) -> AdvanceResult:
        """Next current gap: the lowest-index OPEN gap's id, or None if there is none.

        Called by ``maybe_plan`` after INIT, by ``execute`` after a SPLIT and by
        ``observe`` on gap resolution/exhaustion (ADR-009 §4). Only ``current_gap_id`` is
        applied; ``observe`` resets ``stagnation_count``/``transforms_in_streak`` itself
        when it advances (ADR-009 A6 as decided 2026-10-05).
        """
        raise NotImplementedError


class TraceWriter(ABC):
    """Writes and reads one run's TraceSteps and EvidenceBundles (OBSERVABILITY §1-§2, §4).

    Shared by the engine's ``persist`` node and every baseline, so all systems produce the
    same records. Bound to one ``RunContext``: ``run_id`` and ``system`` are stamped on every
    record it creates (ADR-009 §6), and records from another context are refused.
    """

    @property
    @abstractmethod
    def context(self) -> RunContext:
        """The run this writer belongs to."""
        raise NotImplementedError

    def new_step(self, **fields: Any) -> TraceStep:
        """A TraceStep stamped with this writer's ``run_id`` and ``system``."""
        return TraceStep(run_id=self.context.run_id, system=self.context.system, **fields)

    @abstractmethod
    def write_step(self, step: TraceStep) -> None:
        """Append ``step`` to its task's trace.

        Raises:
            ValueError: ``step`` belongs to another run or system, or its ``step`` number is
                not the next one for its task (1, 2, 3, ... — catches duplicates and gaps).
        """
        raise NotImplementedError

    @abstractmethod
    def write_bundle(self, bundle: EvidenceBundle) -> None:
        """Store the terminal bundle of ``bundle.task_id`` (replacing any earlier one).

        Raises:
            ValueError: ``bundle.system`` is not this writer's system.
        """
        raise NotImplementedError

    @abstractmethod
    def read_steps(self, task_id: str) -> list[TraceStep]:
        """Every step of ``task_id`` in order; ``[]`` if it has none."""
        raise NotImplementedError

    @abstractmethod
    def read_bundle(self, task_id: str) -> EvidenceBundle | None:
        """The stored bundle of ``task_id``, or None."""
        raise NotImplementedError

    @staticmethod
    def summarize(steps: Sequence[TraceStep]) -> TraceSummary:
        """``TraceSummary`` of a task's steps (OBSERVABILITY §2, made precise in Week 13):
        steps / retrieval_calls / llm_calls / tokens / reranker_calls from the last step's
        ``budget_after`` (so they agree with the budget); ``cache_hits`` = steps with
        ``cache_hit``; ``total_latency_ms`` = sum of ``latency_ms["total"]``; ``actions`` =
        the action sequence.

        Raises:
            ValueError: no steps.
        """
        if not steps:
            raise ValueError("cannot summarise a task with no trace steps")
        budget = steps[-1].budget_after
        return TraceSummary(
            steps=budget.steps_used,
            retrieval_calls=budget.retrieval_calls_used,
            cache_hits=sum(1 for s in steps if s.cache_hit),
            llm_calls=budget.llm_calls_used,
            tokens=budget.tokens_used,
            reranker_calls=budget.reranker_calls_used,
            total_latency_ms=sum(s.latency_ms.get("total", 0.0) for s in steps),
            actions=[s.action for s in steps],
        )

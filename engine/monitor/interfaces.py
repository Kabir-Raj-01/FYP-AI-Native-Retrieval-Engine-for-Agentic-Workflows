"""TrajectoryMonitor interface (ACTION_POLICY §6; ADR-009 §4, §12)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.state.models import (
    ActionDecision,
    AdvanceResult,
    EvidenceUpdate,
    MonitorVerdict,
    RetrievalState,
)

__all__ = ["TrajectoryMonitor"]


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
        ``observe`` on gap resolution/exhaustion (ADR-009 §4). Which further fields it
        resets is an open spec question (Week 7 checkpoint, S5).
        """
        raise NotImplementedError

"""Planner interface (PLANNER_SPEC §1-§6; ADR-009 §3, §10)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.state.models import PlanMode, PlanOutcome, RetrievalState

__all__ = ["Planner"]


class Planner(ABC):
    """Bounded LLM query planner. Never writes evidence or resolves gaps."""

    @abstractmethod
    def plan(self, state: RetrievalState, mode: PlanMode) -> PlanOutcome:
        """Plan subgoals for ``state`` in ``mode`` (INIT, or REWRITE/SPLIT for TRANSFORM).

        For TRANSFORM, ``mode`` is ``ActionDecision.plan_mode``, decided by the
        controller; the planner never re-derives it. Makes at most
        ``planner.max_attempts`` LLM calls (each only while
        ``budget.llm_calls_used < max_llm_calls``), validates each response
        (``engine/planner/validator.py``), and otherwise returns the deterministic
        fallback plan. Pure with respect to ``state``: the calling node applies the
        outcome (gaps, query_history, plan_events, budget).

        Returns:
            PlanOutcome with the accepted or fallback ``PlanResult`` (``mode`` equal to the
            requested mode), every ACCEPTED/REJECTED/FALLBACK event, and the LLM calls and
            tokens consumed.

        Raises:
            LLMBudgetExceeded: the per-run LLM cap was hit (aborts the run). Transport and
            output errors are not raised: they become REJECTED attempts.
        """
        raise NotImplementedError

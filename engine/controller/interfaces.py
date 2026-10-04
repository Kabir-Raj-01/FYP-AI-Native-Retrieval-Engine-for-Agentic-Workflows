"""Controller interface (ACTION_POLICY §2-§5, §7; ADR-009 §3, §12)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.state.models import ActionDecision, CacheHit, RetrievalState

__all__ = ["Controller"]


class Controller(ABC):
    """Deterministic action selection: rules, then utility, then fixed tie-break.

    Never calls an LLM (CLAUDE.md rule 9). The same state, cache hit and config always
    yield the same decision.
    """

    @abstractmethod
    def select_action(self, state: RetrievalState, cache_hit: CacheHit | None) -> ActionDecision:
        """Choose the next action for ``state`` (ACTION_POLICY §2).

        ``cache_hit`` is the ``Cache.lookup`` result for the pending query (None on a
        miss). The decision records the eligible set, per-action ``ActionScores`` for
        eligible non-STOP actions, the rule that fired, a ``stop_reason`` when the action
        is STOP, and ``plan_mode`` exactly when the action is TRANSFORM.
        """
        raise NotImplementedError

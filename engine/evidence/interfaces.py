"""EvidenceManager interface (EVIDENCE_MANAGER_SPEC §1-§5; ADR-009 §4)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.state.models import EvidenceUpdate, RetrievalState, ScoredCandidate

__all__ = ["EvidenceManager"]


class EvidenceManager(ABC):
    """The only component that decides evidence acceptance and gap resolution."""

    @abstractmethod
    def update(
        self, state: RetrievalState, scored: list[ScoredCandidate], step: int
    ) -> EvidenceUpdate:
        """Decide acceptance, eviction and gap resolution for one step's ``scored`` list.

        ``step`` is the iteration's ``n = budget.steps_used + 1``, stamped on new items
        and on gaps closed this step. Pure with respect to ``state``: the result carries
        the accepted items, rejections with reason codes, evictions, resolved gaps and
        their resolving items, flag changes on existing items, and the step's
        ``evidence_gain`` in [0,1]; the ``score_and_update`` node applies it.
        """
        raise NotImplementedError

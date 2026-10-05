"""Scripted fakes of the controlled-loop components (TESTING §2 pattern).

They exist so the graph and the Engine loop can be tested before the real Planner,
Controller, EvidenceManager and TrajectoryMonitor are built, and so golden/fault tests can
script exact trajectories. Their logic is deliberately simple; the real components are
tested against their own specs.
"""

from __future__ import annotations

from collections.abc import Sequence

from engine.controller.interfaces import Controller
from engine.evidence.interfaces import EvidenceManager
from engine.monitor.interfaces import TrajectoryMonitor
from engine.planner.interfaces import Planner
from engine.state.models import (
    Action,
    ActionDecision,
    AdvanceResult,
    CacheHit,
    EvidenceItem,
    EvidenceUpdate,
    GapStatus,
    MonitorVerdict,
    NeedType,
    PlanEvent,
    PlanMode,
    PlanOutcome,
    PlanResult,
    PlanStatus,
    RetrievalState,
    ScoredCandidate,
    StopReason,
    SubgoalPlan,
)

__all__ = ["FakeController", "FakeEvidenceManager", "FakeMonitor", "FakePlanner"]


class FakePlanner(Planner):
    """INIT -> ``init`` subgoals (default: the goal as one MIXED subgoal); REWRITE/SPLIT ->
    the next of ``transforms`` (default: a REWRITE of the pending query). Each call reports
    ``llm_calls`` and ``tokens``. ``fail`` is raised instead of planning, if set."""

    def __init__(
        self,
        init: Sequence[SubgoalPlan] | None = None,
        transforms: Sequence[PlanResult] = (),
        llm_calls: int = 1,
        tokens: int = 100,
        fail: Exception | None = None,
    ) -> None:
        self._init = list(init) if init is not None else None
        self._transforms = list(transforms)
        self._llm_calls = llm_calls
        self._tokens = tokens
        self.fail = fail
        self.calls: list[PlanMode] = []

    def plan(self, state: RetrievalState, mode: PlanMode) -> PlanOutcome:
        self.calls.append(mode)
        if self.fail is not None:
            raise self.fail
        if mode == PlanMode.INIT:
            subgoals = self._init or [
                SubgoalPlan(description=state.goal, query=state.goal, need_type=NeedType.MIXED)
            ]
            result = PlanResult(mode=mode, subgoals=subgoals)
        elif self._transforms:
            result = self._transforms.pop(0)
        else:
            result = PlanResult(
                mode=PlanMode.REWRITE,
                subgoals=[
                    SubgoalPlan(
                        description=state.current_subgoal,
                        query=f"{state.pending_query} rewritten",
                        need_type=state.current_need_type,
                    )
                ],
            )
        event = PlanEvent(
            step=state.budget.steps_used + 1, mode=mode, attempt=1, status=PlanStatus.ACCEPTED
        )
        return PlanOutcome(
            result=result, events=[event], llm_calls=self._llm_calls, tokens=self._tokens
        )


class FakeController(Controller):
    """Plays ``actions`` in order, then repeats the last one. TRANSFORM carries
    ``plan_mode``; STOP carries ``stop_reason``. Records each call's cache hit."""

    def __init__(
        self,
        actions: Sequence[Action] = (Action.DENSE,),
        plan_mode: PlanMode = PlanMode.REWRITE,
        stop_reason: StopReason = StopReason.LOW_MARGINAL_UTILITY,
    ) -> None:
        self._actions = list(actions)
        self._plan_mode = plan_mode
        self._stop_reason = stop_reason
        self.hits: list[CacheHit | None] = []

    def select_action(self, state: RetrievalState, cache_hit: CacheHit | None) -> ActionDecision:
        self.hits.append(cache_hit)
        action = self._actions[min(len(self.hits), len(self._actions)) - 1]
        return ActionDecision(
            action=action,
            eligible=[action],
            scores={},
            rule="scripted",
            plan_mode=self._plan_mode if action == Action.TRANSFORM else None,
            stop_reason=self._stop_reason if action == Action.STOP else None,
        )


class FakeEvidenceManager(EvidenceManager):
    """Accepts up to ``accept`` unseen candidates with relevance >= ``threshold`` (score
    order); with ``resolve``, the first accepted item resolves the current gap. Gain =
    accepted / accept. ``fail`` is raised instead of updating, if set."""

    def __init__(
        self,
        accept: int = 2,
        threshold: float = 0.0,
        resolve: bool = False,
        fail: Exception | None = None,
    ) -> None:
        self._accept = accept
        self._threshold = threshold
        self._resolve = resolve
        self.fail = fail

    def update(
        self, state: RetrievalState, scored: list[ScoredCandidate], step: int
    ) -> EvidenceUpdate:
        if self.fail is not None:
            raise self.fail
        have = {e.evidence_id for e in state.evidence}
        decision = state.last_decision
        source = decision.action if decision is not None else Action.DENSE
        query = state.retrieval_history[-1].query if state.retrieval_history else ""
        accepted: list[EvidenceItem] = []
        rejected: dict[str, str] = {}
        for s in scored:
            c = s.candidate
            if c.chunk_id in have:
                rejected[c.chunk_id] = "DUPLICATE"
            elif s.relevance < self._threshold:
                rejected[c.chunk_id] = "LOW_RELEVANCE"
            elif len(accepted) >= self._accept:
                rejected[c.chunk_id] = "STEP_CAP"
            else:
                accepted.append(
                    EvidenceItem(
                        evidence_id=c.chunk_id,
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        text=c.text,
                        summary=c.text[:40],
                        relevance=s.relevance,
                        novelty=s.novelty,
                        final_score=s.final_score,
                        source_action=source,
                        query=query,
                        metadata=dict(c.metadata),
                        added_at_step=step,
                        vector=c.vector,
                    )
                )
        resolutions: dict[str, str] = {}
        if self._resolve and accepted and state.current_gap_id is not None:
            gap = state.current_gap_id
            accepted[0] = accepted[0].model_copy(update={"pinned": True, "supports_gap_ids": [gap]})
            resolutions = {gap: accepted[0].evidence_id}
        return EvidenceUpdate(
            accepted=accepted,
            rejected=rejected,
            evicted=[],
            resolved_gap_ids=list(resolutions),
            gap_resolutions=resolutions,
            evidence_gain=min(1.0, len(accepted) / self._accept) if self._accept else 0.0,
        )


class FakeMonitor(TrajectoryMonitor):
    """ACTION_POLICY §6.3-§6.4 without stagnation logic: advance when the current gap is
    RESOLVED; stop on no OPEN gaps, ``steps_used + 1 >= max_steps``, the token cap, or a
    STOP decision. ``advance`` = lowest-index OPEN gap."""

    def observe(
        self, state: RetrievalState, decision: ActionDecision, update: EvidenceUpdate | None
    ) -> MonitorVerdict:
        current = state.current_gap_id
        gap = state.current_gap
        if gap is not None and gap.status != GapStatus.OPEN:
            current = self.advance(state).current_gap_id
        open_gaps = [g for g in state.gaps if g.status == GapStatus.OPEN]
        roots = [g for g in state.gaps if g.parent_gap_id is None]
        b = state.budget
        stop: StopReason | None = None
        if not open_gaps:
            resolved = all(g.status == GapStatus.RESOLVED for g in roots)
            stop = StopReason.ALL_GAPS_RESOLVED if resolved else StopReason.STAGNATION
        elif b.steps_used + 1 >= b.max_steps:
            stop = StopReason.MAX_STEPS
        elif b.tokens_used >= b.max_tokens:
            stop = StopReason.MAX_TOKENS
        elif decision.action == Action.STOP:
            stop = decision.stop_reason
        return MonitorVerdict(
            should_continue=stop is None,
            stop_reason=stop,
            advanced_to_gap=current if current != state.current_gap_id else None,
            stagnation_count=state.stagnation_count,
            transforms_in_streak=state.transforms_in_streak,
            action_gain_ema=dict(state.action_gain_ema),
            current_gap_id=current,
        )

    def advance(self, state: RetrievalState) -> AdvanceResult:
        first = next((g.gap_id for g in state.gaps if g.status == GapStatus.OPEN), None)
        return AdvanceResult(current_gap_id=first, reset_stagnation=True)

"""TrajectoryMonitor (ACTION_POLICY §6; ADR-009 §4, §12, addendum A6).

``observe`` runs §6 in order — progress bookkeeping, parent propagation, gap advancement,
stop checks — and returns full new values for the ``monitor`` node to apply verbatim.

Decisions on ADR-009 A6 (2026-10-05, for Samay's review):
- A SUPERSEDED parent keeps that status; its *effective* status (RESOLVED when all children
  are, EXHAUSTED when all children are closed and one is EXHAUSTED) is computed here for
  the stop checks only, so invariant 4 and the ownership matrix stay as written.
- Advancing resets ``stagnation_count`` and ``transforms_in_streak`` (monitor-owned) but not
  ``last_result`` (owned by ``score_and_update``); RERANK_EXPAND already requires a
  ``last_result`` for the current gap's query, so a stale one is never used.
A STOP with rule ``fixed_policy`` (ACTION_POLICY §7: "STOP for this gap") or ``min_utility``
(§2 step 8, as amended 2026-10-08) exhausts the current gap and advances; the task stops
only when no OPEN gap is left, then with that STOP's reason.
"""

from __future__ import annotations

from engine.config import EngineConfig
from engine.monitor.interfaces import TrajectoryMonitor
from engine.state.models import (
    Action,
    ActionDecision,
    AdvanceResult,
    EvidenceGap,
    EvidenceUpdate,
    GapStatus,
    MonitorVerdict,
    RetrievalState,
    StopReason,
)

__all__ = ["RuleTrajectoryMonitor"]

_CANDIDATE_ACTIONS = frozenset(
    {Action.REUSE, Action.DENSE, Action.SPARSE_HYBRID, Action.RERANK_EXPAND}
)
# engine/controller/policy.py FIXED_POLICY_RULE, MIN_UTILITY_RULE: STOPs that end a gap only
_GAP_STOP_RULES = frozenset({"fixed_policy", "min_utility"})
_CLOSED = frozenset({GapStatus.RESOLVED, GapStatus.EXHAUSTED})


def effective_status(gap: EvidenceGap, gaps: list[EvidenceGap]) -> GapStatus:
    """A SUPERSEDED gap counts as RESOLVED when all its children are, as EXHAUSTED when
    all are closed and any is EXHAUSTED (recursively); otherwise its own status."""
    if gap.status != GapStatus.SUPERSEDED:
        return gap.status
    children = [effective_status(g, gaps) for g in gaps if g.parent_gap_id == gap.gap_id]
    if children and all(s == GapStatus.RESOLVED for s in children):
        return GapStatus.RESOLVED
    if children and all(s in _CLOSED for s in children):
        return GapStatus.EXHAUSTED
    return GapStatus.SUPERSEDED


class RuleTrajectoryMonitor(TrajectoryMonitor):
    def __init__(self, cfg: EngineConfig) -> None:
        self._cfg = cfg

    def advance(self, state: RetrievalState) -> AdvanceResult:
        first = next((g.gap_id for g in state.gaps if g.status == GapStatus.OPEN), None)
        return AdvanceResult(current_gap_id=first, reset_stagnation=True)

    def observe(
        self, state: RetrievalState, decision: ActionDecision, update: EvidenceUpdate | None
    ) -> MonitorVerdict:
        cfg = self._cfg
        m = cfg.monitor
        stagnation = state.stagnation_count
        streak = state.transforms_in_streak
        emas = dict(state.action_gain_ema)
        action = decision.action

        # 1. progress bookkeeping
        if action in _CANDIDATE_ACTIONS and update is not None:
            gain = update.evidence_gain
            d = cfg.controller.ema_decay
            emas[action] = d * emas.get(action, cfg.controller.prior_action_gain) + (1 - d) * gain
            if gain < m.stagnation_threshold:
                stagnation += 1
            else:
                stagnation, streak = 0, 0
        elif action == Action.TRANSFORM:
            streak += 1

        # 3. gap advancement (2. parent propagation is effective_status, used below)
        changes: dict[str, GapStatus] = {}
        current = state.current_gap_id
        gap = state.current_gap
        advanced: str | None = None
        exhausted: str | None = None
        if gap is not None and cfg.ablation.gap_tracking:
            gap_stop = action == Action.STOP and decision.rule in _GAP_STOP_RULES
            stuck = stagnation >= m.stagnation_patience and streak >= m.max_transforms_per_streak
            if gap.status == GapStatus.OPEN and (gap_stop or stuck):
                changes[gap.gap_id] = GapStatus.EXHAUSTED
                exhausted = gap.gap_id
            if gap.status != GapStatus.OPEN or exhausted is not None:
                after = [
                    g.model_copy(update={"status": changes[g.gap_id]}) if g.gap_id in changes else g
                    for g in state.gaps
                ]
                nxt = self.advance(state.model_copy(update={"gaps": after})).current_gap_id
                if nxt != current:
                    current, advanced = nxt, nxt
                    stagnation, streak = 0, 0

        # 4. stop checks, first match wins
        gaps = [
            g.model_copy(update={"status": changes[g.gap_id]}) if g.gap_id in changes else g
            for g in state.gaps
        ]
        open_gaps = [g for g in gaps if g.status == GapStatus.OPEN]
        roots = [g for g in gaps if g.parent_gap_id is None]
        b = state.budget
        stop: StopReason | None = None
        if not open_gaps:
            resolved = all(effective_status(g, gaps) == GapStatus.RESOLVED for g in roots)
            if resolved:
                stop = StopReason.ALL_GAPS_RESOLVED
            elif exhausted is not None and decision.stop_reason is not None:
                stop = decision.stop_reason  # the gap-ending STOP closed the last gap
            else:
                stop = StopReason.STAGNATION
        elif b.steps_used + 1 >= b.max_steps:  # ADR-009 §12
            stop = StopReason.MAX_STEPS
        elif b.tokens_used >= b.max_tokens:
            stop = StopReason.MAX_TOKENS
        elif action == Action.STOP and (decision.rule not in _GAP_STOP_RULES or exhausted is None):
            stop = decision.stop_reason  # a gap-ending STOP that exhausted a gap goes on

        return MonitorVerdict(
            should_continue=stop is None,
            stop_reason=stop,
            advanced_to_gap=advanced,
            exhausted_gap=exhausted,
            stagnation_count=stagnation,
            transforms_in_streak=streak,
            action_gain_ema=emas,
            gap_status_changes=changes,
            current_gap_id=current,
            note=f"gain={update.evidence_gain:.3f}" if update is not None else "",
        )

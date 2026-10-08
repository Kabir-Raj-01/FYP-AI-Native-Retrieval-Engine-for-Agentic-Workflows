"""engine/monitor/trajectory.py (ACTION_POLICY §6; ADR-009 §12 and the A6 decisions)."""

from __future__ import annotations

from typing import Any

import pytest

from engine.config import EngineConfig
from engine.monitor.trajectory import RuleTrajectoryMonitor, effective_status
from engine.state.models import (
    Action,
    ActionDecision,
    EvidenceUpdate,
    GapStatus,
    RetrievalState,
    StopReason,
)
from tests.builders import make_budget, make_gap, make_state


def decision(action: Action = Action.DENSE, **kw: Any) -> ActionDecision:
    return ActionDecision(
        action=action, eligible=[action], scores={}, rule=kw.pop("rule", "utility_argmax"), **kw
    )


def update(gain: float) -> EvidenceUpdate:
    return EvidenceUpdate(
        accepted=[], rejected={}, evicted=[], resolved_gap_ids=[], evidence_gain=gain
    )


def two_gaps(**kw: Any) -> RetrievalState:
    fields: dict[str, Any] = {
        "gaps": [make_gap("g0"), make_gap("g1")],
        "current_gap_id": "g0",
        "needs_initial_plan": False,
    }
    fields.update(kw)
    return make_state(**fields)


def test_low_gain_counts_stagnation_and_updates_the_ema(cfg: EngineConfig) -> None:
    v = RuleTrajectoryMonitor(cfg).observe(two_gaps(), decision(), update(0.0))
    d = cfg.controller.ema_decay
    assert v.stagnation_count == 1 and v.should_continue
    assert v.action_gain_ema[Action.DENSE] == pytest.approx(d * cfg.controller.prior_action_gain)


def test_good_gain_resets_stagnation_and_the_transform_streak(cfg: EngineConfig) -> None:
    s = two_gaps(stagnation_count=1, transforms_in_streak=1)
    v = RuleTrajectoryMonitor(cfg).observe(s, decision(), update(0.9))
    assert (v.stagnation_count, v.transforms_in_streak) == (0, 0)


def test_transform_extends_the_streak_without_touching_stagnation(cfg: EngineConfig) -> None:
    s = two_gaps(stagnation_count=1)
    v = RuleTrajectoryMonitor(cfg).observe(s, decision(Action.TRANSFORM), None)
    assert (v.stagnation_count, v.transforms_in_streak) == (1, 1)
    assert v.action_gain_ema == {}


def test_resolved_current_gap_advances_and_resets_counters(cfg: EngineConfig) -> None:
    s = two_gaps(
        gaps=[make_gap("g0", status=GapStatus.RESOLVED, resolved_by="c1"), make_gap("g1")],
        stagnation_count=1,
        transforms_in_streak=1,
    )
    v = RuleTrajectoryMonitor(cfg).observe(s, decision(), update(0.0))
    assert (v.current_gap_id, v.advanced_to_gap) == ("g1", "g1")
    assert (v.stagnation_count, v.transforms_in_streak) == (0, 0)  # A6: counters only
    assert v.should_continue


def test_stuck_gap_is_exhausted_after_patience_and_a_transform(cfg: EngineConfig) -> None:
    m = cfg.monitor
    s = two_gaps(
        stagnation_count=m.stagnation_patience - 1, transforms_in_streak=m.max_transforms_per_streak
    )
    v = RuleTrajectoryMonitor(cfg).observe(s, decision(), update(0.0))
    assert v.gap_status_changes == {"g0": GapStatus.EXHAUSTED} and v.exhausted_gap == "g0"
    assert v.current_gap_id == "g1" and v.should_continue
    not_yet = two_gaps(stagnation_count=m.stagnation_patience - 1, transforms_in_streak=0)
    assert (
        RuleTrajectoryMonitor(cfg).observe(not_yet, decision(), update(0.0)).gap_status_changes
        == {}
    )


def test_stop_checks_in_order(cfg: EngineConfig) -> None:
    mon = RuleTrajectoryMonitor(cfg)
    resolved = make_state(
        gaps=[make_gap("g0", status=GapStatus.RESOLVED, resolved_by="c1")], current_gap_id="g0"
    )
    assert (
        mon.observe(resolved, decision(), update(0.5)).stop_reason == StopReason.ALL_GAPS_RESOLVED
    )
    exhausted = make_state(gaps=[make_gap("g0", status=GapStatus.EXHAUSTED)], current_gap_id="g0")
    assert mon.observe(exhausted, decision(), update(0.5)).stop_reason == StopReason.STAGNATION
    last_step = two_gaps(budget=make_budget(max_steps=3, steps_used=2))
    v = mon.observe(last_step, decision(), update(0.5))
    assert (v.should_continue, v.stop_reason) == (False, StopReason.MAX_STEPS)  # steps_used + 1
    assert mon.observe(
        two_gaps(budget=make_budget(max_steps=4, steps_used=2)), decision(), update(0.5)
    ).should_continue
    tokens = two_gaps(budget=make_budget(max_tokens=10, tokens_used=10))
    assert mon.observe(tokens, decision(), update(0.5)).stop_reason == StopReason.MAX_TOKENS
    stop = decision(Action.STOP, stop_reason=StopReason.LOW_MARGINAL_UTILITY)
    assert mon.observe(two_gaps(), stop, None).stop_reason == StopReason.LOW_MARGINAL_UTILITY


def test_superseded_parent_counts_by_its_children(cfg: EngineConfig) -> None:
    parent = make_gap("g0", status=GapStatus.SUPERSEDED)
    done = [
        make_gap("g1", parent_gap_id="g0", status=GapStatus.RESOLVED, resolved_by="c1"),
        make_gap("g2", parent_gap_id="g0", status=GapStatus.RESOLVED, resolved_by="c2"),
    ]
    s = make_state(gaps=[parent, *done], current_gap_id="g2")
    v = RuleTrajectoryMonitor(cfg).observe(s, decision(), update(0.5))
    assert v.stop_reason == StopReason.ALL_GAPS_RESOLVED
    assert v.gap_status_changes == {}  # the parent stays SUPERSEDED (A6)
    one_failed = [done[0], make_gap("g2", parent_gap_id="g0", status=GapStatus.EXHAUSTED)]
    assert effective_status(parent, [parent, *one_failed]) == GapStatus.EXHAUSTED
    s = make_state(gaps=[parent, *one_failed], current_gap_id="g2")
    assert (
        RuleTrajectoryMonitor(cfg).observe(s, decision(), update(0.5)).stop_reason
        == StopReason.STAGNATION
    )
    assert (
        effective_status(parent, [parent, done[0], make_gap("g2", parent_gap_id="g0")])
        == GapStatus.SUPERSEDED
    )


def test_fixed_policy_stop_exhausts_the_gap_and_moves_on(cfg: EngineConfig) -> None:
    stop = decision(Action.STOP, stop_reason=StopReason.STAGNATION, rule="fixed_policy")
    v = RuleTrajectoryMonitor(cfg).observe(two_gaps(), stop, None)
    assert (v.should_continue, v.current_gap_id, v.gap_status_changes) == (
        True,
        "g1",
        {"g0": GapStatus.EXHAUSTED},
    )
    last = make_state(gaps=[make_gap("g0")], current_gap_id="g0")
    v = RuleTrajectoryMonitor(cfg).observe(last, stop, None)
    assert (v.should_continue, v.stop_reason) == (False, StopReason.STAGNATION)


def test_min_utility_stop_ends_the_gap_not_the_task(cfg: EngineConfig) -> None:
    # live pilot 2026-10-08: every task stopped after its first subgoal's retrieval
    stop = decision(Action.STOP, stop_reason=StopReason.LOW_MARGINAL_UTILITY, rule="min_utility")
    v = RuleTrajectoryMonitor(cfg).observe(two_gaps(), stop, None)
    assert (v.should_continue, v.current_gap_id, v.exhausted_gap) == (True, "g1", "g0")
    last = make_state(gaps=[make_gap("g0")], current_gap_id="g0")
    v = RuleTrajectoryMonitor(cfg).observe(last, stop, None)
    assert (v.should_continue, v.stop_reason) == (False, StopReason.LOW_MARGINAL_UTILITY)
    cfg.ablation.gap_tracking = False  # one gap, the goal: the task stops as before
    v = RuleTrajectoryMonitor(cfg).observe(two_gaps(), stop, None)
    assert (v.should_continue, v.stop_reason) == (False, StopReason.LOW_MARGINAL_UTILITY)


def test_gap_tracking_off_never_advances_or_exhausts(cfg: EngineConfig) -> None:
    cfg.ablation.gap_tracking = False
    m = cfg.monitor
    s = two_gaps(
        stagnation_count=m.stagnation_patience, transforms_in_streak=m.max_transforms_per_streak
    )
    v = RuleTrajectoryMonitor(cfg).observe(s, decision(), update(0.0))
    assert v.gap_status_changes == {} and v.current_gap_id == "g0"


def test_advance_picks_the_lowest_index_open_gap(cfg: EngineConfig) -> None:
    s = make_state(
        gaps=[make_gap("g0", status=GapStatus.EXHAUSTED), make_gap("g1"), make_gap("g2")]
    )
    assert RuleTrajectoryMonitor(cfg).advance(s).current_gap_id == "g1"
    assert RuleTrajectoryMonitor(cfg).advance(make_state()).current_gap_id is None

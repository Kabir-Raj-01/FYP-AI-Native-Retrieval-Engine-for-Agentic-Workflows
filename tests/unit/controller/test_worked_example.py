"""ACTION_POLICY §8 worked example, reproduced exactly (TESTING §4.3)."""

from __future__ import annotations

import math

import pytest

from engine.config import EngineConfig
from engine.controller.policy import RuleUtilityController
from engine.state.models import Action, NeedType, RetrievalMode, RetrievalRecord
from tests.builders import make_budget, make_gap, make_state
from tests.fakes import FakeEmbedder

PENDING = "pending query"
GAP = "the gap description"
EARLIER_DENSE = "an earlier dense query"
VECTORS = {
    PENDING: [1.0, 0.0, 0.0, 0.0],
    GAP: [0.8, 0.6, 0.0, 0.0],  # gap_align = cos(q, gap) = 0.8
    EARLIER_DENSE: [0.9, 0.0, math.sqrt(1 - 0.81), 0.0],  # R(DENSE) = 0.9
}


def test_worked_example_switches_to_sparse_hybrid(cfg: EngineConfig) -> None:
    state = make_state(
        gaps=[make_gap("g0", description=GAP, query=PENDING, need_type=NeedType.MIXED)],
        current_gap_id="g0",
        needs_initial_plan=False,
        budget=make_budget(
            max_steps=8,
            steps_used=2,
            max_retrieval_calls=4,
            retrieval_calls_used=2,
            max_llm_calls=8,
            llm_calls_used=1,
        ),  # pressure = max(2/8, 2/4, 1/8, 0) = 0.5
        retrieval_history=[
            RetrievalRecord(
                step=1,
                action=Action.DENSE,
                mode=RetrievalMode.DENSE,
                query=EARLIER_DENSE,
                k=20,
                candidate_ids=[],
                new_candidate_ids=[],
                cache_hit=False,
                latency_ms=1.0,
            )
        ],
        action_gain_ema={Action.DENSE: 0.2},  # SPARSE_HYBRID uses the prior 0.5
        stagnation_count=1,  # TRANSFORM eligible, G = 1/2
    )
    latency = cfg.controller.default_latency_ms.model_dump()  # DENSE 60, SH 90, TRANSFORM 1500
    controller = RuleUtilityController(cfg, FakeEmbedder(dimension=4, vectors=VECTORS), latency)
    d = controller.select_action(state, None)

    assert d.eligible == [Action.SPARSE_HYBRID, Action.DENSE, Action.TRANSFORM]
    expected = {
        Action.DENSE: (0.59, 0.375, 0.9, 0.04, -0.18125),
        Action.SPARSE_HYBRID: (0.71, 0.375, 0.0, 0.06, 0.06875),
        Action.TRANSFORM: (0.5, 0.75, 0.0, 1.0, -0.3125),
    }
    for action, (g, c, r, lat, u) in expected.items():
        s = d.scores[action]
        assert pytest.approx((g, c, r, lat, u)) == (s.G, s.C, s.R, s.L, s.U), action
    assert (d.action, d.rule, d.plan_mode, d.stop_reason) == (
        Action.SPARSE_HYBRID,
        "utility_argmax",
        None,
        None,
    )

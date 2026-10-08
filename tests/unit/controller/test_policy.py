"""engine/controller/policy.py: rules, preconditions, utility, tie-break, fixed policy
(ACTION_POLICY §2-§4, §7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from engine.config import EngineConfig
from engine.controller.policy import FIXED_POLICY_RULE, RuleUtilityController, load_latency_ms
from engine.errors import ConfigError
from engine.state.models import (
    Action,
    CacheHit,
    GapStatus,
    LastResult,
    NeedType,
    PlanMode,
    QueryRecord,
    RetrievalMode,
    RetrievalRecord,
    RetrievalState,
    StopReason,
)
from tests.builders import make_budget, make_cache_entry, make_evidence, make_gap, make_state
from tests.fakes import FakeEmbedder


def state(need: NeedType = NeedType.SEMANTIC, query: str = "q0", **kw: Any) -> RetrievalState:
    fields: dict[str, Any] = {
        "gaps": [make_gap("g0", query=query, need_type=need)],
        "current_gap_id": "g0",
        "needs_initial_plan": False,
    }
    fields.update(kw)
    return make_state(**fields)


def issued(action: Action, query: str, step: int = 1) -> RetrievalRecord:
    mode = RetrievalMode.DENSE if action == Action.DENSE else RetrievalMode.HYBRID
    return RetrievalRecord(
        step=step, action=action, mode=mode, query=query, k=20,
        candidate_ids=[], new_candidate_ids=[], cache_hit=False, latency_ms=1.0,
    )  # fmt: skip


def controller(cfg: EngineConfig) -> RuleUtilityController:
    return RuleUtilityController(
        cfg, FakeEmbedder(), cfg.controller.default_latency_ms.model_dump()
    )


def test_budget_guard_on_spent_steps(cfg: EngineConfig) -> None:
    d = controller(cfg).select_action(state(budget=make_budget(max_steps=2, steps_used=2)), None)
    assert (d.action, d.stop_reason, d.rule) == (Action.STOP, StopReason.MAX_STEPS, "budget_guard")


def test_spent_retrieval_budget_stops_unless_reuse_is_possible(cfg: EngineConfig) -> None:
    s = state(budget=make_budget(max_retrieval_calls=2, retrieval_calls_used=2))
    d = controller(cfg).select_action(s, None)
    assert (d.action, d.stop_reason) == (Action.STOP, StopReason.MAX_RETRIEVAL_CALLS)
    hit = CacheHit(entry=make_cache_entry(), similarity=1.0, match="exact")
    assert controller(cfg).select_action(s, hit).action == Action.REUSE


def test_preferred_mode_first_then_the_strategy_switch(cfg: EngineConfig) -> None:
    c = controller(cfg)
    assert c.select_action(state(NeedType.SEMANTIC), None).eligible == [Action.DENSE]
    assert c.select_action(state(NeedType.LEXICAL), None).eligible == [Action.SPARSE_HYBRID]
    assert c.select_action(state(NeedType.MIXED), None).eligible == [
        Action.SPARSE_HYBRID,
        Action.DENSE,
    ]
    # the preferred mode already issued the query -> the other mode opens up
    tried = state(NeedType.SEMANTIC, retrieval_history=[issued(Action.DENSE, "q0")])
    assert c.select_action(tried, None).eligible == [Action.SPARSE_HYBRID, Action.TRANSFORM][:1]
    # stagnation also opens the other mode (and TRANSFORM)
    stuck = state(NeedType.SEMANTIC, stagnation_count=1)
    assert c.select_action(stuck, None).eligible == [
        Action.SPARSE_HYBRID,
        Action.DENSE,
        Action.TRANSFORM,
    ]


def test_both_modes_issued_makes_transform_eligible_with_full_gain(cfg: EngineConfig) -> None:
    s = state(
        NeedType.MIXED,
        retrieval_history=[issued(Action.DENSE, "q0"), issued(Action.SPARSE_HYBRID, "q0", 2)],
        budget=make_budget(retrieval_calls_used=2),
    )
    d = controller(cfg).select_action(s, None)
    assert d.eligible == [Action.TRANSFORM]
    assert d.scores[Action.TRANSFORM].G == 1.0


def test_no_eligible_action_reason(cfg: EngineConfig) -> None:
    history = [issued(Action.DENSE, "q0"), issued(Action.SPARSE_HYBRID, "q0", 2)]
    llm_spent = state(
        NeedType.MIXED,
        retrieval_history=history,
        budget=make_budget(retrieval_calls_used=2, max_llm_calls=1, llm_calls_used=1),
    )
    d = controller(cfg).select_action(llm_spent, None)
    assert (d.action, d.stop_reason, d.rule) == (
        Action.STOP,
        StopReason.MAX_LLM_CALLS,
        "no_eligible_action",
    )
    cfg.ablation.planner = False
    d = controller(cfg).select_action(llm_spent, None)
    assert d.stop_reason == StopReason.STAGNATION


def test_min_utility_stop_only_after_the_current_gap_was_searched(cfg: EngineConfig) -> None:
    cfg.controller.min_utility = 0.99
    fresh = controller(cfg).select_action(state(), None)
    assert fresh.action == Action.DENSE  # a task always retrieves at least once
    # an earlier gap used the retrieval budget, but this gap's query was never issued:
    # every gap gets its first retrieval (ACTION_POLICY §2 step 8, amended 2026-10-08)
    other_gap = state(
        budget=make_budget(retrieval_calls_used=1),
        retrieval_history=[issued(Action.SPARSE_HYBRID, "an earlier gap's query")],
    )
    assert controller(cfg).select_action(other_gap, None).action == Action.DENSE
    searched = state(
        budget=make_budget(retrieval_calls_used=1),
        retrieval_history=[issued(Action.SPARSE_HYBRID, "Q0")],  # normalised match
    )
    d = controller(cfg).select_action(searched, None)
    assert (d.action, d.stop_reason, d.rule) == (
        Action.STOP,
        StopReason.LOW_MARGINAL_UTILITY,
        "min_utility",
    )
    assert Action.DENSE in d.scores  # the scores are still recorded


def test_ties_prefer_sparse_hybrid_over_dense(cfg: EngineConfig) -> None:
    cfg.controller.type_match.DENSE.mixed = cfg.controller.type_match.SPARSE_HYBRID.mixed
    cfg.controller.default_latency_ms.SPARSE_HYBRID = cfg.controller.default_latency_ms.DENSE
    d = controller(cfg).select_action(state(NeedType.MIXED), None)
    assert pytest.approx(d.scores[Action.SPARSE_HYBRID].U) == d.scores[Action.DENSE].U
    assert d.action == Action.SPARSE_HYBRID


def test_plan_mode_rewrite_until_two_failed_queries_then_split(cfg: EngineConfig) -> None:
    w = cfg.controller.weights  # gain only, so TRANSFORM is chosen and plan_mode is set
    w.alpha, w.beta, w.gamma, w.delta = 1.0, 0.0, 0.0, 0.0

    def with_queries(n: int, **kw: Any) -> RetrievalState:
        qs = [f"q{i}" for i in range(n)]
        return state(
            NeedType.SEMANTIC,
            query=qs[-1],
            stagnation_count=1,
            query_history=[
                QueryRecord(
                    step=1, gap_id="g0", query=q, need_type=NeedType.SEMANTIC, source="planner"
                )
                for q in qs
            ],
            retrieval_history=[
                issued(a, q) for q in qs for a in (Action.DENSE, Action.SPARSE_HYBRID)
            ],
            budget=make_budget(max_retrieval_calls=10, retrieval_calls_used=2 * n),
            **kw,
        )

    c = controller(cfg)
    assert c.select_action(with_queries(1), None).plan_mode == PlanMode.REWRITE
    assert c.select_action(with_queries(2), None).plan_mode == PlanMode.SPLIT
    cfg.planner.max_gaps = 1  # no room for children
    assert controller(cfg).select_action(with_queries(2), None).plan_mode == PlanMode.REWRITE


def test_rerank_expand_needs_an_ambiguous_last_result_for_the_pending_query(
    cfg: EngineConfig,
) -> None:
    def last(top: list[float], query: str = "q0", depth: int = 20) -> LastResult:
        return LastResult(
            step=1, query=query, mode=RetrievalMode.DENSE, depth=depth,
            candidate_ids=["c1", "c2"], top_relevances=top,
        )  # fmt: skip

    c = controller(cfg)
    history = [issued(Action.DENSE, "q0")]
    ok = state(retrieval_history=history, last_result=last([0.9, 0.5, 0.1]))
    d = c.select_action(ok, None)
    assert Action.RERANK_EXPAND in d.eligible
    ema = cfg.controller.prior_action_gain
    assert pytest.approx(0.5 * (1 / 3) + 0.5 * ema) == d.scores[Action.RERANK_EXPAND].G
    for bad in (
        last([0.9, 0.1]),  # nothing in the band
        last([0.5], query="another query"),  # not for the pending query
        last([0.5], depth=cfg.retrieval.max_k),  # already at max depth
    ):
        assert (
            Action.RERANK_EXPAND
            not in c.select_action(state(retrieval_history=history, last_result=bad), None).eligible
        )


def test_rerank_expand_redundancy_is_the_share_already_in_evidence(cfg: EngineConfig) -> None:
    s = state(
        retrieval_history=[issued(Action.DENSE, "q0")],
        evidence=[make_evidence("c1")],
        last_result=LastResult(
            step=1,
            query="q0",
            mode=RetrievalMode.DENSE,
            depth=20,
            candidate_ids=["c1", "c2"],
            top_relevances=[0.5],
        ),
    )
    d = controller(cfg).select_action(s, None)
    assert d.scores[Action.RERANK_EXPAND].R == 0.5


def test_reuse_gain_is_similarity_times_share_not_yet_evidence(cfg: EngineConfig) -> None:
    entry = make_cache_entry()  # candidates c1, c2
    hit = CacheHit(entry=entry, similarity=0.9, match="semantic")
    d = controller(cfg).select_action(state(evidence=[make_evidence("c1")]), hit)
    assert pytest.approx(0.9 * 0.5) == d.scores[Action.REUSE].G
    cfg.ablation.cache = False
    assert Action.REUSE not in controller(cfg).select_action(state(), hit).eligible


def test_budget_awareness_and_redundancy_ablations(cfg: EngineConfig) -> None:
    s = state(
        NeedType.MIXED,
        budget=make_budget(max_steps=4, steps_used=2),
        retrieval_history=[issued(Action.DENSE, "q0 similar")],
    )
    on = controller(cfg).select_action(s, None).scores[Action.DENSE]
    cfg.ablation.budget_awareness = False
    cfg.ablation.redundancy_signal = False
    off = controller(cfg).select_action(s, None).scores[Action.DENSE]
    assert pytest.approx(0.5 * (0.5 + 0.5 * 0.5)) == on.C and pytest.approx(0.25) == off.C
    assert off.R == 0.0


def test_fixed_policy_dense_then_sparse_then_gap_stop(cfg: EngineConfig) -> None:
    cfg.ablation.adaptive_actions = False
    c = controller(cfg)
    hit = CacheHit(entry=make_cache_entry(), similarity=1.0, match="exact")
    first = c.select_action(state(NeedType.SEMANTIC), hit)
    assert (first.action, first.rule) == (Action.DENSE, FIXED_POLICY_RULE)  # never REUSE
    second = c.select_action(
        state(NeedType.SEMANTIC, retrieval_history=[issued(Action.DENSE, "q0")]), None
    )
    assert second.action == Action.SPARSE_HYBRID
    both = state(
        NeedType.SEMANTIC,
        retrieval_history=[issued(Action.DENSE, "q0"), issued(Action.SPARSE_HYBRID, "q0", 2)],
        budget=make_budget(retrieval_calls_used=2),
    )
    d = c.select_action(both, None)
    assert (d.action, d.stop_reason, d.rule) == (
        Action.STOP,
        StopReason.STAGNATION,
        FIXED_POLICY_RULE,
    )
    lexical = c.select_action(state(NeedType.LEXICAL), None)
    assert lexical.action == Action.SPARSE_HYBRID


def test_decisions_are_deterministic(cfg: EngineConfig) -> None:
    s = state(NeedType.MIXED, stagnation_count=1)
    assert controller(cfg).select_action(s, None) == controller(cfg).select_action(s, None)


def test_gap_alignment_ignores_closed_gaps(cfg: EngineConfig) -> None:
    embedder = FakeEmbedder(
        dimension=2, vectors={"q0": [1.0, 0.0], "closed": [1.0, 0.0], "open": [0.0, 1.0]}
    )
    s = state(
        gaps=[
            make_gap(
                "g0", description="closed", query="q0", status=GapStatus.RESOLVED, resolved_by="c1"
            ),
            make_gap("g1", description="open", query="q0"),
        ],
        current_gap_id="g1",
    )
    c = RuleUtilityController(cfg, embedder, cfg.controller.default_latency_ms.model_dump())
    g = c.select_action(s, None).scores[Action.DENSE].G
    w = cfg.controller.gain_subweights
    assert g == pytest.approx(w.recent_gain * cfg.controller.prior_action_gain + w.type_match * 1.0)


def test_latency_profile_file_or_defaults(cfg: EngineConfig, tmp_path: Path) -> None:
    cfg.controller.latency_profile_path = tmp_path / "missing.yaml"
    assert load_latency_ms(cfg) == {
        k: float(v) for k, v in cfg.controller.default_latency_ms.model_dump().items()
    }
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "latency_ms: {REUSE: 4, DENSE: 140, SPARSE_HYBRID: 150, RERANK_EXPAND: 170,"
        " TRANSFORM: 1500}\n",
        encoding="utf-8",
    )
    cfg.controller.latency_profile_path = profile
    assert load_latency_ms(cfg)["DENSE"] == 140.0
    profile.write_text("latency_ms: {DENSE: 140}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="exactly the keys"):
        load_latency_ms(cfg)


def test_reuse_only_once_per_pending_query(cfg: EngineConfig) -> None:
    hit = CacheHit(entry=make_cache_entry(), similarity=1.0, match="exact")
    reused = RetrievalRecord(
        step=2, action=Action.REUSE, mode=RetrievalMode.DENSE, query="q0", k=2,
        candidate_ids=[], new_candidate_ids=[], cache_hit=True, latency_ms=0.0,
    )  # fmt: skip
    s = state(retrieval_history=[issued(Action.DENSE, "q0"), reused])
    assert Action.REUSE not in controller(cfg).select_action(s, hit).eligible
    other = state(query="a new query", retrieval_history=[issued(Action.DENSE, "q0"), reused])
    assert Action.REUSE in controller(cfg).select_action(other, hit).eligible

"""engine/planner/llm_planner.py with FakeLLMClient (PLANNER_SPEC §6, §8)."""

from __future__ import annotations

from typing import Any

import pytest

from engine.config import EngineConfig
from engine.errors import LLMBudgetExceeded, LLMError, LLMOutputError
from engine.planner.llm_planner import LLMPlanner
from engine.planner.prompting import prompt_hashes
from engine.state.models import NeedType, PlanMode, PlanStatus, QueryRecord, RetrievalState
from tests.builders import make_budget, make_evidence, make_gap, make_state
from tests.fakes import FakeLLMClient

GOAL = "Who directs the research lab that built the KX-41 drone?"
VALID = {
    "subgoals": [
        {"description": "who built the KX-41", "query": "KX-41 developer", "need_type": "lexical"},
        {"description": "who directs that lab", "query": "lab director", "need_type": "semantic"},
    ],
    "referenced_entities": ["KX-41"],
}
HALLUCINATED = {
    "subgoals": [{"description": "x", "query": "Orrin Vale Holdings", "need_type": "lexical"}],
    "referenced_entities": ["Orrin Vale Holdings"],
}


def init_state(**kw: Any) -> RetrievalState:
    return make_state(goal=GOAL, **kw)


def statuses(outcome: Any) -> list[tuple[PlanStatus, str | None]]:
    return [(e.status, e.reason) for e in outcome.events]


def test_valid_plan_is_accepted_first_time(cfg: EngineConfig) -> None:
    llm = FakeLLMClient({"planner": [VALID]}, input_tokens=50, output_tokens=30)
    outcome = LLMPlanner(cfg, llm).plan(init_state(), PlanMode.INIT)
    assert [s.query for s in outcome.result.subgoals] == ["KX-41 developer", "lab director"]
    assert statuses(outcome) == [(PlanStatus.ACCEPTED, None)]
    assert (outcome.llm_calls, outcome.tokens) == (1, 80)
    assert outcome.events[0].step == 1


def test_hallucinated_entity_is_retried_and_the_second_answer_used(cfg: EngineConfig) -> None:
    llm = FakeLLMClient({"planner": [HALLUCINATED, VALID]})
    outcome = LLMPlanner(cfg, llm).plan(init_state(), PlanMode.INIT)
    assert statuses(outcome) == [
        (PlanStatus.REJECTED, "UNGROUNDED_ENTITY"),
        (PlanStatus.ACCEPTED, None),
    ]
    retry_prompt = llm.requests[1].messages[-1].content
    assert "rejected (UNGROUNDED_ENTITY)" in retry_prompt and "only entities" in retry_prompt


def test_two_bad_answers_fall_back_after_exactly_max_attempts_calls(cfg: EngineConfig) -> None:
    llm = FakeLLMClient({"planner": [LLMOutputError("bad json"), LLMError("timeout")]})
    outcome = LLMPlanner(cfg, llm).plan(init_state(), PlanMode.INIT)
    assert len(llm.requests) == cfg.planner.max_attempts == 2
    assert statuses(outcome) == [
        (PlanStatus.REJECTED, "SCHEMA"),
        (PlanStatus.REJECTED, "LLM_ERROR"),
        (PlanStatus.FALLBACK, None),
    ]
    (sg,) = outcome.result.subgoals
    assert (sg.description, sg.query, sg.need_type) == (GOAL, GOAL, NeedType.LEXICAL)
    assert outcome.llm_calls == 2  # failed attempts count; tokens only from replies
    assert outcome.tokens == 0


def test_no_llm_call_without_task_llm_budget(cfg: EngineConfig) -> None:
    llm = FakeLLMClient()
    state = init_state(budget=make_budget(max_llm_calls=1, llm_calls_used=1))
    outcome = LLMPlanner(cfg, llm).plan(state, PlanMode.INIT)
    assert llm.requests == [] and statuses(outcome) == [(PlanStatus.FALLBACK, None)]
    assert outcome.llm_calls == 0


def test_budget_allows_only_the_remaining_calls(cfg: EngineConfig) -> None:
    llm = FakeLLMClient({"planner": [HALLUCINATED, VALID]})
    state = init_state(budget=make_budget(max_llm_calls=2, llm_calls_used=1))
    outcome = LLMPlanner(cfg, llm).plan(state, PlanMode.INIT)
    assert len(llm.requests) == 1 and outcome.events[-1].status == PlanStatus.FALLBACK


def test_per_run_cap_propagates(cfg: EngineConfig) -> None:
    llm = FakeLLMClient({"planner": [LLMBudgetExceeded("cap")]})
    with pytest.raises(LLMBudgetExceeded):
        LLMPlanner(cfg, llm).plan(init_state(), PlanMode.INIT)


def test_ablations_skip_the_llm(cfg: EngineConfig) -> None:
    cfg.ablation.planner = False
    llm = FakeLLMClient()
    assert LLMPlanner(cfg, llm).plan(init_state(), PlanMode.INIT).result.subgoals[0].query == GOAL
    cfg.ablation.planner = True
    cfg.ablation.gap_tracking = False  # INIT = exactly one gap, the goal
    outcome = LLMPlanner(cfg, llm).plan(init_state(), PlanMode.INIT)
    assert llm.requests == [] and len(outcome.result.subgoals) == 1


def rewrite_state(queries: list[str]) -> RetrievalState:
    return make_state(
        goal=GOAL,
        gaps=[make_gap("g0", description="the lab that built the drone", query=queries[-1])],
        current_gap_id="g0",
        needs_initial_plan=False,
        query_history=[
            QueryRecord(step=1, gap_id="g0", query=q, need_type=NeedType.MIXED, source="planner")
            for q in queries
        ],
    )


def test_rewrite_fallback_uses_the_gap_description_then_the_goal(cfg: EngineConfig) -> None:
    llm = FakeLLMClient({"planner": [LLMError("x"), LLMError("y")]})
    outcome = LLMPlanner(cfg, llm).plan(rewrite_state(["drone lab"]), PlanMode.REWRITE)
    assert outcome.result.subgoals[0].query == "the lab that built the drone"
    llm = FakeLLMClient({"planner": [LLMError("x"), LLMError("y")]})
    tried = rewrite_state(["drone lab", "the lab that built the drone"])
    outcome = LLMPlanner(cfg, llm).plan(tried, PlanMode.SPLIT)
    (sg,) = outcome.result.subgoals  # SPLIT fallback is a one-subgoal rewrite
    assert sg.query == GOAL and sg.description == "the lab that built the drone"


def test_rewrite_prompt_lists_previous_queries_and_other_gaps(cfg: EngineConfig) -> None:
    llm = FakeLLMClient({"planner": [LLMError("x"), LLMError("y")]})
    state = rewrite_state(["drone lab", "KX-41 maker"])
    state = state.model_copy(
        update={"gaps": [*state.gaps, make_gap("g1", description="director name")]}
    )
    LLMPlanner(cfg, llm).plan(state, PlanMode.REWRITE)
    user = llm.requests[0].messages[1].content
    assert "- drone lab" in user and "- KX-41 maker" in user and "- director name" in user


def test_evidence_summaries_are_capped_and_best_first(cfg: EngineConfig) -> None:
    cfg.planner.max_evidence_summaries = 2
    evidence = [
        make_evidence(f"e{i}", summary=f"summary {i}", final_score=i / 10) for i in range(4)
    ]
    llm = FakeLLMClient({"planner": [VALID]})
    LLMPlanner(cfg, llm).plan(init_state(evidence=evidence), PlanMode.INIT)
    user = llm.requests[0].messages[1].content
    assert "summary 3" in user and "summary 2" in user and "summary 1" not in user


def test_prompt_hashes_cover_every_template() -> None:
    hashes = prompt_hashes()
    assert set(hashes) == {
        "planner/_evidence.j2",
        "planner/_rules.j2",
        "planner/init.j2",
        "planner/rewrite.j2",
        "planner/split.j2",
    }
    assert all(len(h) == 64 for h in hashes.values())

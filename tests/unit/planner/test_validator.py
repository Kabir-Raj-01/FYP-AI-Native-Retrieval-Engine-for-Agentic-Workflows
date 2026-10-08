"""engine/planner/validator.py: reason codes in order (PLANNER_SPEC §5, §8)."""

from __future__ import annotations

from typing import Any

import pytest

from engine.config import EngineConfig
from engine.planner.validator import PlanRejectedError, validate
from engine.state.models import (
    GapStatus,
    NeedType,
    PlanLLMOutput,
    PlanMode,
    QueryRecord,
    RetrievalState,
    SubgoalPlan,
)
from tests.builders import make_gap, make_state

GOAL = "Who directs the research lab that built the KX-41 drone?"
SOURCES = [GOAL, "The KX-41 inspection drone was developed by the Talvane Robotics Lab."]


def out(*queries: str, entities: tuple[str, ...] = (), rationale: str = "") -> PlanLLMOutput:
    return PlanLLMOutput(
        subgoals=[
            SubgoalPlan(description=f"need {q}", query=q, need_type=NeedType.MIXED) for q in queries
        ],
        referenced_entities=list(entities),
        rationale=rationale,
    )


def code(output: PlanLLMOutput, mode: PlanMode, state: RetrievalState, cfg: EngineConfig) -> str:
    with pytest.raises(PlanRejectedError) as exc:
        validate(output, mode, state, SOURCES, cfg)
    return exc.value.reason


def rewrite_state(**kw: Any) -> RetrievalState:
    fields: dict[str, Any] = {
        "gaps": [make_gap("g0", query="KX-41 builder")],
        "current_gap_id": "g0",
        "query_history": [
            QueryRecord(
                step=1,
                gap_id="g0",
                query="KX-41 builder",
                need_type=NeedType.MIXED,
                source="planner",
            )
        ],
    }
    fields.update(kw)
    return make_state(**fields)


def test_valid_init_plan_is_accepted_with_truncated_rationale(cfg: EngineConfig) -> None:
    cfg.planner.rationale_max_chars = 5
    result = validate(
        out("KX-41 developer", entities=("KX-41",), rationale="long rationale"),
        PlanMode.INIT,
        make_state(),
        SOURCES,
        cfg,
    )
    assert (result.mode, result.rationale, len(result.subgoals)) == (PlanMode.INIT, "long ", 1)


def test_count_bounds_per_mode_and_gap_cap(cfg: EngineConfig) -> None:
    cfg.planner.max_subgoals = 3
    assert code(out("a", "b", "c", "d", "e"), PlanMode.INIT, make_state(), cfg) == "COUNT"
    assert code(out("new one", "new two"), PlanMode.REWRITE, rewrite_state(), cfg) == "COUNT"
    assert code(out("only one"), PlanMode.SPLIT, rewrite_state(), cfg) == "COUNT"
    cfg.planner.max_gaps = 2  # 1 live gap - 1 parent + 3 children = 3 > 2
    assert code(out("x one", "x two", "x three"), PlanMode.SPLIT, rewrite_state(), cfg) == "COUNT"


def test_reused_or_empty_query_is_rejected(cfg: EngineConfig) -> None:
    assert (
        code(out("kx-41 Builder"), PlanMode.REWRITE, rewrite_state(), cfg) == "EMPTY_OR_DUPLICATE"
    )
    assert code(out("   "), PlanMode.INIT, make_state(), cfg) == "EMPTY_OR_DUPLICATE"


def test_hallucinated_entity_is_ungrounded(cfg: EngineConfig) -> None:
    bad = out("Orrin Vale Holdings results", entities=("Orrin Vale Holdings",))
    assert code(bad, PlanMode.INIT, make_state(), cfg) == "UNGROUNDED_ENTITY"


def test_entity_used_but_not_declared(cfg: EngineConfig) -> None:
    sneaky = out("Orrin Vale Holdings results")  # in the query, not in referenced_entities
    assert code(sneaky, PlanMode.INIT, make_state(), cfg) == "UNDECLARED_ENTITY"
    # a groundable span may be left undeclared (it is in the goal)
    validate(out("KX-41 developer"), PlanMode.INIT, make_state(), SOURCES, cfg)


def test_keyword_queries_juxtaposing_grounded_names_are_accepted(cfg: EngineConfig) -> None:
    # seen live (qwen2.5:3b): names from the goal glued together in a keyword query
    goal = "Did The Verge report Apple defending its Google Search deal, per Forbes?"
    plan = PlanLLMOutput(
        subgoals=[
            SubgoalPlan(
                description="Find the report by The Verge on Apple's Google Search deal.",
                query="The Verge Apple Google Search deal",
                need_type=NeedType.LEXICAL,
            ),
            SubgoalPlan(
                description="Find what Forbes wrote.",
                query="Forbes Apple Google",
                need_type=NeedType.LEXICAL,
            ),
        ],
    )
    validate(plan, PlanMode.INIT, make_state(), [goal], cfg)
    invented = plan.model_copy(
        update={
            "subgoals": [
                SubgoalPlan(
                    description="Find it.", query="Orrin Vale Apple", need_type=NeedType.LEXICAL
                )
            ]
        }
    )
    assert code(invented, PlanMode.INIT, make_state(), cfg) == "UNDECLARED_ENTITY"


def test_split_within_bounds_is_accepted(cfg: EngineConfig) -> None:
    state = rewrite_state(
        gaps=[make_gap("g0", query="KX-41 builder"), make_gap("g1", status=GapStatus.EXHAUSTED)]
    )
    result = validate(out("KX-41 developer", "lab director"), PlanMode.SPLIT, state, SOURCES, cfg)
    assert result.mode == PlanMode.SPLIT and len(result.subgoals) == 2

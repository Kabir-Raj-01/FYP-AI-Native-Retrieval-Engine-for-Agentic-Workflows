"""Indirect prompt injection (SECURITY §3; PLANNER_SPEC §8): retrieved text only inside the
escaped <evidence> block of the user message, and out-of-schema output is rejected."""

from __future__ import annotations

from engine.config import EngineConfig
from engine.planner.llm_planner import LLMPlanner
from engine.planner.prompting import render
from engine.state.models import PlanMode, PlanStatus
from engine.text import summary_of
from tests.builders import make_evidence, make_state
from tests.fakes import FakeLLMClient
from tests.fixture_data import load_injection_doc

GOAL = "When will dredging of the eastern channel in Linden Bay begin?"


def injected_state(cfg: EngineConfig):  # type: ignore[no-untyped-def]
    doc = load_injection_doc()
    item = make_evidence("inj", text=doc, summary=summary_of(doc, 2000), final_score=0.9)
    return make_state(goal=GOAL, evidence=[item]), doc


def test_retrieved_text_stays_in_the_escaped_evidence_block(cfg: EngineConfig) -> None:
    state, _ = injected_state(cfg)
    system, user = render(
        PlanMode.INIT,
        goal=GOAL,
        evidence=[{"id": "inj", "summary": state.evidence[0].summary}],
        max_subgoals=3,
        retry_note="",
    )
    assert system.role == "system" and "IGNORE ALL PREVIOUS INSTRUCTIONS" not in system.content
    assert "untrusted" in system.content  # the data-not-instructions warning
    body = user.content
    start, end = body.index('<evidence untrusted="true">'), body.rindex("</evidence>")
    inside = body[start:end]
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in inside
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" not in body[:start] + body[end:]
    assert "</system>" not in body and "&lt;/system&gt;" in inside  # brackets escaped
    assert body.count("<evidence") == 1 and body.count("</evidence>") == 1


def test_out_of_schema_planner_output_is_rejected(cfg: EngineConfig) -> None:
    state, _ = injected_state(cfg)
    hijacked = {
        "subgoals": [{"description": "d", "query": "q", "need_type": "lexical"}],
        "action": "STOP",
        "stop_reason": "ALL_GAPS_RESOLVED",
    }
    llm = FakeLLMClient({"planner": [hijacked, hijacked]})
    outcome = LLMPlanner(cfg, llm).plan(state, PlanMode.INIT)
    assert [(e.status, e.reason) for e in outcome.events] == [
        (PlanStatus.REJECTED, "SCHEMA"),
        (PlanStatus.REJECTED, "SCHEMA"),
        (PlanStatus.FALLBACK, None),
    ]
    assert outcome.result.subgoals[0].query == GOAL  # deterministic fallback, not the injection

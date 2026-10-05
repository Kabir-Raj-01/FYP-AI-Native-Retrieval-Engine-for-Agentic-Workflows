"""Fault injection (TESTING §4.5) beyond the backend failure in test_loop.py."""

from __future__ import annotations

import pytest

from engine.errors import BackendError, LLMBudgetExceeded, LLMError
from engine.planner.llm_planner import LLMPlanner
from engine.state.models import (
    Action,
    ActionDecision,
    EvidenceUpdate,
    MonitorVerdict,
    PlanStatus,
    RetrievalState,
    TaskStatus,
)
from engine.state.sqlite import connect, read_cache_entries
from tests.conftest import EngineFactory
from tests.fakes import FakeController, FakeLLMClient, FakeMonitor

GOAL = "Linden Bay tram expansion council harbour"
INIT = {
    "subgoals": [
        {"description": "the tram plan", "query": "Linden Bay tram", "need_type": "mixed"}
    ],
    "referenced_entities": ["Linden Bay"],
}


def transform_engine(engine_with_fakes: EngineFactory, *responses: object) -> tuple:  # type: ignore[type-arg]
    from engine.config import load_config
    from tests.conftest import TEST_CONFIG_OVERRIDES

    llm = FakeLLMClient({"planner": [INIT, *responses]})
    cfg = load_config(environ={}, overrides=TEST_CONFIG_OVERRIDES)
    fake = engine_with_fakes(
        planner=LLMPlanner(cfg, llm),
        controller=FakeController([Action.DENSE, Action.TRANSFORM, Action.DENSE]),
    )
    fake.engine.create_task(GOAL, task_id="t-1")
    fake.engine.step("t-1")
    return fake, fake.engine.state("t-1")


def test_llm_error_during_transform_is_absorbed_by_the_planner_fallback(
    engine_with_fakes: EngineFactory,
) -> None:
    fake, _ = transform_engine(engine_with_fakes, LLMError("down"), LLMError("down"))
    trace = fake.engine.step("t-1")  # PLANNER_SPEC §6: rejected attempts, then the fallback
    assert [(e.status, e.reason) for e in trace.plan_events] == [
        (PlanStatus.REJECTED, "LLM_ERROR"),
        (PlanStatus.REJECTED, "LLM_ERROR"),
        (PlanStatus.FALLBACK, None),
    ]
    s = fake.engine.state("t-1")
    assert (s.status, s.budget.steps_used) == (TaskStatus.ACTIVE, 2)


def test_run_cap_during_transform_leaves_the_stored_state(engine_with_fakes: EngineFactory) -> None:
    fake, after_one = transform_engine(engine_with_fakes, LLMBudgetExceeded("cap"))
    with pytest.raises(LLMBudgetExceeded):
        fake.engine.step("t-1")
    assert fake.engine.state("t-1") == after_one  # still ACTIVE at version 1 (run aborts)


def test_no_orphan_cache_entry_after_a_failure_late_in_the_step(
    engine_with_fakes: EngineFactory,
) -> None:
    class FailsOnce(FakeMonitor):  # fails in `monitor`, after the entry was queued
        failed = False

        def observe(
            self, state: RetrievalState, decision: ActionDecision, update: EvidenceUpdate | None
        ) -> MonitorVerdict:
            if state.budget.steps_used == 1 and not self.failed:
                FailsOnce.failed = True
                raise BackendError("monitor-time failure")
            return super().observe(state, decision, update)

    fake = engine_with_fakes(
        monitor=FailsOnce(), controller=FakeController([Action.DENSE, Action.SPARSE_HYBRID])
    )
    fake.engine.create_task(GOAL, task_id="t-1")
    fake.engine.step("t-1")
    with pytest.raises(BackendError):
        fake.engine.step("t-1")
    entries = read_cache_entries(connect(fake.db), "t-1")
    assert [e.entry_id for e in entries] == ["step-1"]  # step 2's queued entry never committed
    hit = fake.parts.cache.lookup("t-1", GOAL, fake.engine.state("t-1").current_need_type)
    assert hit is not None and hit.entry.entry_id == "step-1"
    fake.engine.step("t-1")  # the retry commits its own step-2 entry exactly once
    assert [e.entry_id for e in read_cache_entries(connect(fake.db), "t-1")] == ["step-1", "step-2"]

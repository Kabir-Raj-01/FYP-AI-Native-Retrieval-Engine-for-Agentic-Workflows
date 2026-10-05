"""The controlled loop with the real Controller, TrajectoryMonitor, EvidenceManager and
state-aware Reranker (fake planner, fake cross-encoder scorer, fake retriever) — TESTING §4.2:
invariants hold after every iteration (``persist`` checks them) and ``max_steps`` gives
exactly that many real iterations."""

from __future__ import annotations

import re

import pytest

from engine.config import EngineConfig
from engine.controller.policy import RuleUtilityController
from engine.evidence.manager import ThresholdEvidenceManager
from engine.monitor.trajectory import RuleTrajectoryMonitor
from engine.rerank.state_aware import StateAwareReranker
from engine.state.models import Action, Budget, StopReason, TaskStatus
from tests.conftest import EngineFactory, FakeEngine
from tests.fakes import FakeEmbedder
from tests.fixture_data import MiniTask

TOKEN = re.compile(r"[a-z0-9]+")


def overlap_scorer(query: str, texts: list[str]) -> list[float]:
    """Relevance = share of the query's tokens found in the text."""
    q = set(TOKEN.findall(query.lower()))
    return [len(q & set(TOKEN.findall(t.lower()))) / max(1, len(q)) for t in texts]


def real_engine(cfg: EngineConfig, engine_with_fakes: EngineFactory) -> FakeEngine:
    embedder = FakeEmbedder()
    return engine_with_fakes(
        controller=RuleUtilityController(
            cfg, embedder, cfg.controller.default_latency_ms.model_dump()
        ),
        evidence=ThresholdEvidenceManager(cfg, embedder),
        monitor=RuleTrajectoryMonitor(cfg),
        reranker=StateAwareReranker(cfg, embedder, overlap_scorer),
        embedder=embedder,
    )


@pytest.mark.parametrize(
    "ablation",
    [None, "adaptive_actions", "gap_tracking", "cache", "state_aware_rerank", "budget_awareness"],
)
def test_every_mini_task_runs_to_a_stop_with_invariants_held(
    cfg: EngineConfig,
    engine_with_fakes: EngineFactory,
    mini_tasks: list[MiniTask],
    ablation: str | None,
) -> None:
    if ablation is not None:
        setattr(cfg.ablation, ablation, False)
    cfg.evidence.relevance_threshold = 0.3  # the overlap scorer's scale
    fake = real_engine(cfg, engine_with_fakes)
    for task in mini_tasks:
        fake.engine.create_task(task.question, task_id=task.task_id)
        bundle = fake.engine.run(task.task_id)  # persist raises on any invariant violation
        s = fake.engine.state(task.task_id)
        assert s.status == TaskStatus.STOPPED and bundle.stop_reason == s.stop_reason
        assert s.budget.steps_used <= s.budget.max_steps
        assert s.budget.retrieval_calls_used <= s.budget.max_retrieval_calls
        steps = fake.parts.traces.read_steps(task.task_id)
        assert [t.step for t in steps] == list(range(1, s.budget.steps_used + 1))
        assert all(t.rule != "budget_guard" or t.action == Action.STOP for t in steps)
        if ablation == "adaptive_actions":
            assert {t.action for t in steps} <= {Action.DENSE, Action.SPARSE_HYBRID, Action.STOP}


def test_max_steps_runs_exactly_n_real_iterations_with_real_components(
    cfg: EngineConfig, engine_with_fakes: EngineFactory
) -> None:
    w = cfg.controller.weights  # gain only + min_utility 0: never a utility STOP
    w.alpha, w.beta, w.gamma, w.delta = 1.0, 0.0, 0.0, 0.0
    cfg.controller.min_utility = 0.0
    cfg.evidence.relevance_threshold = 1.0  # nothing resolves, so only the step cap stops
    cfg.monitor.max_transforms_per_streak = 3  # and no gap can be exhausted within 3 steps
    fake = real_engine(cfg, engine_with_fakes)
    fake.engine.create_task(
        "Linden Bay tram expansion council harbour",
        Budget(max_steps=3, max_retrieval_calls=4, max_llm_calls=6, max_tokens=20000),
        task_id="t-1",
    )
    bundle = fake.engine.run("t-1")
    steps = fake.parts.traces.read_steps("t-1")
    assert bundle.stop_reason == StopReason.MAX_STEPS
    assert fake.engine.state("t-1").budget.steps_used == 3
    assert len(steps) == 3 and Action.STOP not in [t.action for t in steps]
    assert "budget_guard" not in [t.rule for t in steps]  # ADR-009 §12: never fires


def test_the_real_planner_drives_the_loop(
    cfg: EngineConfig, engine_with_fakes: EngineFactory
) -> None:
    from engine.errors import LLMError
    from engine.planner.llm_planner import LLMPlanner
    from engine.state.models import PlanStatus
    from tests.fakes import FakeLLMClient

    init = {
        "subgoals": [
            {
                "description": "the drone's maker",
                "query": "KX-41 drone developer",
                "need_type": "lexical",
            },
            {
                "description": "the lab's director",
                "query": "Talvane lab director",
                "need_type": "mixed",
            },
        ],
        "referenced_entities": ["KX-41"],
    }
    llm = FakeLLMClient({"planner": [init, *[LLMError("down")] * 20]})
    embedder = FakeEmbedder()
    cfg.evidence.relevance_threshold = 0.3
    fake = engine_with_fakes(
        planner=LLMPlanner(cfg, llm),
        controller=RuleUtilityController(
            cfg, embedder, cfg.controller.default_latency_ms.model_dump()
        ),
        evidence=ThresholdEvidenceManager(cfg, embedder),
        monitor=RuleTrajectoryMonitor(cfg),
        reranker=StateAwareReranker(cfg, embedder, overlap_scorer),
        embedder=embedder,
    )
    fake.engine.create_task(
        "Who directs the research lab that built the KX-41 drone?", task_id="t-1"
    )
    bundle = fake.engine.run("t-1")
    s = fake.engine.state("t-1")
    assert [g.query for g in s.gaps[:2]] == ["KX-41 drone developer", "Talvane lab director"]
    assert s.plan_events[0].status == PlanStatus.ACCEPTED
    assert s.budget.llm_calls_used == sum(t.llm_calls for t in fake.parts.traces.read_steps("t-1"))
    assert bundle.stop_reason is not None

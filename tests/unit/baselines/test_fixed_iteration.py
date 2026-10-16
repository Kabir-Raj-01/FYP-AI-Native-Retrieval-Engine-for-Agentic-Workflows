"""baselines/fixed_iteration.py with fakes (BASELINES_SPEC §2 fixed_iteration)."""

from __future__ import annotations

from pathlib import Path

from baselines import fixed_iteration
from baselines.base import BaselineContext
from engine.config import EngineConfig
from engine.monitor.trace import JsonlTraceWriter
from engine.state.models import Action, NeedType, PlanMode, RunContext, StopReason, SubgoalPlan
from eval.datasets.base import Task
from tests.builders import NOW
from tests.conftest import EngineFactory, FakeEngine
from tests.fakes import FakePlanner, FakeReranker

TASK = Task(
    task_id="multihop:0001",
    dataset="multihop",
    question="Linden Bay tram expansion council harbour",
    answer="a",
    question_type="inference_query",
    gold_groups=[],
    split="dev",
)
SUBGOALS = [
    SubgoalPlan(description="tram plan", query="Linden Bay tram", need_type=NeedType.SEMANTIC),
    SubgoalPlan(description="council vote", query="council harbour", need_type=NeedType.LEXICAL),
]


def run(fake: FakeEngine, cfg: EngineConfig, tmp_path: Path, hops: int) -> tuple[object, list]:  # type: ignore[type-arg]
    traces = JsonlTraceWriter(tmp_path / "fi", RunContext(run_id="r", system="fixed_iteration"))
    ctx = BaselineContext(
        cfg=cfg, engine=fake.engine, traces=traces, retrieval_budget=hops, now=lambda: NOW
    )
    bundle = fixed_iteration.run_task(TASK, ctx)
    return bundle, traces.read_steps(TASK.task_id)


def test_subgoal_queries_then_a_planner_rewrite(
    cfg: EngineConfig, engine_with_fakes: EngineFactory, tmp_path: Path
) -> None:
    planner = FakePlanner(init=SUBGOALS)
    reranker = FakeReranker(default=0.6)
    fake = engine_with_fakes(planner=planner, reranker=reranker)
    bundle, steps = run(fake, cfg, tmp_path, hops=3)
    queries = [r.query for r in fake.retriever.requests]
    assert queries == ["Linden Bay tram", "council harbour", "council harbour rewritten"]
    assert reranker.goals == [TASK.question] * 3  # every hop's relevance knows the goal
    assert planner.calls == [PlanMode.INIT, PlanMode.REWRITE]
    assert [s.step for s in steps] == [1, 2, 3]
    assert {s.action for s in steps} == {Action.SPARSE_HYBRID}  # fixed_iteration.mode: hybrid
    assert [s.stop_reason for s in steps] == [None, None, StopReason.FIXED_ITERATIONS_COMPLETE]
    assert steps[-1].budget_after.retrieval_calls_used == 3  # always exactly N calls
    assert (steps[0].llm_calls, steps[2].llm_calls) == (1, 1)
    assert [e.mode for e in steps[2].plan_events] == [PlanMode.REWRITE]
    assert bundle.stop_reason == StopReason.FIXED_ITERATIONS_COMPLETE  # type: ignore[attr-defined]
    ids = [e.evidence_id for e in bundle.evidence]  # type: ignore[attr-defined]
    assert len(ids) == len(set(ids)) <= cfg.rerank.top_n_output


def test_spent_llm_budget_reuses_the_last_query_deeper(
    cfg: EngineConfig, engine_with_fakes: EngineFactory, tmp_path: Path
) -> None:
    cfg.budget_defaults.max_llm_calls = 1  # INIT spends it
    planner = FakePlanner(init=SUBGOALS[:1])
    fake = engine_with_fakes(planner=planner, reranker=FakeReranker(default=0.6))
    run(fake, cfg, tmp_path, hops=3)
    r = cfg.retrieval
    assert [(q.query, q.k) for q in fake.retriever.requests] == [
        ("Linden Bay tram", r.default_k),
        ("Linden Bay tram", min(r.default_k + r.expand_step_k, r.max_k)),
        ("Linden Bay tram", min(r.default_k + 2 * r.expand_step_k, r.max_k)),
    ]
    assert planner.calls == [PlanMode.INIT]


def test_only_relevant_candidates_are_accepted(
    cfg: EngineConfig, engine_with_fakes: EngineFactory, tmp_path: Path
) -> None:
    fake = engine_with_fakes(planner=FakePlanner(init=SUBGOALS), reranker=FakeReranker(default=0.1))
    bundle, steps = run(fake, cfg, tmp_path, hops=2)
    assert bundle.evidence == []  # type: ignore[attr-defined]
    assert all(s.accepted_ids == [] for s in steps)

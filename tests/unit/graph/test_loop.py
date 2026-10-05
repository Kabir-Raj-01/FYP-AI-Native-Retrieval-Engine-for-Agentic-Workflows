"""The controlled loop end to end on fakes: graph nodes + Engine.create_task/step/run
(ARCHITECTURE §3-§5, §8; ADR-009; TESTING §4.2 and §4.5)."""

from __future__ import annotations

import threading
from itertools import count

import pytest

from engine.config import EngineConfig
from engine.core import Engine
from engine.errors import (
    BackendError,
    ConfigError,
    RunTimeout,
    TaskConflict,
    TaskExists,
    TaskNotActive,
)
from engine.state.models import (
    Action,
    ActionDecision,
    Budget,
    CacheHit,
    GapStatus,
    NeedType,
    PlanMode,
    PlanResult,
    RetrievalMode,
    RetrievalState,
    StopReason,
    SubgoalPlan,
    TaskStatus,
)
from engine.state.sqlite import connect, read_cache_entries
from tests.conftest import EngineFactory, FakeEngine
from tests.fakes import (
    CorpusChunk,
    FakeController,
    FakeEvidenceManager,
    FakePlanner,
    FakeRetriever,
)

GOAL = "Linden Bay tram expansion council harbour"


def budget(max_steps: int = 8, max_retrieval_calls: int = 4) -> Budget:
    return Budget(
        max_steps=max_steps,
        max_retrieval_calls=max_retrieval_calls,
        max_llm_calls=6,
        max_tokens=20000,
    )


def start(fake: FakeEngine, **kw: object) -> RetrievalState:
    return fake.engine.create_task(GOAL, task_id="t-1", **kw)  # type: ignore[arg-type]


def two_gaps() -> FakePlanner:
    return FakePlanner(
        init=[
            SubgoalPlan(
                description="tram plan", query="Linden Bay tram", need_type=NeedType.SEMANTIC
            ),
            SubgoalPlan(
                description="council vote", query="council harbour", need_type=NeedType.LEXICAL
            ),
        ]
    )


def test_first_step_plans_retrieves_scores_and_persists(engine_with_fakes: EngineFactory) -> None:
    fake = engine_with_fakes()
    start(fake)
    trace = fake.engine.step("t-1")
    s = fake.engine.state("t-1")
    # maybe_plan: one gap from INIT, made current by advance()
    assert [g.gap_id for g in s.gaps] == ["g0"] and s.current_gap_id == "g0"
    assert s.needs_initial_plan is False
    assert [(q.step, q.gap_id, q.source) for q in s.query_history] == [(1, "g0", "planner")]
    assert [(e.step, e.mode) for e in s.plan_events] == [(1, PlanMode.INIT)]
    # execute: one dense search for the pending query at default_k
    (req,) = fake.retriever.requests
    assert (req.query, req.mode, req.k) == (GOAL, RetrievalMode.DENSE, 20)
    (record,) = s.retrieval_history
    assert (record.step, record.action, record.cache_hit) == (1, Action.DENSE, False)
    assert s.seen_candidate_ids == record.candidate_ids
    # score_and_update: reranked against the state, evidence applied, last_result set
    assert fake.reranker.calls[0][2] is True  # state-aware call (state passed)
    assert len(s.evidence) == 2 and s.last_update is not None
    assert s.last_result is not None and s.last_result.depth == 20
    # persist: counted, versioned, saved with the queued cache entry, traced
    assert (s.version, s.budget.steps_used, s.budget.retrieval_calls_used) == (1, 1, 1)
    assert (s.budget.llm_calls_used, s.budget.reranker_calls_used) == (1, 1)
    (entry,) = read_cache_entries(connect(fake.db), "t-1")
    assert (entry.entry_id, entry.mode, entry.created_at_step) == ("step-1", RetrievalMode.DENSE, 1)
    assert entry.accepted_ids == [e.evidence_id for e in s.evidence]
    assert (trace.step, trace.action, trace.rule, trace.current_gap_id) == (
        1,
        Action.DENSE,
        "scripted",
        "g0",
    )
    assert trace.accepted_ids == entry.accepted_ids and trace.llm_calls == 1
    assert set(trace.latency_ms) >= {
        "plan",
        "select",
        "execute",
        "rerank",
        "evidence",
        "monitor",
        "total",
    }
    assert fake.parts.traces.read_steps("t-1") == [trace]


def test_max_steps_runs_exactly_n_real_iterations(engine_with_fakes: EngineFactory) -> None:
    controller = FakeController([Action.DENSE, Action.SPARSE_HYBRID, Action.DENSE])
    fake = engine_with_fakes(controller=controller)
    start(fake, budget=budget(max_steps=3))
    bundle = fake.engine.run("t-1")
    s = fake.engine.state("t-1")
    assert (s.status, s.stop_reason, s.budget.steps_used) == (
        TaskStatus.STOPPED,
        StopReason.MAX_STEPS,
        3,
    )
    steps = fake.parts.traces.read_steps("t-1")
    assert [t.action for t in steps] == [Action.DENSE, Action.SPARSE_HYBRID, Action.DENSE]
    assert Action.STOP not in [t.action for t in steps]  # no wasted STOP-only step
    assert bundle.stop_reason == StopReason.MAX_STEPS and bundle.trace_summary.steps == 3
    stored = fake.parts.traces.read_bundle("t-1")
    assert stored is not None and stored.model_dump() == bundle.model_dump()  # vectors null
    with pytest.raises(TaskNotActive):
        fake.engine.step("t-1")


def test_resolving_every_gap_stops_with_all_gaps_resolved(engine_with_fakes: EngineFactory) -> None:
    fake = engine_with_fakes(planner=two_gaps(), evidence=FakeEvidenceManager(resolve=True))
    start(fake)
    bundle = fake.engine.run("t-1")
    s = fake.engine.state("t-1")
    assert bundle.stop_reason == StopReason.ALL_GAPS_RESOLVED
    assert [(g.gap_id, g.status, g.closed_at_step) for g in s.gaps] == [
        ("g0", GapStatus.RESOLVED, 1),
        ("g1", GapStatus.RESOLVED, 2),
    ]
    resolver = {e.evidence_id: e for e in s.evidence}[s.gaps[0].resolved_by or ""]
    assert resolver.pinned and resolver.supports_gap_ids == ["g0"]
    # g1 is lexical: SPARSE_HYBRID would search sparse; DENSE searches its own query
    assert [r.query for r in fake.retriever.requests] == ["Linden Bay tram", "council harbour"]


def test_sparse_hybrid_uses_sparse_for_lexical_needs_else_hybrid(
    engine_with_fakes: EngineFactory,
) -> None:
    controller = FakeController([Action.SPARSE_HYBRID])
    fake = engine_with_fakes(
        planner=two_gaps(), controller=controller, evidence=FakeEvidenceManager(resolve=True)
    )
    start(fake)
    fake.engine.run("t-1")
    assert [r.mode for r in fake.retriever.requests] == [RetrievalMode.HYBRID, RetrievalMode.SPARSE]


def test_reuse_serves_the_cached_candidates_without_a_retrieval_call(
    engine_with_fakes: EngineFactory,
) -> None:
    controller = FakeController([Action.DENSE, Action.REUSE])
    fake = engine_with_fakes(controller=controller, evidence=FakeEvidenceManager(accept=1))
    start(fake)
    fake.engine.step("t-1")
    trace = fake.engine.step("t-1")
    s = fake.engine.state("t-1")
    hit = controller.hits[1]
    assert isinstance(hit, CacheHit) and hit.match == "exact"
    assert len(fake.retriever.requests) == 1 and s.budget.retrieval_calls_used == 1
    reuse = s.retrieval_history[1]
    assert (reuse.action, reuse.cache_hit, reuse.mode, reuse.query) == (
        Action.REUSE,
        True,
        RetrievalMode.DENSE,
        GOAL,
    )
    assert reuse.k == len(hit.entry.candidates) and reuse.new_candidate_ids == []
    assert (trace.cache_hit, trace.action) == (True, Action.REUSE)
    assert [e.entry_id for e in read_cache_entries(connect(fake.db), "t-1")] == ["step-1"]
    assert s.evidence[1].source_action == Action.REUSE  # second item accepted from the hit


def test_rerank_expand_searches_deeper_and_reranks_only_unseen(
    cfg: EngineConfig, engine_with_fakes: EngineFactory, mini_corpus: list[CorpusChunk]
) -> None:
    cfg.retrieval.default_k = 3  # the goal matches 6 mini-corpus chunks
    cfg.retrieval.expand_step_k = 3
    controller = FakeController([Action.DENSE, Action.RERANK_EXPAND])
    fake = engine_with_fakes(controller=controller)
    start(fake)
    fake.engine.step("t-1")
    trace = fake.engine.step("t-1")
    first, second = fake.retriever.requests
    assert (second.query, second.mode, second.k) == (first.query, first.mode, 6)
    s = fake.engine.state("t-1")
    expand = s.retrieval_history[1]
    assert len(expand.candidate_ids) == 6 and len(expand.new_candidate_ids) == 3
    assert fake.reranker.calls[1][1] == expand.new_candidate_ids  # only unseen passed on
    assert trace.candidate_ids == expand.new_candidate_ids  # post-filter (OBSERVABILITY §2)
    entries = read_cache_entries(connect(fake.db), "t-1")
    assert [len(e.candidates) for e in entries] == [6]  # deeper result replaced step 1's


def test_transform_rewrite_changes_only_the_current_gap_query(
    engine_with_fakes: EngineFactory,
) -> None:
    controller = FakeController([Action.DENSE, Action.TRANSFORM, Action.DENSE])
    fake = engine_with_fakes(controller=controller)
    start(fake)
    fake.engine.step("t-1")
    trace = fake.engine.step("t-1")
    s = fake.engine.state("t-1")
    assert s.gaps[0].query == f"{GOAL} rewritten" and s.gaps[0].description == GOAL
    assert s.pending_query == f"{GOAL} rewritten"
    assert [q.query for q in s.query_history] == [GOAL, f"{GOAL} rewritten"]
    assert s.last_update is None and len(s.retrieval_history) == 1  # no candidates
    assert (trace.action, trace.query, trace.llm_calls, trace.evidence_gain) == (
        Action.TRANSFORM,
        None,
        1,
        0.0,
    )
    assert [e.mode for e in trace.plan_events] == [PlanMode.REWRITE]
    fake.engine.step("t-1")
    assert fake.retriever.requests[-1].query == f"{GOAL} rewritten"


def test_transform_split_supersedes_the_gap_and_moves_to_the_first_child(
    engine_with_fakes: EngineFactory,
) -> None:
    split = PlanResult(
        mode=PlanMode.SPLIT,
        subgoals=[
            SubgoalPlan(description="part a", query="tram route", need_type=NeedType.SEMANTIC),
            SubgoalPlan(description="part b", query="council vote", need_type=NeedType.LEXICAL),
        ],
    )
    controller = FakeController([Action.DENSE, Action.TRANSFORM], plan_mode=PlanMode.SPLIT)
    fake = engine_with_fakes(controller=controller, planner=FakePlanner(transforms=[split]))
    start(fake)
    fake.engine.step("t-1")
    fake.engine.step("t-1")
    s = fake.engine.state("t-1")
    assert [(g.gap_id, g.status, g.parent_gap_id) for g in s.gaps] == [
        ("g0", GapStatus.SUPERSEDED, None),
        ("g1", GapStatus.OPEN, "g0"),
        ("g2", GapStatus.OPEN, "g0"),
    ]
    assert s.current_gap_id == "g1" and s.pending_query == "tram route"


def test_stop_decision_ends_the_task_with_its_reason(engine_with_fakes: EngineFactory) -> None:
    fake = engine_with_fakes(controller=FakeController([Action.DENSE, Action.STOP]))
    start(fake)
    bundle = fake.engine.run("t-1")
    assert bundle.stop_reason == StopReason.LOW_MARGINAL_UTILITY
    assert [t.action for t in fake.parts.traces.read_steps("t-1")] == [Action.DENSE, Action.STOP]


def test_backend_failure_leaves_the_last_saved_state_and_no_cache_entry(
    engine_with_fakes: EngineFactory, mini_corpus: list[CorpusChunk]
) -> None:
    retriever = FakeRetriever(mini_corpus, fail_on_call=2)
    fake = engine_with_fakes(
        retriever=retriever, controller=FakeController([Action.DENSE, Action.SPARSE_HYBRID])
    )
    start(fake)
    fake.engine.step("t-1")
    after_one = fake.engine.state("t-1")
    with pytest.raises(BackendError):
        fake.engine.step("t-1")
    assert fake.engine.state("t-1") == after_one  # still ACTIVE, still version 1
    assert [e.entry_id for e in read_cache_entries(connect(fake.db), "t-1")] == ["step-1"]
    assert [t.step for t in fake.parts.traces.read_steps("t-1")] == [1]
    fake.engine.step("t-1")  # retryable: the next step runs as step 2
    assert fake.engine.state("t-1").budget.steps_used == 2


def test_non_retryable_error_marks_the_task_failed(engine_with_fakes: EngineFactory) -> None:
    evidence = FakeEvidenceManager()
    fake = engine_with_fakes(evidence=evidence)
    start(fake)
    fake.engine.step("t-1")
    evidence.fail = RuntimeError("bug in a component")
    with pytest.raises(RuntimeError, match="bug"):
        fake.engine.step("t-1")
    s = fake.engine.state("t-1")
    assert (s.status, s.stop_reason, s.version, s.budget.steps_used) == (
        TaskStatus.FAILED,
        StopReason.ERROR,
        2,
        1,
    )
    assert [t.step for t in fake.parts.traces.read_steps("t-1")] == [1]  # no TraceStep for it
    assert fake.engine.result("t-1").stop_reason == StopReason.ERROR
    with pytest.raises(TaskNotActive):
        fake.engine.run("t-1")


def test_invariant_violation_is_non_retryable(engine_with_fakes: EngineFactory) -> None:
    class Duplicating(FakeEvidenceManager):  # accepts an item that is already evidence
        def update(self, state, scored, step):  # type: ignore[no-untyped-def]
            u = super().update(state, scored, step)
            return u.model_copy(update={"accepted": [*state.evidence[:1], *u.accepted]})

    fake = engine_with_fakes(
        evidence=Duplicating(), controller=FakeController([Action.DENSE, Action.SPARSE_HYBRID])
    )
    start(fake)
    fake.engine.step("t-1")
    with pytest.raises(Exception, match=r"\[inv 2\]"):
        fake.engine.step("t-1")
    assert fake.engine.state("t-1").status == TaskStatus.FAILED


def test_concurrent_step_on_the_same_task_is_rejected(engine_with_fakes: EngineFactory) -> None:
    entered, release = threading.Event(), threading.Event()

    class Blocking(FakeController):
        def select_action(
            self, state: RetrievalState, cache_hit: CacheHit | None
        ) -> ActionDecision:
            entered.set()
            release.wait(5)
            return super().select_action(state, cache_hit)

    fake = engine_with_fakes(controller=Blocking())
    start(fake)
    worker = threading.Thread(target=fake.engine.step, args=("t-1",))
    worker.start()
    assert entered.wait(5)
    with pytest.raises(TaskConflict, match="in progress"):
        fake.engine.step("t-1")
    release.set()
    worker.join()
    assert fake.engine.state("t-1").budget.steps_used == 1


def test_two_engines_on_one_db_conflict_on_save(engine_with_fakes: EngineFactory) -> None:
    fake = engine_with_fakes()
    start(fake)
    stale = fake.engine.state("t-1")
    fake.engine.step("t-1")
    other = Engine(
        fake.engine.cfg, retriever=fake.retriever, reranker=fake.reranker, controlled=fake.parts
    )
    with pytest.raises(TaskConflict, match="stored version 1, expected 0"):
        other._iterate(stale)  # bypasses the lock with a state loaded before step 1
    assert fake.engine.state("t-1").version == 1


def test_run_timeout_keeps_the_last_completed_step(
    cfg: EngineConfig, engine_with_fakes: EngineFactory
) -> None:
    cfg.api.run_timeout_s = 1
    ticks = count(0.0, 0.4)  # each clock read advances 0.4 s
    fake = engine_with_fakes(clock=lambda: next(ticks))
    start(fake)
    with pytest.raises(RunTimeout):
        fake.engine.run("t-1")
    s = fake.engine.state("t-1")
    assert s.status == TaskStatus.ACTIVE and s.budget.steps_used >= 1
    assert len(fake.parts.traces.read_steps("t-1")) == s.budget.steps_used


def test_create_task_defaults_and_duplicates(
    cfg: EngineConfig, engine_with_fakes: EngineFactory
) -> None:
    fake = engine_with_fakes()
    s = start(fake)
    assert s.budget.max_steps == cfg.budget_defaults.max_steps
    with pytest.raises(TaskExists):
        start(fake)
    with pytest.raises(TaskNotActive, match="still active"):
        fake.engine.result("t-1")


def test_controlled_path_needs_its_components(
    cfg: EngineConfig, mini_corpus: list[CorpusChunk]
) -> None:
    engine = Engine(cfg, retriever=FakeRetriever(mini_corpus))
    with pytest.raises(ConfigError, match="controlled path needs"):
        engine.create_task(GOAL)

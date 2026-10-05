"""baselines/dense_oneshot.py with fakes (BASELINES_SPEC §2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from baselines import dense_oneshot
from baselines.base import BaselineContext
from engine.config import EngineConfig
from engine.core import Engine
from engine.monitor.trace import JsonlTraceWriter
from engine.state.models import Action, RetrievalMode, RunContext, StopReason
from eval.datasets.base import Task
from tests.builders import NOW
from tests.fakes import CorpusChunk, FakeEmbedder, FakeReranker, FakeRetriever

CTX = RunContext(run_id="r1", system=dense_oneshot.SYSTEM)


def task(question: str = "Linden Bay tram expansion council harbour") -> Task:
    return Task(
        task_id="multihop:0001",
        dataset="multihop",
        question=question,
        answer="a",
        question_type="temporal_query",
        gold_groups=[["c1"]],
        split="dev",
    )


def make_ctx(
    cfg: EngineConfig,
    tmp_path: Path,
    mini_corpus: list[CorpusChunk],
    reranker: FakeReranker | None = None,
) -> tuple[BaselineContext, FakeRetriever]:
    retriever = FakeRetriever(mini_corpus, embedder=FakeEmbedder())
    engine = Engine(cfg, retriever=retriever, reranker=reranker)
    ctx = BaselineContext(
        cfg=cfg,
        engine=engine,
        traces=JsonlTraceWriter(tmp_path, CTX),
        retrieval_budget=1,
        now=lambda: NOW,
    )
    return ctx, retriever


def test_one_dense_call_and_a_bundle_of_top_n(
    cfg: EngineConfig, tmp_path: Path, mini_corpus: list[CorpusChunk]
) -> None:
    cfg.rerank.top_n_output = 3
    ctx, retriever = make_ctx(cfg, tmp_path, mini_corpus)
    bundle = dense_oneshot.run_task(task(), ctx)
    assert [(r.mode, r.k) for r in retriever.requests] == [
        (RetrievalMode.DENSE, cfg.retrieval.default_k)
    ]
    assert len(bundle.evidence) == 3
    scores = [e.final_score for e in bundle.evidence]
    assert scores == sorted(scores, reverse=True)
    assert bundle.stop_reason == StopReason.ONE_SHOT_COMPLETE
    assert bundle.system == dense_oneshot.SYSTEM
    assert all(e.relevance == 0.0 and e.source_action == Action.DENSE for e in bundle.evidence)
    assert all(len(e.summary) <= cfg.planner.evidence_summary_chars for e in bundle.evidence)


def test_writes_exactly_one_trace_step_and_the_bundle(
    cfg: EngineConfig, tmp_path: Path, mini_corpus: list[CorpusChunk]
) -> None:
    ctx, _ = make_ctx(cfg, tmp_path, mini_corpus)
    bundle = dense_oneshot.run_task(task(), ctx)
    (step,) = ctx.traces.read_steps("multihop:0001")
    assert (step.step, step.action, step.rule, step.mode) == (
        1,
        Action.DENSE,
        "one_shot",
        RetrievalMode.DENSE,
    )
    assert step.utility_scores == {}
    assert step.accepted_ids == [e.evidence_id for e in bundle.evidence]
    assert (step.llm_calls, step.budget_after.retrieval_calls_used) == (0, 1)
    assert step.stop_reason == StopReason.ONE_SHOT_COMPLETE
    assert ctx.traces.read_bundle("multihop:0001") is not None
    assert bundle.trace_summary.retrieval_calls == 1
    assert bundle.trace_summary.actions == [Action.DENSE]


def test_uses_the_reranker_when_one_exists(
    cfg: EngineConfig, tmp_path: Path, mini_corpus: list[CorpusChunk]
) -> None:
    kelp = next(c.chunk_id for c in mini_corpus if "kelp tank" in c.text)
    reranker = FakeReranker({kelp: 0.9}, default=0.1, top_n=cfg.rerank.top_n_output)
    ctx, _ = make_ctx(cfg, tmp_path, mini_corpus, reranker)
    bundle = dense_oneshot.run_task(task("Linden Bay aquarium kelp tank"), ctx)
    assert bundle.evidence[0].chunk_id == kelp
    assert bundle.evidence[0].relevance == 0.9
    assert bundle.trace_summary.reranker_calls == 1


@pytest.mark.parametrize("budget", [1, 3])
def test_budget_records_the_sub_run_cap(
    cfg: EngineConfig, tmp_path: Path, mini_corpus: list[CorpusChunk], budget: int
) -> None:
    ctx, _ = make_ctx(cfg, tmp_path, mini_corpus)
    ctx = BaselineContext(
        cfg=ctx.cfg, engine=ctx.engine, traces=ctx.traces, retrieval_budget=budget, now=ctx.now
    )
    dense_oneshot.run_task(task(), ctx)
    (step,) = ctx.traces.read_steps("multihop:0001")
    assert step.budget_after.max_retrieval_calls == budget

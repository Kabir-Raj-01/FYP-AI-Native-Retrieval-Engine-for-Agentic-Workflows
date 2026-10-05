"""Shared pytest fixtures (docs/TESTING.md §2)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import structlog

from engine.cache.task_cache import TaskCache
from engine.config import EngineConfig, load_config
from engine.core import Engine
from engine.graph.nodes import Components
from engine.monitor.trace import JsonlTraceWriter
from engine.state.models import Budget, EvidenceGap, EvidenceItem, RetrievalState, RunContext
from engine.state.store import SqliteTaskStore
from tests.builders import NOW, make_state
from tests.fakes import (
    FakeController,
    FakeEmbedder,
    FakeEvidenceManager,
    FakeMonitor,
    FakePlanner,
    FakeReranker,
    FakeRetriever,
)
from tests.fakes.retriever import CorpusChunk
from tests.fixture_data import MiniTask, load_mini_corpus, load_mini_tasks

# Test overrides on top of base.yaml: every LLM role uses the fake provider with a named
# model, so code that calls require_llm_model() works without ADR-004.
TEST_CONFIG_OVERRIDES = [
    f"llm.{role}.{key}={value}"
    for role in ("planner", "verifier", "generator")
    for key, value in (("provider", "fake"), ("model", "fake-model"))
]

StateFactory = Callable[..., RetrievalState]


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """structlog's configuration is process-global; never let one test's leak into another."""
    yield
    structlog.reset_defaults()


@pytest.fixture(scope="session")
def _session_cfg() -> EngineConfig:
    # environ={} keeps the developer's .env and shell variables out of unit tests.
    return load_config(environ={}, overrides=TEST_CONFIG_OVERRIDES)


@pytest.fixture
def cfg(_session_cfg: EngineConfig) -> EngineConfig:
    """base.yaml + TEST_CONFIG_OVERRIDES; a fresh deep copy per test."""
    return _session_cfg.model_copy(deep=True)


@pytest.fixture
def state_factory(cfg: EngineConfig) -> StateFactory:
    """Build a RetrievalState with the given gaps/evidence/budget (budget defaults from
    ``cfg.budget_defaults``); any other RetrievalState field can be passed by name."""

    def make(
        *,
        gaps: Sequence[EvidenceGap] = (),
        evidence: Sequence[EvidenceItem] = (),
        budget: Budget | None = None,
        **overrides: Any,
    ) -> RetrievalState:
        if budget is None:
            d = cfg.budget_defaults
            budget = Budget(
                max_steps=d.max_steps,
                max_retrieval_calls=d.max_retrieval_calls,
                max_llm_calls=d.max_llm_calls,
                max_tokens=d.max_tokens,
            )
        return make_state(gaps=list(gaps), evidence=list(evidence), budget=budget, **overrides)

    return make


@pytest.fixture
def tmp_state_db(tmp_path: Path) -> Path:
    """Path for a throwaway SQLite state DB (not created until a store or cache connects)."""
    return tmp_path / "state.sqlite3"


@dataclass
class FakeEngine:
    engine: Engine
    parts: Components
    retriever: FakeRetriever
    reranker: FakeReranker
    db: Path


EngineFactory = Callable[..., FakeEngine]


@pytest.fixture
def engine_with_fakes(
    cfg: EngineConfig, tmp_path: Path, mini_corpus: list[CorpusChunk]
) -> EngineFactory:
    """Build an Engine whose controlled path runs on fakes over the mini corpus, a real
    SQLite TaskStore/cache in ``tmp_path`` and a JSONL trace writer. Keyword arguments
    replace components (``planner=``, ``controller=``, ``evidence=``, ``retriever=``...);
    ``clock=``/``now=`` are passed to the Engine."""

    def make(
        clock: Callable[[], float] | None = None, now: Callable[[], Any] | None = None, **parts: Any
    ) -> FakeEngine:
        db = tmp_path / "state.sqlite3"
        embedder = FakeEmbedder()
        retriever = parts.pop("retriever", None) or FakeRetriever(mini_corpus, embedder=embedder)
        reranker = parts.pop("reranker", None) or FakeReranker(default=0.5)
        built = Components(
            store=SqliteTaskStore(cfg, db, now=lambda: NOW),
            cache=TaskCache(cfg, embedder, db),
            planner=FakePlanner(),
            controller=FakeController(),
            retriever=retriever,
            reranker=reranker,
            evidence=FakeEvidenceManager(),
            monitor=FakeMonitor(),
            embedder=embedder,
            traces=JsonlTraceWriter(
                tmp_path / "run", RunContext(run_id="r1", system="controller_on")
            ),
        )
        built = replace(built, **parts)
        kwargs: dict[str, Any] = {"now": now or (lambda: NOW)}
        if clock is not None:
            kwargs["clock"] = clock
        engine = Engine(cfg, retriever=retriever, reranker=reranker, controlled=built, **kwargs)
        return FakeEngine(engine, built, retriever, reranker, db)

    return make


@pytest.fixture(scope="session")
def mini_corpus() -> list[CorpusChunk]:
    return load_mini_corpus()


@pytest.fixture(scope="session")
def mini_tasks() -> list[MiniTask]:
    return load_mini_tasks()

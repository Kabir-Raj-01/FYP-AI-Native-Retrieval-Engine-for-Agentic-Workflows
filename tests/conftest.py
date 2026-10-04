"""Shared pytest fixtures (docs/TESTING.md §2)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from engine.config import EngineConfig, load_config
from engine.state.models import Budget, EvidenceGap, EvidenceItem, RetrievalState
from tests.builders import make_state
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
    """Path for a throwaway SQLite state DB (not created; the TaskStore is Weeks 8-9)."""
    return tmp_path / "state.sqlite3"


@pytest.fixture
def engine_with_fakes() -> None:
    """Placeholder: the Engine facade (engine/core.py) is not built yet."""
    pytest.skip("engine_with_fakes needs the Engine facade (docs/TASKS.md Weeks 8-9)")


@pytest.fixture(scope="session")
def mini_corpus() -> list[CorpusChunk]:
    return load_mini_corpus()


@pytest.fixture(scope="session")
def mini_tasks() -> list[MiniTask]:
    return load_mini_tasks()

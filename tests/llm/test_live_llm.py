"""Live LLM checks (TESTING §1 "LLM" layer; ADR-004): the real planner and generator on the
configured provider (the local Ollama by default). Run by hand before milestones, with the
provider up (SETUP §2):

    pytest -m llm

The response cache is off, so the provider is really called. A role whose endpoint is
unreachable (or whose API key is missing) is skipped, never failed; CI never runs this layer.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx
import pytest

from engine import factory
from engine.config import EngineConfig, LLMRole, load_config
from engine.llm.interfaces import Message
from engine.state.models import Budget, PlanMode, PlanStatus, RetrievalState

pytestmark = pytest.mark.llm

# dev task multihop:0082 (the FYP-I demo question), never a test-split question
GOAL = (
    "Does the Polygon article suggest that Martin Scorsese has less autonomy in his "
    "filmmaking choices compared to earlier in his career, while The Independent - Life and "
    "Style article indicates that he has previously engaged with newer media platforms like "
    "TikTok through his daughter's videos?"
)


@pytest.fixture(scope="module")
def live_cfg() -> EngineConfig:
    return load_config(overrides=["llm.cache_enabled=false"])  # real config + .env


def require_provider(cfg: EngineConfig, role: LLMRole) -> None:
    r = getattr(cfg.llm, role)
    cfg.require_llm_model(role)
    if r.provider == "openai_compatible":
        try:
            httpx.get(f"{r.base_url.rstrip('/')}/models", timeout=3).raise_for_status()
        except httpx.HTTPError as exc:
            pytest.skip(f"{role}: endpoint {r.base_url} unreachable ({type(exc).__name__})")
    elif not os.environ.get(r.api_key_env):
        pytest.skip(f"{role}: {r.api_key_env} is not set")


def test_planner_init_on_a_dev_question_is_accepted_and_grounded(live_cfg: EngineConfig) -> None:
    require_provider(live_cfg, "planner")
    now = datetime.now(UTC)
    d = live_cfg.budget_defaults
    state = RetrievalState(
        task_id="llm-test",
        goal=GOAL,
        budget=Budget(
            max_steps=d.max_steps,
            max_retrieval_calls=d.max_retrieval_calls,
            max_llm_calls=d.max_llm_calls,
            max_tokens=d.max_tokens,
        ),
        created_at=now,
        updated_at=now,
    )
    outcome = factory.build_planner(live_cfg, factory.build_llm_client(live_cfg)).plan(
        state, PlanMode.INIT
    )
    reasons = [(e.status.value, e.reason) for e in outcome.events]
    assert outcome.events[-1].status == PlanStatus.ACCEPTED, reasons  # not the fallback
    subgoals = outcome.result.subgoals
    assert 1 <= len(subgoals) <= live_cfg.planner.max_subgoals
    assert all(s.query.strip() and s.description.strip() for s in subgoals)


def test_generator_answers(live_cfg: EngineConfig) -> None:
    require_provider(live_cfg, "generator")
    llm = factory.build_llm_client(live_cfg)
    reply = llm.text("generator", [Message(role="user", content="Reply with exactly: OK")])
    assert reply.value.strip()

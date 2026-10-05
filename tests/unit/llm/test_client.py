"""engine/llm/client.py with scripted providers (LLM_CLIENT_SPEC §5-§8, §10)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from structlog.testing import capture_logs

from engine.config import EngineConfig, LLMPrice
from engine.errors import ConfigError, LLMBudgetExceeded, LLMError, LLMOutputError
from engine.llm.client import ProviderLLMClient, strip_to_json
from engine.llm.interfaces import Message
from engine.llm.providers.base import Completion, TransportError

SECRET = "sk-test-secret-value"  # pragma: allowlist secret


class Plan(BaseModel):
    query: str


class Scripted:
    """Returns (or raises) the scripted items in order; records each request."""

    def __init__(self, items: Sequence[Completion | Exception]) -> None:
        self.items = list(items)
        self.requests: list[dict[str, Any]] = []

    def complete(
        self, model: str, system: str, messages: list[Message], max_tokens: int, temperature: float
    ) -> Completion:
        self.requests.append({"model": model, "system": system, "messages": messages})
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok(
    text: str = '{"query": "kx-41 drone"}', tokens: tuple[int | None, int | None] = (100, 20)
) -> Completion:
    return Completion(text, *tokens)


@pytest.fixture
def llm_cfg(cfg: EngineConfig, tmp_path: Path) -> EngineConfig:
    cfg.paths.llm_cache_db = tmp_path / "llm.sqlite3"
    return cfg


def client(cfg: EngineConfig, provider: Scripted, **kw: Any) -> ProviderLLMClient:
    return ProviderLLMClient(
        cfg, {"planner": provider, "generator": provider}, sleep=lambda s: None, **kw
    )


MESSAGES = [Message(role="system", content="You plan."), Message(role="user", content="goal")]


def test_structured_parses_and_puts_the_schema_in_the_system_text(llm_cfg: EngineConfig) -> None:
    provider = Scripted([ok()])
    result = client(llm_cfg, provider).structured("planner", MESSAGES, Plan)
    assert result.value == Plan(query="kx-41 drone")
    assert (result.input_tokens, result.output_tokens, result.cached) == (100, 20, False)
    (req,) = provider.requests
    assert req["system"].startswith("You plan.") and '"query"' in req["system"]
    assert [m.role for m in req["messages"]] == ["user"]  # system text never in user turns
    assert req["model"] == llm_cfg.llm.planner.model


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"query": "q"}\n```',
        'Here is the plan:\n{"query": "q"}\nHope this helps.',
        '```\n{"query": "q"}\n```',
    ],
)
def test_fences_and_surrounding_prose_are_stripped(text: str) -> None:
    assert strip_to_json(text) == '{"query": "q"}'


@pytest.mark.parametrize("text", ["not json at all", '{"query": 5, "extra": 1}', '{"other": "x"}'])
def test_invalid_output_raises_llm_output_error(llm_cfg: EngineConfig, text: str) -> None:
    with pytest.raises(LLMOutputError):
        client(llm_cfg, Scripted([ok(text)])).structured("planner", MESSAGES, Plan)


def test_transport_errors_are_retried_then_succeed(llm_cfg: EngineConfig) -> None:
    provider = Scripted([TransportError("timeout"), TransportError("503"), ok()])
    with capture_logs() as logs:
        result = client(llm_cfg, provider).structured("planner", MESSAGES, Plan)
    assert result.value.query == "kx-41 drone" and len(provider.requests) == 3
    (event,) = [e for e in logs if e["event"] == "llm.call"]
    assert event["attempt"] == 3


def test_retry_exhaustion_raises_llm_error(llm_cfg: EngineConfig) -> None:
    n = llm_cfg.llm.planner.max_transport_retries + 1
    provider = Scripted([TransportError("timeout")] * n)
    with pytest.raises(LLMError, match=f"after {n} attempts"):
        client(llm_cfg, provider).structured("planner", MESSAGES, Plan)
    assert len(provider.requests) == n


def test_non_retryable_provider_errors_are_not_retried(llm_cfg: EngineConfig) -> None:
    provider = Scripted([LLMError(f"HTTP 401 bad key {SECRET}"), ok()])
    with pytest.raises(LLMError) as exc:
        client(llm_cfg, provider, environ={"LLM_API_KEY": SECRET}).structured(
            "planner", MESSAGES, Plan
        )
    assert len(provider.requests) == 1
    assert SECRET not in str(exc.value) and "***" in str(exc.value)  # key redacted


def test_cache_hit_returns_the_same_value_with_zero_cost(llm_cfg: EngineConfig) -> None:
    llm_cfg.llm.prices = {"fake-model": LLMPrice(input_per_mtok=3.0, output_per_mtok=15.0)}
    provider = Scripted([ok()])
    c = client(llm_cfg, provider)
    first = c.structured("planner", MESSAGES, Plan)
    with capture_logs() as logs:
        second = c.structured("planner", MESSAGES, Plan)
    assert second.value == first.value and len(provider.requests) == 1
    assert (second.cached, second.cost_usd) == (True, 0.0)
    assert (second.input_tokens, second.output_tokens) == (100, 20)  # logical usage recorded
    expected = (100 * 3.0 + 20 * 15.0) / 1_000_000
    assert first.cost_usd == pytest.approx(expected)
    (event,) = [e for e in logs if e["event"] == "llm.call"]
    assert event["cost_usd_equivalent"] == pytest.approx(expected, abs=1e-6)
    # a new client on the same DB (the next run) also hits; real calls are not counted
    again = client(llm_cfg, Scripted([]))
    assert again.structured("planner", MESSAGES, Plan).cached
    assert again.run_totals == {"llm_calls": 0, "cost_usd": 0.0}


def test_namespace_or_disabled_cache_bypasses_earlier_responses(llm_cfg: EngineConfig) -> None:
    client(llm_cfg, Scripted([ok()])).structured("planner", MESSAGES, Plan)
    llm_cfg.llm.cache_namespace = "repeat-2"
    fresh = Scripted([ok()])
    assert not client(llm_cfg, fresh).structured("planner", MESSAGES, Plan).cached
    llm_cfg.llm.cache_enabled = False
    twice = Scripted([ok(), ok()])
    c = client(llm_cfg, twice)
    c.structured("planner", MESSAGES, Plan)
    c.structured("planner", MESSAGES, Plan)
    assert len(twice.requests) == 2


def test_per_run_call_and_cost_caps(llm_cfg: EngineConfig) -> None:
    llm_cfg.llm.cache_enabled = False
    llm_cfg.llm.max_calls_per_run = 1
    c = client(llm_cfg, Scripted([ok(), ok()]))
    c.text("generator", [Message(role="user", content="answer")])
    with pytest.raises(LLMBudgetExceeded):
        c.text("generator", [Message(role="user", content="answer")])
    llm_cfg.llm.max_calls_per_run = 100
    llm_cfg.llm.max_cost_usd_per_run = 0.001
    llm_cfg.llm.prices = {"fake-model": LLMPrice(input_per_mtok=1000.0, output_per_mtok=0.0)}
    c = client(llm_cfg, Scripted([ok(), ok()]))
    c.text("generator", [Message(role="user", content="answer")])  # costs 0.1 > cap
    with pytest.raises(LLMBudgetExceeded, match="per-run LLM cap"):
        c.text("generator", [Message(role="user", content="answer")])


def test_missing_usage_is_estimated_and_flagged(llm_cfg: EngineConfig) -> None:
    with capture_logs() as logs:
        r = client(llm_cfg, Scripted([ok("x" * 40, tokens=(None, None))])).text(
            "generator", MESSAGES
        )
    assert r.output_tokens == 10 and r.input_tokens > 0
    (event,) = [e for e in logs if e["event"] == "llm.call"]
    assert event["estimated_tokens"] is True


def test_role_and_provider_configuration_errors(llm_cfg: EngineConfig) -> None:
    llm_cfg.llm.planner.model = None
    with pytest.raises(ConfigError, match=r"llm.planner.model is null"):
        client(llm_cfg, Scripted([])).structured("planner", MESSAGES, Plan)
    llm_cfg.llm.planner.model = "m"
    with pytest.raises(ConfigError, match="unknown LLM role"):
        client(llm_cfg, Scripted([])).text("critic", MESSAGES)
    plain = ProviderLLMClient(llm_cfg, environ={})
    with pytest.raises(ConfigError, match="for tests"):  # TEST_CONFIG_OVERRIDES: provider fake
        plain.text("planner", MESSAGES)
    llm_cfg.llm.planner.provider = "anthropic"
    with pytest.raises(ConfigError, match="LLM_API_KEY is not set"):
        ProviderLLMClient(llm_cfg, environ={}).text("planner", MESSAGES)

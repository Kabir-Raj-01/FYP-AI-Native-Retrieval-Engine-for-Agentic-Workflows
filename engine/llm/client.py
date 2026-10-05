"""The LLM client: every LLM call in the project (LLM_CLIENT_SPEC; CLAUDE.md rule 5).

- Structured output (§5): the JSON schema of the requested model is appended to the system
  text; the reply is fence/prose-stripped, ``json.loads``-ed and validated; any failure is
  ``LLMOutputError`` (no other repair).
- Transport (§6): one provider request per attempt; TransportError is retried up to
  ``llm.<role>.max_transport_retries`` times with exponential backoff + jitter (tenacity),
  then ``LLMError``.
- Response cache (§7): key = sha256 of the canonical request (provider, model, role,
  temperature, max tokens, messages, schema, ``llm.cache_namespace``). A hit returns the
  recorded usage with ``cost_usd = 0`` (the estimate goes to ``cost_usd_equivalent`` in the
  event), so callers' budgets count logical calls, cached or not.
- Per-run caps (§8): ``llm.max_calls_per_run`` and ``llm.max_cost_usd_per_run`` count real
  provider requests (cache hits spend no credit); reaching either -> ``LLMBudgetExceeded``.
- Tokens come from the provider's usage; when a server reports none they are estimated as
  ``ceil(chars / 4)`` and the event says ``estimated_tokens=true``.
API keys are read from the environment variable named by ``llm.<role>.api_key_env`` when a
role is first used, and are never logged (redacted from error messages).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from engine.config import EngineConfig, LLMRoleConfig
from engine.errors import ConfigError, LLMBudgetExceeded, LLMError, LLMOutputError
from engine.llm.cache import LLMResponseCache
from engine.llm.interfaces import LLMClient, LLMResult, Message, SchemaT
from engine.llm.providers.base import Completion, Provider, TransportError
from engine.monitor.logging import get_logger

__all__ = ["ProviderLLMClient", "strip_to_json"]

_ROLES = ("planner", "verifier", "generator")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_SCHEMA_INSTRUCTION = (
    "Return only one JSON object that matches this JSON schema - no prose, no code fences."
)


def strip_to_json(text: str) -> str:
    """Strip code fences and leading/trailing prose around one JSON object (§5.2)."""
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    return body[start : end + 1] if 0 <= start < end else body.strip()


def _estimate(text: str) -> int:
    return math.ceil(len(text) / 4)


class ProviderLLMClient(LLMClient):
    def __init__(
        self,
        cfg: EngineConfig,
        providers: Mapping[str, Provider] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """``providers`` (role -> Provider) replaces the configured ones (tests)."""
        self._cfg = cfg
        self._providers: dict[str, Provider] = dict(providers or {})
        self._environ = environ if environ is not None else os.environ
        self._clock = clock
        self._sleep = sleep
        self._cache = LLMResponseCache(cfg.paths.llm_cache_db) if cfg.llm.cache_enabled else None
        self._lock = threading.Lock()
        self._calls = 0  # real provider requests this run
        self._cost = 0.0
        self._log = get_logger("engine.llm")

    # ------------------------------------------------------------------------ interface
    def structured(
        self, role: str, messages: list[Message], schema: type[SchemaT]
    ) -> LLMResult[SchemaT]:
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
        raw = self._call(role, messages, schema_json)
        try:
            value = schema.model_validate(json.loads(strip_to_json(raw.value)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMOutputError(f"{role}: reply is not valid {schema.__name__}: {exc}") from exc
        return LLMResult(
            value=value,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            cost_usd=raw.cost_usd,
            cached=raw.cached,
            latency_ms=raw.latency_ms,
            model=raw.model,
        )

    def text(self, role: str, messages: list[Message]) -> LLMResult[str]:
        return self._call(role, messages, None)

    @property
    def run_totals(self) -> dict[str, float]:
        """Real provider requests and spend so far (for the run manifest)."""
        with self._lock:
            return {"llm_calls": self._calls, "cost_usd": round(self._cost, 6)}

    # -------------------------------------------------------------------------- the call
    def _role(self, role: str) -> LLMRoleConfig:
        if role not in _ROLES:
            raise ConfigError(f"unknown LLM role {role!r}")
        role_cfg: LLMRoleConfig = getattr(self._cfg.llm, role)
        return role_cfg

    def _call(self, role: str, messages: list[Message], schema_json: str | None) -> LLMResult[str]:
        role_cfg = self._role(role)
        model = self._cfg.require_llm_model(role)  # type: ignore[arg-type]
        system = "\n\n".join(
            [m.content for m in messages if m.role == "system"]
            + ([f"{_SCHEMA_INSTRUCTION}\n{schema_json}"] if schema_json else [])
        )
        convo = [m for m in messages if m.role != "system"]
        key = self._key(role, role_cfg, model, messages, schema_json)

        cached = self._cache.get(key) if self._cache is not None else None
        if cached is not None:
            text, usage = cached
            return self._result(role, model, text, usage, cached=True, latency_ms=0.0, attempts=0)

        with self._lock:
            limits = self._cfg.llm
            if self._calls >= limits.max_calls_per_run or self._cost >= limits.max_cost_usd_per_run:
                raise LLMBudgetExceeded(
                    f"per-run LLM cap reached ({self._calls} calls, ${self._cost:.4f})"
                )
        provider = self._provider(role, role_cfg)
        start = self._clock()
        attempts = 0
        completion: Completion | None = None
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(role_cfg.max_transport_retries + 1),
                wait=wait_exponential_jitter(),
                retry=retry_if_exception_type(TransportError),
                sleep=self._sleep,
                reraise=True,
            ):
                with attempt:
                    attempts += 1
                    completion = provider.complete(
                        model, system, convo, role_cfg.max_output_tokens, role_cfg.temperature
                    )
        except TransportError as exc:
            raise LLMError(
                self._redact(f"{role}: {exc} after {attempts} attempts", role_cfg)
            ) from exc
        except LLMError as exc:
            raise LLMError(self._redact(str(exc), role_cfg)) from exc
        if completion is None:  # unreachable: Retrying ran at least once or raised
            raise LLMError(f"{role}: no completion")
        latency_ms = (self._clock() - start) * 1000.0
        estimated = completion.input_tokens is None or completion.output_tokens is None
        usage = {
            "input_tokens": completion.input_tokens
            if completion.input_tokens is not None
            else _estimate(system + "".join(m.content for m in convo)),
            "output_tokens": completion.output_tokens
            if completion.output_tokens is not None
            else _estimate(completion.text),
            "estimated_tokens": estimated,
        }
        with self._lock:
            self._calls += 1
            self._cost += self._price(model, usage)
        if self._cache is not None:
            self._cache.put(key, completion.text, usage)
        return self._result(
            role,
            model,
            completion.text,
            usage,
            cached=False,
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def _result(
        self,
        role: str,
        model: str,
        text: str,
        usage: dict[str, Any],
        *,
        cached: bool,
        latency_ms: float,
        attempts: int,
    ) -> LLMResult[str]:
        estimate = self._price(model, usage)
        cost = 0.0 if cached else estimate
        self._log.info(
            "llm.call",
            role=role,
            model=model,
            tokens=usage["input_tokens"] + usage["output_tokens"],
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cost_usd=round(cost, 6),
            cost_usd_equivalent=round(estimate, 6),
            cached=cached,
            latency_ms=round(latency_ms, 2),
            attempt=attempts,
            estimated_tokens=bool(usage.get("estimated_tokens", False)),
        )
        return LLMResult(
            value=text,
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
            cost_usd=cost,
            cached=cached,
            latency_ms=latency_ms,
            model=model,
        )

    # ------------------------------------------------------------------------- helpers
    def _key(
        self,
        role: str,
        role_cfg: LLMRoleConfig,
        model: str,
        messages: list[Message],
        schema_json: str | None,
    ) -> str:
        canonical = json.dumps(
            {
                "provider": role_cfg.provider,
                "model": model,
                "role": role,
                "temperature": role_cfg.temperature,
                "max_tokens": role_cfg.max_output_tokens,
                "messages": [m.model_dump() for m in messages],
                "schema": schema_json,
                "namespace": self._cfg.llm.cache_namespace,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _price(self, model: str, usage: Mapping[str, Any]) -> float:
        price = self._cfg.llm.prices.get(model)
        if price is None:
            return 0.0
        return (
            float(
                usage["input_tokens"] * price.input_per_mtok
                + usage["output_tokens"] * price.output_per_mtok
            )
            / 1_000_000
        )

    def _provider(self, role: str, role_cfg: LLMRoleConfig) -> Provider:
        with self._lock:
            if role in self._providers:
                return self._providers[role]
            key = self._environ.get(role_cfg.api_key_env, "")
            if role_cfg.provider == "anthropic":
                if not key:
                    raise ConfigError(
                        f"llm.{role}: environment variable {role_cfg.api_key_env} is not set"
                    )
                from engine.llm.providers.anthropic import AnthropicProvider

                provider: Provider = AnthropicProvider(key, role_cfg.timeout_s)
            elif role_cfg.provider == "openai_compatible":
                from engine.llm.providers.openai_compatible import OpenAICompatibleProvider

                provider = OpenAICompatibleProvider(role_cfg.base_url, key, role_cfg.timeout_s)
            else:
                raise ConfigError(f"llm.{role}.provider 'fake' is for tests (FakeLLMClient)")
            self._providers[role] = provider
            return provider

    def _redact(self, message: str, role_cfg: LLMRoleConfig) -> str:
        key = self._environ.get(role_cfg.api_key_env, "")
        return message.replace(key, "***") if key else message

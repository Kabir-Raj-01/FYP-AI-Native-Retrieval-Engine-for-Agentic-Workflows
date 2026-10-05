"""Provider wrappers map SDK responses and errors (LLM_CLIENT_SPEC §4, §6); no network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from anthropic.types import TextBlock

from engine.errors import LLMError
from engine.llm.interfaces import Message
from engine.llm.providers.anthropic import AnthropicProvider
from engine.llm.providers.base import TransportError
from engine.llm.providers.openai_compatible import OpenAICompatibleProvider

REQUEST = httpx2.Request("POST", "http://test.invalid/v1")
MSGS = [Message(role="user", content="hello")]


def status_error(module: Any, cls: str, code: int) -> Exception:
    response = httpx2.Response(code, request=REQUEST)
    return getattr(module, cls)("boom", response=response, body=None)  # type: ignore[no-any-return]


class Raising:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def create(self, **kwargs: Any) -> Any:
        raise self.exc


def anthropic_with(create: Any) -> AnthropicProvider:
    return AnthropicProvider("k", 5, client=SimpleNamespace(messages=create))


def openai_with(create: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        "http://x", "k", 5, client=SimpleNamespace(chat=SimpleNamespace(completions=create))
    )


def test_real_sdk_clients_have_retries_off() -> None:
    a = AnthropicProvider("k", 7)
    o = OpenAICompatibleProvider("http://127.0.0.1:1/v1", "", 7)
    assert (a._client.max_retries, o._client.max_retries) == (0, 0)
    assert (a._client.timeout, o._client.timeout) == (7, 7)


def test_anthropic_text_blocks_and_usage() -> None:
    calls: list[dict[str, Any]] = []

    class Create:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text='{"a": '), TextBlock(type="text", text="1}")],
                usage=SimpleNamespace(input_tokens=12, output_tokens=3),
            )

    c = anthropic_with(Create()).complete("m", "sys", MSGS, 600, 0.0)
    assert (c.text, c.input_tokens, c.output_tokens) == ('{"a": 1}', 12, 3)
    assert calls[0]["system"] == "sys" and "temperature" not in calls[0]


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (anthropic.APITimeoutError(request=REQUEST), TransportError),
        (anthropic.APIConnectionError(request=REQUEST), TransportError),
        (status_error(anthropic, "RateLimitError", 429), TransportError),
        (status_error(anthropic, "InternalServerError", 503), TransportError),
        (status_error(anthropic, "BadRequestError", 400), LLMError),
        (status_error(anthropic, "AuthenticationError", 401), LLMError),
    ],
)
def test_anthropic_error_mapping(exc: Exception, kind: type[Exception]) -> None:
    with pytest.raises(kind):
        anthropic_with(Raising(exc)).complete("m", "", MSGS, 10, 0.0)


def test_anthropic_refusal_is_an_llm_error() -> None:
    class Refuse:
        def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(stop_reason="refusal", content=[], usage=None)

    with pytest.raises(LLMError, match="refused"):
        anthropic_with(Refuse()).complete("m", "", MSGS, 10, 0.0)


def test_openai_compatible_messages_usage_and_missing_usage() -> None:
    calls: list[dict[str, Any]] = []

    class Create:
        def __init__(self, usage: Any) -> None:
            self.usage = usage

        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            msg = SimpleNamespace(message=SimpleNamespace(content="hi"))
            return SimpleNamespace(choices=[msg], usage=self.usage)

    c = openai_with(Create(SimpleNamespace(prompt_tokens=9, completion_tokens=2))).complete(
        "m", "sys", MSGS, 50, 0.0
    )
    assert (c.text, c.input_tokens, c.output_tokens) == ("hi", 9, 2)
    assert calls[0]["messages"][0] == {"role": "system", "content": "sys"}
    assert (calls[0]["temperature"], calls[0]["max_tokens"]) == (0.0, 50)
    local = openai_with(Create(None)).complete("m", "", MSGS, 50, 0.0)
    assert (local.input_tokens, local.output_tokens) == (None, None)


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (openai.APITimeoutError(request=REQUEST), TransportError),
        (status_error(openai, "RateLimitError", 429), TransportError),
        (status_error(openai, "InternalServerError", 500), TransportError),
        (status_error(openai, "NotFoundError", 404), LLMError),
    ],
)
def test_openai_error_mapping(exc: Exception, kind: type[Exception]) -> None:
    with pytest.raises(kind):
        openai_with(Raising(exc)).complete("m", "", MSGS, 10, 0.0)

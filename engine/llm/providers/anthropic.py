"""Claude models via the Anthropic API (LLM_CLIENT_SPEC §4; LIBRARY_NOTES "LLM providers").

anthropic 1.11.0 (checked on the installed package): ``messages.create`` takes ``system`` as
a separate parameter and has no ``temperature`` argument, so ``llm.<role>.temperature`` is
not sent — reproducibility comes from the response cache. Only ``text`` content blocks are
returned (thinking blocks are skipped). SDK retries are off (``max_retries=0``); the client
retries TransportError itself.
"""

from __future__ import annotations

from typing import Any

import anthropic
from anthropic.types import TextBlock

from engine.errors import LLMError
from engine.llm.interfaces import Message
from engine.llm.providers.base import Completion, TransportError

__all__ = ["AnthropicProvider"]

_RETRYABLE = (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)


class AnthropicProvider:
    def __init__(self, api_key: str, timeout_s: float, client: Any = None) -> None:
        self._client = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout_s, max_retries=0
        )

    def complete(
        self, model: str, system: str, messages: list[Message], max_tokens: int, temperature: float
    ) -> Completion:
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
        except _RETRYABLE as exc:  # APITimeoutError is an APIConnectionError
            raise TransportError(f"anthropic: {type(exc).__name__}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"anthropic: HTTP {exc.status_code} {type(exc).__name__}") from exc
        if response.stop_reason == "refusal":
            raise LLMError(f"anthropic: {model} refused the request")
        text = "".join(b.text for b in response.content if isinstance(b, TextBlock))
        usage = response.usage
        return Completion(text, usage.input_tokens, usage.output_tokens)

"""OpenAI API and OpenAI-compatible servers (Ollama, LM Studio, routers) via ``base_url``
(LLM_CLIENT_SPEC §4).

openai 3.24.0 (checked on the installed package): ``chat.completions.create(model, messages,
temperature, max_tokens)``; the system text is sent as the first message. ``usage`` may be
missing on local servers -> token counts None (the client estimates them). SDK retries off.
"""

from __future__ import annotations

from typing import Any

import openai
from openai.types.chat import ChatCompletionMessageParam

from engine.errors import LLMError
from engine.llm.interfaces import Message
from engine.llm.providers.base import Completion, TransportError

__all__ = ["OpenAICompatibleProvider"]

_RETRYABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, timeout_s: float, client: Any = None) -> None:
        self._client = client or openai.OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",  # local servers need no key; the SDK needs one
            timeout=timeout_s,
            max_retries=0,
        )

    def complete(
        self, model: str, system: str, messages: list[Message], max_tokens: int, temperature: float
    ) -> Completion:
        payload: list[ChatCompletionMessageParam] = (
            [{"role": "system", "content": system}] if system else []
        )
        for m in messages:
            payload.append({"role": m.role, "content": m.content})  # type: ignore[arg-type,misc]
        try:
            response = self._client.chat.completions.create(
                model=model, messages=payload, temperature=temperature, max_tokens=max_tokens
            )
        except _RETRYABLE as exc:  # APITimeoutError is an APIConnectionError
            raise TransportError(f"openai_compatible: {type(exc).__name__}") from exc
        except openai.APIStatusError as exc:
            raise LLMError(
                f"openai_compatible: HTTP {exc.status_code} {type(exc).__name__}"
            ) from exc
        text = response.choices[0].message.content or "" if response.choices else ""
        usage = response.usage
        if usage is None:
            return Completion(text, None, None)
        return Completion(text, usage.prompt_tokens, usage.completion_tokens)

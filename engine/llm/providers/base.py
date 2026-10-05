"""Provider contract for ``engine/llm`` (LLM_CLIENT_SPEC §4, §6).

A provider makes exactly one request (SDK retries disabled) and raises TransportError for
what the client may retry — timeouts, connection errors, HTTP 429 and 5xx — and LLMError
for anything else. Retries, caching, caps and accounting live in ``engine/llm/client.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engine.llm.interfaces import Message

__all__ = ["Completion", "Provider", "TransportError"]


class TransportError(Exception):
    """Retryable transport failure (timeout, connection, HTTP 429 / 5xx)."""


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int | None  # None when the server reports no usage (local servers)
    output_tokens: int | None


class Provider(Protocol):
    def complete(
        self, model: str, system: str, messages: list[Message], max_tokens: int, temperature: float
    ) -> Completion: ...

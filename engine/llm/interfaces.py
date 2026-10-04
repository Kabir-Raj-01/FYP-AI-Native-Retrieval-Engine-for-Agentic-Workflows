"""LLMClient interface and its transport types (LLM_CLIENT_SPEC §3; ADR-009 §10).

``Message`` and ``LLMResult`` live here, not in ``engine/state/models.py``: they are
LLM-transport types used only by ``engine/llm`` and ``engine/planner`` (ADR-009 §10).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

__all__ = ["LLMClient", "LLMResult", "Message"]

T = TypeVar("T")
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class Message(BaseModel):
    """One chat message. Retrieved text never goes in a ``system`` message (SECURITY §3)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    value: T
    input_tokens: int
    output_tokens: int
    cost_usd: float  # estimated from llm.prices; 0 when served from cache
    cached: bool
    latency_ms: float
    model: str


class LLMClient(ABC):
    """Every LLM call in the project (CLAUDE.md rule 5). Roles: planner, verifier,
    generator (``llm.<role>``)."""

    @abstractmethod
    def structured(
        self, role: str, messages: list[Message], schema: type[SchemaT]
    ) -> LLMResult[SchemaT]:
        """Ask ``role``'s model for JSON matching ``schema`` and validate it.

        Raises:
            LLMOutputError: the response is not valid JSON for ``schema`` (after code-fence
                stripping only).
            LLMError: transport failure after ``llm.<role>.max_transport_retries``.
            LLMBudgetExceeded: ``llm.max_calls_per_run`` or ``llm.max_cost_usd_per_run``
                reached.
            ConfigError: ``llm.<role>.model`` is not set.
        """
        raise NotImplementedError

    @abstractmethod
    def text(self, role: str, messages: list[Message]) -> LLMResult[str]:
        """Free-text completion for ``role`` (the eval generator). Same errors as
        :meth:`structured`, except ``LLMOutputError``."""
        raise NotImplementedError

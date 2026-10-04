"""FakeLLMClient (TESTING §2): scripted responses per role, every request recorded."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from engine.errors import LLMOutputError
from engine.llm.interfaces import LLMClient, LLMResult, Message, SchemaT

# A scripted response: a model instance, a JSON string, a dict to validate, a plain string
# (for `text`), or an exception instance to raise.
Response = BaseModel | str | Mapping[str, Any] | Exception

DEFAULT_MODEL = "fake-model"


@dataclass(frozen=True)
class LLMRequest:
    role: str
    messages: list[Message]
    schema: type[BaseModel] | None  # None for text()


class FakeLLMClient(LLMClient):
    """Pops one scripted response per call from the role's queue.

    A string or dict that does not validate against ``schema`` raises ``LLMOutputError``,
    as the real client does. An exhausted queue fails the test loudly.
    """

    def __init__(
        self,
        responses: Mapping[str, Sequence[Response]] | None = None,
        *,
        input_tokens: int = 10,
        output_tokens: int = 5,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._queues: dict[str, deque[Response]] = {
            role: deque(items) for role, items in (responses or {}).items()
        }
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.requests: list[LLMRequest] = []

    def enqueue(self, role: str, *responses: Response) -> None:
        self._queues.setdefault(role, deque()).extend(responses)

    def structured(
        self, role: str, messages: list[Message], schema: type[SchemaT]
    ) -> LLMResult[SchemaT]:
        self.requests.append(LLMRequest(role, list(messages), schema))
        response = self._next(role)
        try:
            if isinstance(response, schema):
                value = response
            elif isinstance(response, str):
                value = schema.model_validate_json(response)
            elif isinstance(response, Mapping):
                value = schema.model_validate(dict(response))
            else:
                raise LLMOutputError(f"scripted {type(response).__name__} is not a {schema}")
        except ValidationError as exc:
            raise LLMOutputError(f"response does not match {schema.__name__}: {exc}") from exc
        return self._result(value)

    def text(self, role: str, messages: list[Message]) -> LLMResult[str]:
        self.requests.append(LLMRequest(role, list(messages), None))
        response = self._next(role)
        if not isinstance(response, str):
            raise TypeError(f"text() needs a scripted str, got {type(response).__name__}")
        return self._result(response)

    def _next(self, role: str) -> Response:
        queue = self._queues.get(role)
        if not queue:
            raise AssertionError(f"FakeLLMClient: no scripted response left for role {role!r}")
        response = queue.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    def _result(self, value: Any) -> LLMResult[Any]:
        return LLMResult[Any](
            value=value,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=0.0,
            cached=False,
            latency_ms=0.0,
            model=self.model,
        )

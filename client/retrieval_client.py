"""HTTP client for the retrieval engine (API_CONTRACT "Python client").

    client = RetrievalClient("http://127.0.0.1:8000")
    task = client.create_task(goal="...", budget=BudgetRequest(max_retrieval_calls=4))
    bundle = client.run(task.task_id)            # or: while not (r := client.step(id)).done: ...
    direct = client.retrieve_direct("...", mode=RetrievalMode.HYBRID, k=10)

API error bodies ``{"error": {"code", "message", "task_id"}}`` become the exceptions below
(one class per API_CONTRACT error code). ``timeout_s`` should match the server's
``api.client_timeout_s`` (longer than ``api.run_timeout_s``, so ``/run`` can answer 504).
"""

from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter

from client.models import (
    BudgetRequest,
    CreateTaskRequest,
    CreateTaskResponse,
    DirectRequest,
    DirectResponse,
    EvidenceBundle,
    HealthResponse,
    RetrievalMode,
    RetrievalState,
    StepResponse,
    TraceStep,
)

__all__ = [
    "BackendUnavailable",
    "InternalError",
    "LLMUnavailable",
    "RetrievalClient",
    "RetrievalClientError",
    "RunBudgetExceeded",
    "RunTimeout",
    "TaskConflict",
    "TaskExists",
    "TaskNotActive",
    "TaskNotFound",
    "ValidationFailed",
]

M = TypeVar("M", bound=BaseModel)


class RetrievalClientError(Exception):
    def __init__(self, code: str, message: str, task_id: str | None, status: int) -> None:
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.task_id, self.status = code, message, task_id, status


class TaskNotFound(RetrievalClientError): ...  # noqa: N818 — names mirror the API codes


class TaskExists(RetrievalClientError): ...  # noqa: N818


class TaskNotActive(RetrievalClientError): ...  # noqa: N818


class TaskConflict(RetrievalClientError): ...  # noqa: N818


class ValidationFailed(RetrievalClientError): ...  # noqa: N818


class LLMUnavailable(RetrievalClientError): ...  # noqa: N818


class BackendUnavailable(RetrievalClientError): ...  # noqa: N818


class RunTimeout(RetrievalClientError): ...  # noqa: N818


class RunBudgetExceeded(RetrievalClientError): ...  # noqa: N818


class InternalError(RetrievalClientError): ...


_BY_CODE: dict[str, type[RetrievalClientError]] = {
    "TASK_NOT_FOUND": TaskNotFound,
    "TASK_EXISTS": TaskExists,
    "TASK_NOT_ACTIVE": TaskNotActive,
    "TASK_CONFLICT": TaskConflict,
    "VALIDATION_ERROR": ValidationFailed,
    "LLM_ERROR": LLMUnavailable,
    "BACKEND_ERROR": BackendUnavailable,
    "RUN_TIMEOUT": RunTimeout,
    "RUN_BUDGET_EXCEEDED": RunBudgetExceeded,
    "INTERNAL_ERROR": InternalError,
}


def _without_computed(state: dict[str, Any]) -> dict[str, Any]:
    """Drop RetrievalState's output-only computed keys before validating (ADR-009 §4a):
    they are serialised for readers but rejected on input; they recompute on load."""
    return {k: v for k, v in state.items() if k not in _COMPUTED}


_COMPUTED = ("current_subgoal", "pending_query", "current_need_type")


class RetrievalClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 150.0,  # mirrors api.client_timeout_s in configs/base.yaml
        http: httpx.Client | None = None,
    ) -> None:
        """``http`` replaces the transport (tests pass a FastAPI TestClient)."""
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout_s)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> RetrievalClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------------------------------- endpoints
    def health(self) -> HealthResponse:
        return self._call("GET", "/v1/health", HealthResponse)

    def create_task(
        self, goal: str, budget: BudgetRequest | None = None, task_id: str | None = None
    ) -> CreateTaskResponse:
        body = CreateTaskRequest(goal=goal, budget=budget or BudgetRequest(), task_id=task_id)
        data = self._send("POST", "/v1/tasks", body).json()
        data["state"] = _without_computed(data["state"])
        return CreateTaskResponse.model_validate(data)

    def step(self, task_id: str) -> StepResponse:
        return self._call("POST", f"/v1/tasks/{task_id}/step", StepResponse)

    def run(self, task_id: str) -> EvidenceBundle:
        return self._call("POST", f"/v1/tasks/{task_id}/run", EvidenceBundle)

    def result(self, task_id: str) -> EvidenceBundle:
        return self._call("GET", f"/v1/tasks/{task_id}/result", EvidenceBundle)

    def state(self, task_id: str) -> RetrievalState:
        data = self._send("GET", f"/v1/tasks/{task_id}/state").json()
        return RetrievalState.model_validate(_without_computed(data))

    def trace(self, task_id: str) -> list[TraceStep]:
        response = self._send("GET", f"/v1/tasks/{task_id}/trace")
        return TypeAdapter(list[TraceStep]).validate_python(response.json())

    def retrieve_direct(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.DENSE,
        k: int | None = None,
        rerank: bool = False,
    ) -> DirectResponse:
        body = DirectRequest(query=query, mode=mode, k=k, rerank=rerank)
        return self._call("POST", "/v1/retrieve/direct", DirectResponse, body)

    # ---------------------------------------------------------------------------- helpers
    def _call(self, method: str, path: str, model: type[M], body: BaseModel | None = None) -> M:
        return model.model_validate(self._send(method, path, body).json())

    def _send(self, method: str, path: str, body: BaseModel | None = None) -> httpx.Response:
        payload: Any = body.model_dump(mode="json") if body is not None else None
        response = self._http.request(method, path, json=payload)
        if response.is_success:
            return response
        try:
            error = response.json()["error"]
        except (ValueError, KeyError, TypeError):
            error = {"code": "INTERNAL_ERROR", "message": response.text, "task_id": None}
        cls = _BY_CODE.get(error.get("code", ""), RetrievalClientError)
        raise cls(
            error.get("code", ""),
            error.get("message", ""),
            error.get("task_id"),
            response.status_code,
        )

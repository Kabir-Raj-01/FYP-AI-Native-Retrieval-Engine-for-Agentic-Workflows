"""The controlled path over HTTP: /v1/tasks (API_CONTRACT; ADR-010).

Thin layer over ``Engine``: request limits from config (422), errors via the app's
EngineError handler (404 TASK_NOT_FOUND, 409 TASK_EXISTS / TASK_NOT_ACTIVE / TASK_CONFLICT,
502 LLM_ERROR, 503 BACKEND_ERROR, 504 RUN_TIMEOUT, 429 RUN_BUDGET_EXCEEDED). Sync handlers
run in FastAPI's threadpool; the Engine's per-task lock makes a concurrent step a 409.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from engine.api.routers.direct import invalid
from engine.api.schemas import BudgetRequest, CreateTaskRequest, CreateTaskResponse, StepResponse
from engine.core import Engine
from engine.state.models import (
    Budget,
    EvidenceBundle,
    GapStatus,
    RetrievalState,
    TaskStatus,
    TraceStep,
)

__all__ = ["router"]

router = APIRouter()


def _engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


def _budget(engine: Engine, req: BudgetRequest) -> Budget:
    return engine.default_budget().model_copy(update=req.model_dump(exclude_none=True))


@router.post("/tasks", response_model=CreateTaskResponse, status_code=201)
def create_task(body: CreateTaskRequest, request: Request) -> CreateTaskResponse:
    engine = _engine(request)
    api = engine.cfg.api
    if not api.goal_min_chars <= len(body.goal) <= api.goal_max_chars:
        raise invalid("goal", f"length must be {api.goal_min_chars}..{api.goal_max_chars}")
    state = engine.create_task(body.goal, _budget(engine, body.budget), body.task_id)
    return CreateTaskResponse(task_id=state.task_id, state=state)


@router.post("/tasks/{task_id}/step", response_model=StepResponse)
def step(task_id: str, request: Request) -> StepResponse:
    engine = _engine(request)
    trace = engine.step(task_id)
    state = engine.state(task_id)
    return StepResponse(
        task_id=task_id,
        step=trace.step,
        action=trace.action,
        done=state.status != TaskStatus.ACTIVE,
        stop_reason=state.stop_reason,
        new_evidence=[e for e in state.evidence if e.added_at_step == trace.step],
        open_gaps=[g for g in state.gaps if g.status == GapStatus.OPEN],
        budget=state.budget,
    )


@router.post("/tasks/{task_id}/run", response_model=EvidenceBundle)
def run(task_id: str, request: Request) -> EvidenceBundle:
    return _engine(request).run(task_id)


@router.get("/tasks/{task_id}/result", response_model=EvidenceBundle)
def result(task_id: str, request: Request) -> EvidenceBundle:
    return _engine(request).result(task_id)


@router.get("/tasks/{task_id}/state", response_model=RetrievalState)
def state(task_id: str, request: Request) -> RetrievalState:
    return _engine(request).state(task_id)


@router.get("/tasks/{task_id}/trace", response_model=list[TraceStep])
def trace(task_id: str, request: Request) -> list[TraceStep]:
    return _engine(request).trace(task_id)

"""HTTP-only request/response models (API_CONTRACT.md "Schemas"; CLAUDE.md rule 2).

Exactly as API_CONTRACT specifies (as amended by ADR-009 §8 and ADR-010: request limits
are checked against config by the routers, not baked in here).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from engine.state.models import (
    Action,
    Budget,
    Candidate,
    EvidenceGap,
    EvidenceItem,
    RetrievalMode,
    RetrievalState,
    StopReason,
)

__all__ = [
    "BudgetRequest",
    "CreateTaskRequest",
    "CreateTaskResponse",
    "DirectRequest",
    "DirectResponse",
    "HealthResponse",
    "StepResponse",
]


class BudgetRequest(BaseModel):  # every field optional -> budget_defaults.*
    max_steps: int | None = Field(default=None, ge=1)
    max_retrieval_calls: int | None = Field(default=None, ge=1)
    max_llm_calls: int | None = Field(default=None, ge=0)  # 0 = no-LLM ablation
    max_tokens: int | None = Field(default=None, ge=1)


class CreateTaskRequest(BaseModel):
    goal: str = Field(min_length=1)  # length bounds: api.goal_min_chars..goal_max_chars (router)
    budget: BudgetRequest = BudgetRequest()
    task_id: str | None = None  # optional caller-supplied id (eval uses dataset ids)


class CreateTaskResponse(BaseModel):
    task_id: str
    state: RetrievalState


class StepResponse(BaseModel):
    task_id: str
    step: int
    action: Action
    done: bool
    stop_reason: StopReason | None
    new_evidence: list[EvidenceItem]
    open_gaps: list[EvidenceGap]
    budget: Budget


class DirectRequest(BaseModel):
    query: str = Field(min_length=1)  # max length: api.query_max_chars (router, ADR-010)
    mode: RetrievalMode = RetrievalMode.DENSE
    k: int | None = Field(default=None, ge=1)  # None -> retrieval.default_k; > max_k -> 422
    rerank: bool = False  # relevance-only cross-encoder rerank


class DirectResponse(BaseModel):
    candidates: list[Candidate]  # without vectors
    relevances: list[float] | None  # present when rerank=true
    latency_ms: float


class HealthResponse(BaseModel):
    """``GET /v1/health`` body: ``{status, qdrant: bool, version, config_hash}``."""

    status: str  # "ok" | "degraded" (backend unreachable or collection missing)
    qdrant: bool
    version: str
    config_hash: str

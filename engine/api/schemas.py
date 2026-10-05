"""HTTP-only request/response models (API_CONTRACT.md "Schemas"; CLAUDE.md rule 2).

``DirectRequest``/``DirectResponse`` are exactly as API_CONTRACT specifies (as amended by
ADR-010: limits are checked against config by the router, not baked in here). The task
schemas (``BudgetRequest``, ``CreateTaskRequest``, ...) come with the ``/tasks`` routes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from engine.state.models import Candidate, RetrievalMode

__all__ = ["DirectRequest", "DirectResponse", "HealthResponse"]


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

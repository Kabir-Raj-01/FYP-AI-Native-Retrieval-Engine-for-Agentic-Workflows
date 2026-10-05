"""POST /v1/retrieve/direct — controller-OFF retrieval (API_CONTRACT; ADR-010; ADR-012)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError

from engine.api.schemas import DirectRequest, DirectResponse
from engine.core import Engine

__all__ = ["router"]

router = APIRouter()


def _invalid(field: str, message: str) -> RequestValidationError:
    return RequestValidationError([{"loc": ("body", field), "msg": message, "type": "value_error"}])


@router.post("/retrieve/direct", response_model=DirectResponse)
def retrieve_direct(body: DirectRequest, request: Request) -> DirectResponse:
    engine: Engine = request.app.state.engine
    cfg = engine.cfg
    # Request limits come from configuration (ADR-010) -> 422 VALIDATION_ERROR.
    if len(body.query) > cfg.api.query_max_chars:
        raise _invalid("query", f"longer than api.query_max_chars ({cfg.api.query_max_chars})")
    if body.k is not None and body.k > cfg.retrieval.max_k:
        raise _invalid("k", f"greater than retrieval.max_k ({cfg.retrieval.max_k})")
    if body.rerank and not engine.has_reranker:
        raise _invalid("rerank", "rerank is not available yet (ADR-005)")

    result = engine.retrieve_direct(body.query, body.mode, body.k, body.rerank)
    if result.scored is not None:
        return DirectResponse(
            candidates=[s.candidate for s in result.scored],
            relevances=[s.relevance for s in result.scored],
            latency_ms=result.latency_ms,
        )
    return DirectResponse(
        candidates=result.candidates, relevances=None, latency_ms=result.latency_ms
    )

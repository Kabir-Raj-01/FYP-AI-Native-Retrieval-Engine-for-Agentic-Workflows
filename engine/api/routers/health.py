"""GET /v1/health — liveness + backend check (API_CONTRACT "Endpoints")."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Request

from engine.api.schemas import HealthResponse
from engine.core import Engine

__all__ = ["router"]

router = APIRouter()
_PACKAGE = "retrieval-engine"


def _version() -> str:
    try:
        return version(_PACKAGE)
    except PackageNotFoundError:  # running from a source tree without installing
        return "unknown"


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Always 200 while the process is alive; ``status`` reports the backend."""
    engine: Engine = request.app.state.engine
    qdrant = engine.health()
    return HealthResponse(
        status="ok" if qdrant else "degraded",
        qdrant=qdrant,
        version=_version(),
        config_hash=request.app.state.config_hash,
    )

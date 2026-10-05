"""FastAPI application (API_CONTRACT.md; ARCHITECTURE §1 "thin routers").

    python -m uvicorn engine.api.app:app --host 127.0.0.1 --port 8000

The Engine is built once at start-up (lifespan) from the config named by
``ENGINE_CONFIG`` (default ``configs/base.yaml``). Errors use the API_CONTRACT body
``{"error": {"code", "message", "task_id"}}``: engine exceptions carry their code and
status (``engine/errors.py``); request-validation failures are 422 ``VALIDATION_ERROR``;
anything else is 500 ``INTERNAL_ERROR`` without internals in the message.

Controlled tasks (``/v1/tasks``) keep their state in ``paths.state_db`` and their traces
and bundles under ``paths.runs_dir/server/`` (run_id ``server``, system ``controller_on``),
where ``GET /v1/tasks/{id}/trace`` reads them back.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from engine import factory
from engine.api.routers import direct, health, tasks
from engine.config import config_hash, load_config
from engine.core import Engine
from engine.errors import EngineError
from engine.monitor.logging import configure_logging, get_logger
from engine.state.models import RunContext

__all__ = ["app", "create_app", "error_body"]

_log = get_logger("engine.api")


def error_body(code: str, message: str, task_id: str | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "task_id": task_id}}


def create_app(engine: Engine | None = None) -> FastAPI:
    """App factory; tests pass a ready Engine (built from fakes), the server builds its own."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if engine is not None:
            built = engine
        else:
            cfg = load_config()
            server = RunContext(run_id="server", system="controller_on")
            traces = factory.build_trace_writer(cfg, server, cfg.paths.runs_dir / "server")
            built = Engine.from_config(cfg, traces=traces)
        configure_logging(built.cfg.logging)
        app.state.engine = built
        app.state.config_hash = config_hash(built.cfg)
        yield

    app = FastAPI(title="AI-Native Retrieval Engine", version="v1", lifespan=lifespan)

    @app.exception_handler(EngineError)
    def engine_error(request: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse(
            status_code=int(exc.http_status), content=error_body(exc.code, exc.message, exc.task_id)
        )

    @app.exception_handler(RequestValidationError)
    def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}"
            for e in exc.errors()
        )
        return JSONResponse(status_code=422, content=error_body("VALIDATION_ERROR", problems))

    @app.exception_handler(Exception)
    def internal_error(request: Request, exc: Exception) -> JSONResponse:
        _log.error(
            "error", type=type(exc).__name__, message=str(exc), component="api", retryable=False
        )
        return JSONResponse(
            status_code=500, content=error_body("INTERNAL_ERROR", "internal server error")
        )

    app.include_router(health.router, prefix="/v1")
    app.include_router(direct.router, prefix="/v1")
    app.include_router(tasks.router, prefix="/v1")
    return app


app = create_app()

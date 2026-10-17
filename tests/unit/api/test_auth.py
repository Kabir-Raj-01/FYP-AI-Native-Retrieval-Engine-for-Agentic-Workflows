"""Optional bearer token (SECURITY §2, ADR-015; SRS NFR-S3)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import engine.api.app as app_module
from client import RetrievalClient, Unauthorized
from engine.api.app import API_TOKEN_ENV, create_app
from engine.config import EngineConfig
from tests.conftest import EngineFactory

TOKEN = "s3cret-test-token"  # pragma: allowlist secret — a dummy for tests
GOAL = "Linden Bay tram expansion council harbour"


def test_without_a_token_no_header_is_needed(engine_with_fakes: EngineFactory) -> None:
    with TestClient(create_app(engine_with_fakes().engine)) as http:
        assert http.get("/v1/health").status_code == 200


def test_with_a_token_v1_needs_the_right_bearer_header(engine_with_fakes: EngineFactory) -> None:
    with TestClient(create_app(engine_with_fakes().engine, api_token=TOKEN)) as http:
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}):
            r = http.get("/v1/health", headers=headers)
            assert r.status_code == 401
            assert r.json()["error"]["code"] == "UNAUTHORIZED"
            assert r.headers["WWW-Authenticate"] == "Bearer"
            assert TOKEN not in r.text  # never echoed
        ok = http.get("/v1/health", headers={"Authorization": f"Bearer {TOKEN}"})
        assert ok.status_code == 200


def test_rejected_requests_never_reach_the_engine(engine_with_fakes: EngineFactory) -> None:
    fake = engine_with_fakes()
    with TestClient(create_app(fake.engine, api_token=TOKEN)) as http:
        assert http.post("/v1/tasks", json={"goal": GOAL, "task_id": "t-1"}).status_code == 401
        body = {"query": GOAL, "mode": "dense"}
        assert http.post("/v1/retrieve/direct", json=body).status_code == 401
        assert http.get("/v1/tasks/t-1/state").status_code == 401
    assert fake.retriever.requests == []
    with TestClient(create_app(fake.engine)) as http:  # same store, no token: nothing created
        assert http.get("/v1/tasks/t-1/state").status_code == 404


def test_docs_stay_readable_with_a_token(engine_with_fakes: EngineFactory) -> None:
    with TestClient(create_app(engine_with_fakes().engine, api_token=TOKEN)) as http:
        assert http.get("/docs").status_code == 200
        assert http.get("/openapi.json").status_code == 200


def test_client_sends_the_token(engine_with_fakes: EngineFactory) -> None:
    with TestClient(create_app(engine_with_fakes().engine, api_token=TOKEN)) as http:
        assert RetrievalClient("http://testserver", token=TOKEN, http=http).health().status == "ok"
    with TestClient(create_app(engine_with_fakes().engine, api_token=TOKEN)) as http:
        with pytest.raises(Unauthorized) as exc:
            RetrievalClient("http://testserver", http=http).health()
        assert (exc.value.status, exc.value.code) == (401, "UNAUTHORIZED")


def test_a_passed_in_engine_ignores_the_environment(
    engine_with_fakes: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_TOKEN_ENV, TOKEN)  # a developer's shell must not break unit tests
    with TestClient(create_app(engine_with_fakes().engine)) as http:
        assert http.get("/v1/health").status_code == 200


def test_the_server_reads_the_token_from_the_environment(
    cfg: EngineConfig, engine_with_fakes: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = engine_with_fakes()

    def from_config(_cfg: EngineConfig, **_: Any) -> Any:
        return fake.engine

    monkeypatch.setattr(app_module, "load_config", lambda: cfg)
    monkeypatch.setattr(app_module.Engine, "from_config", from_config)
    monkeypatch.setattr(app_module.factory, "build_trace_writer", lambda *a, **k: None)
    monkeypatch.setenv(API_TOKEN_ENV, TOKEN)
    with TestClient(create_app()) as http:  # the server path: no Engine passed in
        assert http.get("/v1/health").status_code == 401
        auth = {"Authorization": f"Bearer {TOKEN}"}
        assert http.get("/v1/health", headers=auth).status_code == 200
    monkeypatch.setenv(API_TOKEN_ENV, "")  # empty = check off
    with TestClient(create_app()) as http:
        assert http.get("/v1/health").status_code == 200

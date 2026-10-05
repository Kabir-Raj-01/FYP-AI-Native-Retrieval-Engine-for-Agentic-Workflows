"""/v1/tasks routes on an Engine built from fakes (API_CONTRACT Endpoints, Behaviour, Errors)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from itertools import count
from typing import Any

import pytest
from fastapi.testclient import TestClient

from engine.api.app import create_app
from engine.config import EngineConfig
from engine.state.models import Action, CacheHit, RetrievalState
from tests.conftest import EngineFactory, FakeEngine
from tests.fakes import CorpusChunk, FakeController, FakeRetriever
from tests.unit.api.test_api import assert_error

GOAL = "Linden Bay tram expansion council harbour"


def http(fake: FakeEngine) -> Iterator[TestClient]:
    with TestClient(create_app(fake.engine), raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def fake(engine_with_fakes: EngineFactory) -> FakeEngine:
    return engine_with_fakes(controller=FakeController([Action.DENSE, Action.SPARSE_HYBRID]))


@pytest.fixture
def client(fake: FakeEngine) -> Iterator[TestClient]:
    yield from http(fake)


def create(client: TestClient, **body: Any) -> Any:
    return client.post("/v1/tasks", json={"goal": GOAL, "task_id": "t-1", **body})


def test_create_uses_budget_defaults_and_overrides(client: TestClient, cfg: EngineConfig) -> None:
    r = create(client, budget={"max_steps": 3})
    assert r.status_code == 201
    body = r.json()
    assert body["task_id"] == "t-1" and body["state"]["goal"] == GOAL
    b = body["state"]["budget"]
    assert (b["max_steps"], b["max_retrieval_calls"]) == (
        3,
        cfg.budget_defaults.max_retrieval_calls,
    )
    assert body["state"]["pending_query"] == ""  # computed fields present (STATE_SPEC §4)
    assert_error(create(client), 409, "TASK_EXISTS")


@pytest.mark.parametrize("goal", ["ab", "x" * 2001])
def test_goal_length_limits_from_config(client: TestClient, goal: str) -> None:
    assert_error(client.post("/v1/tasks", json={"goal": goal}), 422, "VALIDATION_ERROR")


def test_step_response_and_trace(client: TestClient) -> None:
    create(client, budget={"max_steps": 2})
    r = client.post("/v1/tasks/t-1/step").json()
    assert (r["task_id"], r["step"], r["action"], r["done"], r["stop_reason"]) == (
        "t-1",
        1,
        "DENSE",
        False,
        None,
    )
    assert r["new_evidence"] and all(e["added_at_step"] == 1 for e in r["new_evidence"])
    assert all(e["vector"] is None for e in r["new_evidence"])  # vectors never serialised
    assert [g["gap_id"] for g in r["open_gaps"]] == ["g0"] and r["budget"]["steps_used"] == 1
    last = client.post("/v1/tasks/t-1/step").json()
    assert (last["done"], last["stop_reason"]) == (True, "MAX_STEPS")
    trace = client.get("/v1/tasks/t-1/trace").json()
    assert [t["step"] for t in trace] == [1, 2]
    assert_error(client.post("/v1/tasks/t-1/step"), 409, "TASK_NOT_ACTIVE")


def test_run_result_and_state(client: TestClient) -> None:
    create(client, budget={"max_steps": 2})
    assert_error(client.get("/v1/tasks/t-1/result"), 409, "TASK_NOT_ACTIVE")
    bundle = client.post("/v1/tasks/t-1/run").json()
    assert (bundle["stop_reason"], bundle["system"]) == ("MAX_STEPS", "controller_on")
    assert client.get("/v1/tasks/t-1/result").json() == bundle
    state = client.get("/v1/tasks/t-1/state").json()
    assert (state["status"], state["version"]) == ("stopped", 2)
    assert_error(client.post("/v1/tasks/t-1/run"), 409, "TASK_NOT_ACTIVE")


@pytest.mark.parametrize(
    ("method", "path"),
    [("post", "step"), ("post", "run"), ("get", "result"), ("get", "state"), ("get", "trace")],
)
def test_unknown_task_is_404(client: TestClient, method: str, path: str) -> None:
    err = assert_error(getattr(client, method)(f"/v1/tasks/nope/{path}"), 404, "TASK_NOT_FOUND")
    assert err["task_id"] == "nope"


def test_backend_failure_is_503_and_leaves_the_state(
    engine_with_fakes: EngineFactory, mini_corpus: list[CorpusChunk]
) -> None:
    fake = engine_with_fakes(retriever=FakeRetriever(mini_corpus, fail_on_call=1))
    for client in http(fake):
        create(client)
        assert_error(client.post("/v1/tasks/t-1/step"), 503, "BACKEND_ERROR")
        state = client.get("/v1/tasks/t-1/state").json()
        assert (state["status"], state["version"]) == ("active", 0)


def test_run_timeout_is_504(cfg: EngineConfig, engine_with_fakes: EngineFactory) -> None:
    cfg.api.run_timeout_s = 1
    ticks = count(0.0, 0.4)
    fake = engine_with_fakes(clock=lambda: next(ticks))
    for client in http(fake):
        create(client)
        assert_error(client.post("/v1/tasks/t-1/run"), 504, "RUN_TIMEOUT")
        assert client.get("/v1/tasks/t-1/state").json()["status"] == "active"


def test_concurrent_step_is_409_task_conflict(engine_with_fakes: EngineFactory) -> None:
    entered, release = threading.Event(), threading.Event()

    class Blocking(FakeController):
        def select_action(self, state: RetrievalState, cache_hit: CacheHit | None) -> Any:
            entered.set()
            release.wait(5)
            return super().select_action(state, cache_hit)

    fake = engine_with_fakes(controller=Blocking())
    for client in http(fake):
        create(client)
        worker = threading.Thread(target=fake.engine.step, args=("t-1",))
        worker.start()
        assert entered.wait(5)
        assert_error(client.post("/v1/tasks/t-1/step"), 409, "TASK_CONFLICT")
        release.set()
        worker.join()

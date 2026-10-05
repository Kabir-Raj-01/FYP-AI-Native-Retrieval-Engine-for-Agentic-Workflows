"""client/retrieval_client.py against the real app on fakes (API_CONTRACT "Python client")."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from client import RetrievalClient, TaskExists, TaskNotActive, TaskNotFound, ValidationFailed
from client.models import BudgetRequest, RetrievalMode
from engine.api.app import create_app
from tests.conftest import EngineFactory

GOAL = "Linden Bay tram expansion council harbour"


@pytest.fixture
def client(engine_with_fakes: EngineFactory) -> Iterator[RetrievalClient]:
    fake = engine_with_fakes()
    with TestClient(create_app(fake.engine), raise_server_exceptions=False) as http:
        yield RetrievalClient("http://testserver", http=http)


def test_task_lifecycle(client: RetrievalClient) -> None:
    task = client.create_task(GOAL, BudgetRequest(max_steps=2), task_id="t-1")
    assert task.task_id == "t-1" and task.state.pending_query == ""
    first = client.step("t-1")
    assert (first.step, first.done) == (1, False)
    bundle = client.run("t-1")
    assert bundle.stop_reason == "MAX_STEPS" and client.result("t-1") == bundle
    state = client.state("t-1")
    assert state.status == "stopped" and state.pending_query == GOAL  # recomputed
    assert [t.step for t in client.trace("t-1")] == [1, 2]


def test_error_codes_become_exceptions(client: RetrievalClient) -> None:
    with pytest.raises(TaskNotFound) as exc:
        client.step("nope")
    assert (exc.value.status, exc.value.task_id) == (404, "nope")
    client.create_task(GOAL, BudgetRequest(max_steps=1), task_id="t-1")
    with pytest.raises(TaskExists):
        client.create_task(GOAL, task_id="t-1")
    with pytest.raises(TaskNotActive):
        client.result("t-1")
    with pytest.raises(ValidationFailed):
        client.create_task("ab")


def test_direct_retrieval_and_health(client: RetrievalClient) -> None:
    direct = client.retrieve_direct("Linden Bay tram", mode=RetrievalMode.HYBRID, k=3)
    assert 1 <= len(direct.candidates) <= 3 and direct.relevances is None
    assert client.health().status in {"ok", "degraded"}

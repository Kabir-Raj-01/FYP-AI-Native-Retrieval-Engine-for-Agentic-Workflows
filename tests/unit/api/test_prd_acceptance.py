"""PRD §3 acceptance criteria not covered elsewhere: FR2 (state survives a server restart),
FR3 (direct retrieval creates no state; both paths share one RetrieverAdapter), FR11
(OpenAPI docs render at ``/docs``) and SRS NFR-R3 (direct mode works with the planner,
cache and controller disabled)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from engine.api.app import create_app
from engine.config import EngineConfig
from engine.core import Engine
from engine.monitor.trace import JsonlTraceWriter
from engine.state.models import RunContext
from tests.conftest import EngineFactory
from tests.fakes import CorpusChunk, FakeEmbedder, FakeReranker, FakeRetriever

GOAL = "Linden Bay tram expansion council harbour"


def test_fr2_task_state_survives_a_server_restart(engine_with_fakes: EngineFactory) -> None:
    with TestClient(create_app(engine_with_fakes().engine)) as before:
        assert before.post("/v1/tasks", json={"goal": GOAL, "task_id": "t-1"}).status_code == 201
        before.post("/v1/tasks/t-1/step")
        saved = before.get("/v1/tasks/t-1/state").json()
    # a new Engine and app over the same state database = the server restarted
    with TestClient(create_app(engine_with_fakes().engine)) as after:
        assert after.get("/v1/tasks/t-1/state").json() == saved


def test_fr3_direct_retrieval_creates_no_state(engine_with_fakes: EngineFactory) -> None:
    fake = engine_with_fakes()
    with TestClient(create_app(fake.engine)) as client:
        client.post("/v1/tasks", json={"goal": GOAL, "task_id": "t-1"})  # the table exists
        for mode in ("dense", "sparse", "hybrid"):
            assert client.post("/v1/retrieve/direct", json={"query": GOAL, "mode": mode}).is_success
    with sqlite3.connect(fake.db) as db:
        assert db.execute("SELECT task_id FROM tasks").fetchall() == [("t-1",)]
        assert db.execute("SELECT COUNT(*) FROM cache_entries").fetchone() == (0,)


def test_fr3_controlled_path_uses_the_direct_paths_retriever(
    cfg: EngineConfig, tmp_path: Path, mini_corpus: list[CorpusChunk]
) -> None:
    cfg.paths.state_db = tmp_path / "state.sqlite3"
    embedder = FakeEmbedder()
    retriever = FakeRetriever(mini_corpus, embedder=embedder)
    engine = Engine(
        cfg,
        retriever=retriever,
        reranker=FakeReranker(default=0.5),
        embedder=embedder,
        traces=JsonlTraceWriter(tmp_path / "run", RunContext(run_id="r", system="controller_on")),
    )
    assert engine._parts().retriever is retriever  # built by the factory, not passed in


def test_fr11_openapi_docs_render(engine_with_fakes: EngineFactory) -> None:
    with TestClient(create_app(engine_with_fakes().engine)) as client:
        docs = client.get("/docs")
        assert docs.status_code == 200 and "swagger-ui" in docs.text
        assert client.get("/openapi.json").json()["paths"]


class _Disabled:
    """Stands in for a disabled controlled-path component: any use fails the test."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> object:
        raise AssertionError(f"direct mode used the disabled {self._name} ({attr})")


def test_nfr_r3_direct_mode_works_with_the_controlled_components_disabled(
    cfg: EngineConfig, engine_with_fakes: EngineFactory
) -> None:
    cfg.ablation.planner = False
    cfg.ablation.cache = False
    off = {n: _Disabled(n) for n in ("planner", "cache", "controller", "evidence", "monitor")}
    fake = engine_with_fakes(**off)
    with TestClient(create_app(fake.engine)) as http:
        r = http.post("/v1/retrieve/direct", json={"query": GOAL, "mode": "hybrid", "rerank": True})
    assert r.status_code == 200 and r.json()["candidates"]
    assert len(fake.retriever.requests) == 1

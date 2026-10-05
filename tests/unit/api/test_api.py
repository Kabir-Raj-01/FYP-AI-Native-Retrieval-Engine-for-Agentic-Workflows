"""engine/api: /v1/health and /v1/retrieve/direct with an Engine built from fakes."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from engine.api.app import create_app
from engine.config import EngineConfig, config_hash
from engine.core import Engine
from engine.errors import TaskNotFound
from tests.fakes import CorpusChunk, FakeEmbedder, FakeReranker, FakeRetriever

URL = "/v1/retrieve/direct"


def kelp_chunk(corpus: list[CorpusChunk]) -> str:
    """The aquarium article's chunk about its kelp tank (fixture mini-d12)."""
    return next(c.chunk_id for c in corpus if "kelp tank" in c.text)


def client_for(engine: Engine) -> Iterator[TestClient]:
    with TestClient(create_app(engine), raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def retriever(mini_corpus: list[CorpusChunk]) -> FakeRetriever:
    return FakeRetriever(mini_corpus, embedder=FakeEmbedder())


@pytest.fixture
def client(cfg: EngineConfig, retriever: FakeRetriever) -> Iterator[TestClient]:
    yield from client_for(Engine(cfg, retriever=retriever))


@pytest.fixture
def rerank_client(
    cfg: EngineConfig, retriever: FakeRetriever, mini_corpus: list[CorpusChunk]
) -> Iterator[TestClient]:
    reranker = FakeReranker({kelp_chunk(mini_corpus): 0.95}, default=0.2, top_n=3)
    yield from client_for(Engine(cfg, retriever=retriever, reranker=reranker))


def assert_error(response: object, status: int, code: str) -> dict[str, object]:
    assert response.status_code == status  # type: ignore[attr-defined]
    body = response.json()  # type: ignore[attr-defined]
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "task_id"}
    assert body["error"]["code"] == code
    return body["error"]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------- health
def test_health_reports_backend_version_and_config_hash(
    client: TestClient, cfg: EngineConfig
) -> None:
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["qdrant"] is True
    assert body["version"]
    assert body["config_hash"] == config_hash(cfg)


def test_health_is_degraded_when_the_backend_is_down(
    cfg: EngineConfig, mini_corpus: list[CorpusChunk]
) -> None:
    for c in client_for(Engine(cfg, retriever=FakeRetriever(mini_corpus, healthy=False))):
        response = c.get("/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["qdrant"] is False


# --------------------------------------------------------------------- direct: success
@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_direct_works_for_every_mode(
    client: TestClient, retriever: FakeRetriever, mode: str
) -> None:
    response = client.post(URL, json={"query": "Linden Bay tram", "mode": mode, "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"candidates", "relevances", "latency_ms"}
    assert 1 <= len(body["candidates"]) <= 5
    assert body["relevances"] is None
    assert body["latency_ms"] >= 0
    assert retriever.requests[-1].mode.value == mode


def test_vectors_are_never_serialised(client: TestClient) -> None:
    body = client.post(URL, json={"query": "Linden Bay"}).json()
    assert body["candidates"]
    assert all(c["vector"] is None for c in body["candidates"])


def test_defaults_are_dense_and_default_k(
    client: TestClient, retriever: FakeRetriever, cfg: EngineConfig
) -> None:
    client.post(URL, json={"query": "Linden Bay"})
    request = retriever.requests[-1]
    assert (request.mode.value, request.k) == ("dense", cfg.retrieval.default_k)


def test_rerank_returns_aligned_relevances_in_relevance_order(
    rerank_client: TestClient, retriever: FakeRetriever, mini_corpus: list[CorpusChunk]
) -> None:
    query = "Linden Bay aquarium kelp tank"  # token overlap retrieves the kelp chunk
    response = rerank_client.post(URL, json={"query": query, "k": 20, "rerank": True})
    body = response.json()
    assert response.status_code == 200
    assert len(body["candidates"]) == len(body["relevances"]) == 3  # FakeReranker top_n
    assert body["relevances"] == sorted(body["relevances"], reverse=True)
    assert body["candidates"][0]["chunk_id"] == kelp_chunk(mini_corpus)
    assert body["relevances"][0] == 0.95


# ---------------------------------------------------------------- direct: validation
@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"query": ""}, "query"),
        ({"query": "x", "k": 0}, "k"),
        ({"query": "x", "mode": "fuzzy"}, "mode"),
        ({}, "query"),
    ],
)
def test_schema_violations_are_422_validation_error(
    client: TestClient, payload: dict[str, object], field: str
) -> None:
    error = assert_error(client.post(URL, json=payload), 422, "VALIDATION_ERROR")
    assert field in str(error["message"])


def test_k_above_max_k_is_422(
    client: TestClient, cfg: EngineConfig, retriever: FakeRetriever
) -> None:
    error = assert_error(
        client.post(URL, json={"query": "x", "k": cfg.retrieval.max_k + 1}), 422, "VALIDATION_ERROR"
    )
    assert "retrieval.max_k" in str(error["message"])
    assert retriever.requests == []


def test_k_equal_to_max_k_is_accepted(client: TestClient, cfg: EngineConfig) -> None:
    assert client.post(URL, json={"query": "x", "k": cfg.retrieval.max_k}).status_code == 200


def test_query_longer_than_api_limit_is_422(client: TestClient, cfg: EngineConfig) -> None:
    long_query = "x" * (cfg.api.query_max_chars + 1)
    error = assert_error(client.post(URL, json={"query": long_query}), 422, "VALIDATION_ERROR")
    assert "api.query_max_chars" in str(error["message"])
    ok = client.post(URL, json={"query": "x" * cfg.api.query_max_chars})
    assert ok.status_code == 200


def test_rerank_without_a_reranker_is_422(client: TestClient) -> None:
    error = assert_error(
        client.post(URL, json={"query": "x", "rerank": True}), 422, "VALIDATION_ERROR"
    )
    assert "rerank is not available" in str(error["message"])


# -------------------------------------------------------------------- error mapping
def test_backend_error_is_503(cfg: EngineConfig, mini_corpus: list[CorpusChunk]) -> None:
    failing = Engine(cfg, retriever=FakeRetriever(mini_corpus, fail_on_call=1))
    for c in client_for(failing):
        assert_error(c.post(URL, json={"query": "x"}), 503, "BACKEND_ERROR")


def test_engine_errors_keep_code_status_and_task_id(
    cfg: EngineConfig, retriever: FakeRetriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    eng = Engine(cfg, retriever=retriever)

    def boom(*args: object, **kwargs: object) -> None:
        raise TaskNotFound("no such task", task_id="t-9")

    monkeypatch.setattr(eng, "retrieve_direct", boom)
    for c in client_for(eng):
        error = assert_error(c.post(URL, json={"query": "x"}), 404, "TASK_NOT_FOUND")
        assert error["task_id"] == "t-9"


def test_unexpected_errors_are_500_without_internals(
    cfg: EngineConfig, retriever: FakeRetriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    eng = Engine(cfg, retriever=retriever)

    def crash(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(eng, "retrieve_direct", crash)
    for c in client_for(eng):
        error = assert_error(c.post(URL, json={"query": "x"}), 500, "INTERNAL_ERROR")
        assert "secret" not in str(error["message"])

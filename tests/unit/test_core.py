"""engine/core.py: Engine.retrieve_direct (ARCHITECTURE §6; ADR-012) with fakes."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from structlog.testing import capture_logs

from engine import factory
from engine.config import EngineConfig
from engine.core import Engine
from engine.errors import BackendError, ConfigError
from engine.state.models import DirectResult, RetrievalMode
from tests.fakes import CorpusChunk, FakeEmbedder, FakeReranker, FakeRetriever


def ticking(step_s: float = 0.05) -> Iterator[float]:
    t = 0.0
    while True:
        yield t
        t += step_s


@pytest.fixture
def retriever(mini_corpus: list[CorpusChunk]) -> FakeRetriever:
    return FakeRetriever(mini_corpus, embedder=FakeEmbedder())


def engine(cfg: EngineConfig, retriever: FakeRetriever, **kw: object) -> Engine:
    clock = ticking()
    return Engine(cfg, retriever=retriever, clock=lambda: next(clock), **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_direct_returns_candidates_and_latency(
    cfg: EngineConfig, retriever: FakeRetriever, mode: RetrievalMode
) -> None:
    result = engine(cfg, retriever).retrieve_direct("Linden Bay tram", mode, k=5)
    assert isinstance(result, DirectResult)
    assert 1 <= len(result.candidates) <= 5
    assert result.scored is None
    assert result.latency_ms == pytest.approx(50.0)  # one clock tick between start and end
    assert retriever.requests[-1].mode == mode


def test_k_defaults_to_retrieval_default_k(cfg: EngineConfig, retriever: FakeRetriever) -> None:
    engine(cfg, retriever).retrieve_direct("Linden Bay", RetrievalMode.DENSE)
    assert retriever.requests[-1].k == cfg.retrieval.default_k


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_rejected(cfg: EngineConfig, retriever: FakeRetriever, k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        engine(cfg, retriever).retrieve_direct("x", RetrievalMode.DENSE, k=k)


def test_k_up_to_max_k_accepted_and_above_rejected(
    cfg: EngineConfig, retriever: FakeRetriever
) -> None:
    eng = engine(cfg, retriever)
    eng.retrieve_direct("x", RetrievalMode.DENSE, k=cfg.retrieval.max_k)
    with pytest.raises(ValueError, match="k must be"):
        eng.retrieve_direct("x", RetrievalMode.DENSE, k=cfg.retrieval.max_k + 1)
    assert len(retriever.requests) == 1  # the rejected call never reached the backend


def test_rerank_uses_the_relevance_only_reranker(
    cfg: EngineConfig, retriever: FakeRetriever, mini_corpus: list[CorpusChunk]
) -> None:
    reranker = FakeReranker({mini_corpus[10].chunk_id: 0.9}, default=0.1)
    eng = engine(cfg, retriever, reranker=reranker)
    result = eng.retrieve_direct("Linden Bay", RetrievalMode.HYBRID, k=20, rerank=True)
    assert result.scored is not None
    assert [s.relevance for s in result.scored] == sorted(
        (s.relevance for s in result.scored), reverse=True
    )
    assert reranker.calls[0][2] is False  # state=None: relevance-only (ADR-009 "Also")
    assert eng.has_reranker
    assert reranker.goals == [None]  # one-shot: relevance against the query itself


def test_rerank_passes_the_goal_on(
    cfg: EngineConfig, retriever: FakeRetriever, mini_corpus: list[CorpusChunk]
) -> None:
    reranker = FakeReranker(default=0.1)
    eng = engine(cfg, retriever, reranker=reranker)
    eng.retrieve_direct("sub", RetrievalMode.DENSE, k=5, rerank=True, goal="the question")
    assert reranker.goals == ["the question"]  # ADR-014: fixed_iteration's hop queries


def test_rerank_without_reranker_is_a_config_error(
    cfg: EngineConfig, retriever: FakeRetriever
) -> None:
    eng = engine(cfg, retriever)
    assert not eng.has_reranker
    with pytest.raises(ConfigError, match="no reranker"):
        eng.retrieve_direct("x", RetrievalMode.DENSE, rerank=True)
    assert retriever.requests == []


def test_backend_errors_propagate(cfg: EngineConfig, mini_corpus: list[CorpusChunk]) -> None:
    eng = engine(cfg, FakeRetriever(mini_corpus, fail_on_call=1))
    with pytest.raises(BackendError):
        eng.retrieve_direct("x", RetrievalMode.DENSE)


def test_retrieval_call_event_is_logged(cfg: EngineConfig, retriever: FakeRetriever) -> None:
    with capture_logs() as logs:
        engine(cfg, retriever).retrieve_direct("Linden Bay", RetrievalMode.SPARSE, k=3)
    event = next(e for e in logs if e["event"] == "retrieval.call")
    assert (event["path"], event["mode"], event["k"], event["query"]) == (
        "direct",
        "sparse",
        3,
        "Linden Bay",
    )
    assert event["n_returned"] <= 3
    assert event["latency_ms"] == pytest.approx(50.0)


def test_retrieval_event_latency_excludes_rerank(
    cfg: EngineConfig, retriever: FakeRetriever
) -> None:
    eng = engine(cfg, retriever, reranker=FakeReranker(default=0.1))
    with capture_logs() as logs:
        result = eng.retrieve_direct("Linden Bay", RetrievalMode.DENSE, k=5, rerank=True)
    (event,) = [e for e in logs if e["event"] == "retrieval.call"]
    assert event["latency_ms"] == pytest.approx(50.0)  # one tick: the search
    assert result.latency_ms == pytest.approx(100.0)  # two ticks: search + rerank (ADR-012)


def test_health_delegates_to_the_retriever(
    cfg: EngineConfig, mini_corpus: list[CorpusChunk]
) -> None:
    assert engine(cfg, FakeRetriever(mini_corpus, healthy=True)).health() is True
    assert engine(cfg, FakeRetriever(mini_corpus, healthy=False)).health() is False


def test_from_config_builds_only_direct_path_components(
    cfg: EngineConfig, monkeypatch: pytest.MonkeyPatch, retriever: FakeRetriever
) -> None:
    built: list[str] = []
    monkeypatch.setattr(
        factory, "build_embedder", lambda c: built.append("embedder") or FakeEmbedder()
    )
    monkeypatch.setattr(factory, "build_sparse_encoder", lambda c: built.append("sparse") or None)
    monkeypatch.setattr(
        factory, "build_retriever", lambda c, e, s: built.append("retriever") or retriever
    )

    def no_planner(*args: object) -> None:
        raise AssertionError("the direct path must not build the planner or LLM (FR3.4)")

    monkeypatch.setattr(factory, "build_planner", no_planner)
    monkeypatch.setattr(factory, "build_llm_client", no_planner)
    eng = Engine.from_config(cfg)
    assert built == ["embedder", "sparse", "retriever"]
    assert eng.has_reranker  # built; the cross-encoder model loads on the first rerank
    assert eng.retrieve_direct("Linden Bay", RetrievalMode.DENSE, k=2).candidates

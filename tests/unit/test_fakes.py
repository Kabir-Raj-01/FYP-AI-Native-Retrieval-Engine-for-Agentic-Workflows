"""Fakes implement the real interfaces and behave as TESTING §2 specifies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from engine.embeddings.interfaces import Embedder
from engine.errors import BackendError, LLMError, LLMOutputError
from engine.llm.interfaces import LLMClient, LLMResult, Message
from engine.rerank.interfaces import Reranker
from engine.retrieval.interfaces import RetrieverAdapter
from engine.state.models import (
    Candidate,
    NeedType,
    PlanLLMOutput,
    RetrievalMode,
    ScoredCandidate,
    SubgoalPlan,
)
from tests.conftest import StateFactory
from tests.fakes import (
    CorpusChunk,
    FakeClock,
    FakeEmbedder,
    FakeLLMClient,
    FakeReranker,
    FakeRetriever,
)

CORPUS = [
    CorpusChunk("c1", "d1", "Harlan Dynamics acquired Voltrion in 2025", {"title": "Deal"}),
    CorpusChunk("c2", "d2", "Voltrion makes solid-state batteries", {"title": "Profile"}),
    CorpusChunk("c3", "d3", "The weather in Port Merrow was mild", {"title": "Weather"}),
]
PLAN_JSON = '{"subgoals": [{"description": "d", "query": "q", "need_type": "mixed"}]}'
MESSAGES = [Message(role="system", content="rules"), Message(role="user", content="goal")]


# ------------------------------------------------------------------------ conformance
@pytest.mark.parametrize(
    ("fake", "interface"),
    [
        (FakeEmbedder, Embedder),
        (FakeRetriever, RetrieverAdapter),
        (FakeReranker, Reranker),
        (FakeLLMClient, LLMClient),
    ],
)
def test_fake_implements_the_interface(fake: type, interface: type) -> None:
    assert issubclass(fake, interface)
    assert not getattr(fake, "__abstractmethods__", frozenset())


def test_fake_methods_return_the_interface_types(state_factory: StateFactory) -> None:
    embedder = FakeEmbedder(dimension=8)
    assert isinstance(embedder.dimension, int)
    for vectors in (embedder.embed_queries(["a", "b"]), embedder.embed_passages(["a"])):
        assert isinstance(vectors, np.ndarray)
        assert vectors.dtype == np.float32

    retriever = FakeRetriever(CORPUS, embedder=embedder)
    candidates = retriever.search("Voltrion", RetrievalMode.DENSE, k=5)
    assert candidates
    assert all(isinstance(c, Candidate) for c in candidates)
    assert isinstance(retriever.health(), bool)

    for state in (None, state_factory()):
        scored = FakeReranker({"c1": 0.9}).rerank("q", candidates, state)
        assert all(isinstance(s, ScoredCandidate) for s in scored)

    llm = FakeLLMClient({"planner": [PLAN_JSON], "generator": ["an answer"]})
    structured = llm.structured("planner", MESSAGES, PlanLLMOutput)
    assert isinstance(structured, LLMResult)
    assert isinstance(structured.value, PlanLLMOutput)
    text = llm.text("generator", MESSAGES)
    assert isinstance(text.value, str)


# --------------------------------------------------------------------------- embedder
def test_embedder_is_deterministic_and_unit_norm() -> None:
    a, b = FakeEmbedder(), FakeEmbedder()
    va = a.embed_queries(["same text"])[0]
    assert np.array_equal(va, b.embed_passages(["same text"])[0])
    assert np.linalg.norm(va) == pytest.approx(1.0, abs=1e-6)


def test_embedder_different_texts_are_near_orthogonal() -> None:
    vectors = FakeEmbedder(dimension=256).embed_queries([f"text {i}" for i in range(20)])
    sims = vectors @ vectors.T
    off_diagonal = sims[~np.eye(len(sims), dtype=bool)]
    assert np.max(np.abs(off_diagonal)) < 0.35


def test_embedder_pinned_vectors_control_similarity() -> None:
    embedder = FakeEmbedder(dimension=2, vectors={"x": [1.0, 0.0], "y": [3.0, 3.0]})
    x, y = embedder.embed_queries(["x", "y"])
    assert float(x @ y) == pytest.approx(np.sqrt(0.5), abs=1e-6)


def test_embedder_rejects_bad_pinned_vectors() -> None:
    with pytest.raises(ValueError, match="shape"):
        FakeEmbedder(dimension=3, vectors={"x": [1.0, 0.0]})
    with pytest.raises(ValueError, match="zero"):
        FakeEmbedder(dimension=2, vectors={"x": [0.0, 0.0]})


def test_embedder_empty_input_and_call_log() -> None:
    embedder = FakeEmbedder(dimension=4)
    assert embedder.embed_queries([]).shape == (0, 4)
    embedder.embed_passages(["p"])
    assert embedder.calls == [("queries", []), ("passages", ["p"])]


# -------------------------------------------------------------------------- retriever
def test_retriever_token_overlap_ranks_and_limits() -> None:
    retriever = FakeRetriever(CORPUS)
    results = retriever.search("Voltrion batteries", RetrievalMode.SPARSE, k=1)
    assert [c.chunk_id for c in results] == ["c2"]
    assert results[0].rank == 0
    assert results[0].mode == RetrievalMode.SPARSE
    assert results[0].vector is None  # no embedder given


def test_retriever_drops_zero_score_chunks() -> None:
    assert FakeRetriever(CORPUS).search("unrelated words", RetrievalMode.DENSE, k=5) == []
    assert FakeRetriever(CORPUS).search("", RetrievalMode.DENSE, k=5) == []


def test_retriever_scripted_results_per_query_and_mode() -> None:
    retriever = FakeRetriever(CORPUS, scripted={("q", RetrievalMode.HYBRID): ["c3", "c1"]})
    assert [c.chunk_id for c in retriever.search("q", RetrievalMode.HYBRID, k=5)] == ["c3", "c1"]
    assert retriever.search("q", RetrievalMode.DENSE, k=5) == []  # unscripted mode


def test_retriever_vectors_come_from_the_embedder() -> None:
    embedder = FakeEmbedder(dimension=8)
    result = FakeRetriever(CORPUS, embedder=embedder).search("Voltrion", RetrievalMode.DENSE, 5)
    expected = embedder.embed_passages([result[0].text])[0]
    assert result[0].vector == pytest.approx(expected.tolist())
    no_vectors = FakeRetriever(CORPUS, embedder=embedder).search(
        "Voltrion", RetrievalMode.DENSE, 5, with_vectors=False
    )
    assert all(c.vector is None for c in no_vectors)


def test_retriever_applies_filters() -> None:
    retriever = FakeRetriever(CORPUS)
    results = retriever.search("Voltrion", RetrievalMode.DENSE, 5, filters={"doc_id": "d1"})
    assert [c.chunk_id for c in results] == ["c1"]


def test_retriever_raises_backend_error_on_the_nth_call() -> None:
    retriever = FakeRetriever(CORPUS, fail_on_call=2)
    retriever.search("Voltrion", RetrievalMode.DENSE, 5)
    with pytest.raises(BackendError):
        retriever.search("Voltrion", RetrievalMode.DENSE, 5)
    retriever.search("Voltrion", RetrievalMode.DENSE, 5)
    assert len(retriever.requests) == 3
    assert retriever.requests[0].k == 5


def test_retriever_health_is_configurable() -> None:
    assert FakeRetriever(CORPUS, healthy=False).health() is False


# --------------------------------------------------------------------------- reranker
def test_reranker_orders_by_mapped_relevance_and_truncates() -> None:
    candidates = FakeRetriever(CORPUS).search("Voltrion Port", RetrievalMode.DENSE, 5)
    reranker = FakeReranker({"c1": 0.2, "c2": 0.9}, default=0.5, top_n=2)
    scored = reranker.rerank("q", candidates, None)
    assert [s.candidate.chunk_id for s in scored] == ["c2", "c3"]
    assert [s.final_score for s in scored] == [s.relevance for s in scored]
    assert reranker.calls[0][2] is False


def test_reranker_empty_input_gives_empty_output() -> None:
    assert FakeReranker().rerank("q", [], None) == []


# -------------------------------------------------------------------------------- llm
def test_llm_accepts_model_dict_and_json_responses() -> None:
    plan = PlanLLMOutput(subgoals=[SubgoalPlan(description="d", query="q", need_type="mixed")])
    llm = FakeLLMClient()
    llm.enqueue("planner", plan, plan.model_dump(), PLAN_JSON)
    values = [llm.structured("planner", MESSAGES, PlanLLMOutput).value for _ in range(3)]
    assert all(v.subgoals[0].need_type == NeedType.MIXED for v in values)


@pytest.mark.parametrize("bad", ["not json", '{"subgoals": []}', {"subgoals": "x"}, 42])
def test_llm_invalid_response_raises_output_error(bad: object) -> None:
    llm = FakeLLMClient({"planner": [bad]})  # type: ignore[list-item]
    with pytest.raises(LLMOutputError):
        llm.structured("planner", MESSAGES, PlanLLMOutput)


def test_llm_scripted_exception_is_raised() -> None:
    llm = FakeLLMClient({"planner": [LLMError("timeout"), PLAN_JSON]})
    with pytest.raises(LLMError, match="timeout"):
        llm.structured("planner", MESSAGES, PlanLLMOutput)
    assert llm.structured("planner", MESSAGES, PlanLLMOutput).value.subgoals


def test_llm_records_requests_and_reports_token_usage() -> None:
    llm = FakeLLMClient({"planner": [PLAN_JSON]}, input_tokens=120, output_tokens=30, model="m")
    result = llm.structured("planner", MESSAGES, PlanLLMOutput)
    assert (result.input_tokens, result.output_tokens, result.model) == (120, 30, "m")
    assert llm.requests[0].role == "planner"
    assert llm.requests[0].schema is PlanLLMOutput
    assert llm.requests[0].messages == MESSAGES


def test_llm_queues_are_per_role_and_exhaustion_fails_loudly() -> None:
    llm = FakeLLMClient({"planner": [PLAN_JSON]})
    with pytest.raises(AssertionError, match="verifier"):
        llm.structured("verifier", MESSAGES, PlanLLMOutput)
    llm.structured("planner", MESSAGES, PlanLLMOutput)
    with pytest.raises(AssertionError, match="planner"):
        llm.structured("planner", MESSAGES, PlanLLMOutput)


def test_llm_text_needs_a_string_response() -> None:
    llm = FakeLLMClient({"generator": [PLAN_JSON, {"not": "text"}]})
    assert llm.text("generator", MESSAGES).value == PLAN_JSON
    with pytest.raises(TypeError):
        llm.text("generator", MESSAGES)
    assert llm.requests[-1].schema is None


# ------------------------------------------------------------------------------ clock
def test_clock_is_controllable() -> None:
    clock = FakeClock()
    start, t0 = clock.now(), clock.monotonic_ms()
    assert start.tzinfo is not None
    clock.advance(250)
    assert clock.monotonic_ms() - t0 == 250
    assert clock.now() - start == timedelta(milliseconds=250)


def test_clock_converts_to_utc_and_rejects_bad_input() -> None:
    plus_five = datetime(2026, 10, 4, 17, 0).astimezone()  # local, aware
    assert FakeClock(plus_five).now().tzinfo == UTC
    with pytest.raises(ValueError, match="timezone"):
        FakeClock(datetime(2026, 10, 4))
    with pytest.raises(ValueError, match="backwards"):
        FakeClock().advance(-1)

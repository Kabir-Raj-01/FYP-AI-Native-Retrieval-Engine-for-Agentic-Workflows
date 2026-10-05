"""Module interfaces: ABCs cannot be instantiated and match the documented signatures.

EXPECTED is written from ARCHITECTURE §2 and the module specs (RETRIEVAL_SPEC §1/§5,
RERANK_SPEC §1, EVIDENCE_MANAGER_SPEC §1, PLANNER_SPEC §1, CACHE_SPEC §3, LLM_CLIENT_SPEC
§3) with the Week 7 plan's approved changes (TaskStore.create task_id, typed filters,
Optional update in observe). A change to any signature must change this table and the doc.
"""

from __future__ import annotations

import inspect
import typing
from abc import ABC
from pathlib import Path

import pytest

from engine import factory
from engine.cache.interfaces import Cache
from engine.config import load_config
from engine.controller.interfaces import Controller
from engine.embeddings.interfaces import Embedder, SparseEncoder
from engine.evidence.interfaces import EvidenceManager
from engine.llm.interfaces import LLMClient, LLMResult, Message
from engine.monitor.interfaces import TraceWriter, TrajectoryMonitor
from engine.planner.interfaces import Planner
from engine.rerank.interfaces import Reranker
from engine.retrieval.interfaces import IndexWriter, RetrieverAdapter
from engine.state.interfaces import TaskStore
from engine.state.models import RunContext

EXPECTED: dict[type[ABC], dict[str, str]] = {
    TaskStore: {
        "create": "(self, goal: str, budget: Budget, task_id: str | None = None) -> RetrievalState",
        "load": "(self, task_id: str) -> RetrievalState",
        "save": "(self, state: RetrievalState, expected_version: int,"
        " cache_entries: Sequence[CacheEntry] = ()) -> None",
    },
    Planner: {"plan": "(self, state: RetrievalState, mode: PlanMode) -> PlanOutcome"},
    Cache: {
        "lookup": "(self, task_id: str, query: str, need_type: NeedType) -> CacheHit | None",
        "put": "(self, entry: CacheEntry) -> None",
    },
    Controller: {
        "select_action": "(self, state: RetrievalState, cache_hit: CacheHit | None)"
        " -> ActionDecision",
    },
    RetrieverAdapter: {
        "search": "(self, query: str, mode: RetrievalMode, k: int,"
        " filters: SearchFilters | None = None, with_vectors: bool = True) -> list[Candidate]",
        "health": "(self) -> bool",
    },
    Reranker: {
        "rerank": "(self, query: str, candidates: list[Candidate],"
        " state: RetrievalState | None) -> list[ScoredCandidate]",
    },
    EvidenceManager: {
        "update": "(self, state: RetrievalState, scored: list[ScoredCandidate], step: int)"
        " -> EvidenceUpdate",
    },
    TrajectoryMonitor: {
        "observe": "(self, state: RetrievalState, decision: ActionDecision,"
        " update: EvidenceUpdate | None) -> MonitorVerdict",
        "advance": "(self, state: RetrievalState) -> AdvanceResult",
    },
    TraceWriter: {
        "context": "property -> RunContext",
        "write_step": "(self, step: TraceStep) -> None",
        "write_bundle": "(self, bundle: EvidenceBundle) -> None",
        "read_steps": "(self, task_id: str) -> list[TraceStep]",
        "read_bundle": "(self, task_id: str) -> EvidenceBundle | None",
    },
    Embedder: {
        "dimension": "property -> int",
        "embed_queries": "(self, texts: list[str]) -> Vectors",
        "embed_passages": "(self, texts: list[str]) -> Vectors",
    },
    SparseEncoder: {
        "encode_documents": "(self, texts: list[str]) -> list[SparseVector]",
        "encode_queries": "(self, texts: list[str]) -> list[SparseVector]",
    },
    IndexWriter: {
        "ensure_collection": "(self) -> None",
        "existing_ids": "(self) -> set[str]",
        "upsert": "(self, points: Sequence[IndexPoint]) -> None",
    },
    LLMClient: {
        "structured": "(self, role: str, messages: list[Message], schema: type[SchemaT])"
        " -> LLMResult[SchemaT]",
        "text": "(self, role: str, messages: list[Message]) -> LLMResult[str]",
    },
}


def describe(cls: type, name: str) -> str:
    attr = inspect.getattr_static(cls, name)
    if isinstance(attr, property):
        assert attr.fget is not None
        return f"property -> {attr.fget.__annotations__['return']}"
    # The modules use `from __future__ import annotations`, so annotations are the source
    # strings; drop the quotes inspect adds around them.
    return str(inspect.signature(attr)).replace("'", "")


@pytest.mark.parametrize("cls", list(EXPECTED), ids=lambda c: c.__name__)
def test_abstract_methods_are_exactly_the_documented_ones(cls: type[ABC]) -> None:
    assert cls.__abstractmethods__ == frozenset(EXPECTED[cls])


@pytest.mark.parametrize(
    ("cls", "name"),
    [(cls, name) for cls, methods in EXPECTED.items() for name in methods],
    ids=lambda x: x if isinstance(x, str) else x.__name__,
)
def test_signature_matches_documentation(cls: type[ABC], name: str) -> None:
    assert describe(cls, name) == EXPECTED[cls][name]


@pytest.mark.parametrize("cls", list(EXPECTED), ids=lambda c: c.__name__)
def test_annotations_resolve(cls: type[ABC]) -> None:
    for name in EXPECTED[cls]:
        attr = inspect.getattr_static(cls, name)
        target = attr.fget if isinstance(attr, property) else attr
        typing.get_type_hints(target)  # raises NameError on an undefined annotation


@pytest.mark.parametrize("cls", list(EXPECTED), ids=lambda c: c.__name__)
def test_abc_cannot_be_instantiated(cls: type[ABC]) -> None:
    with pytest.raises(TypeError, match="abstract"):
        cls()


@pytest.mark.parametrize("cls", list(EXPECTED), ids=lambda c: c.__name__)
def test_every_abstract_method_has_a_docstring(cls: type[ABC]) -> None:
    for name in EXPECTED[cls]:
        assert inspect.getdoc(inspect.getattr_static(cls, name))


def test_llm_transport_models_forbid_extra_fields() -> None:
    message = Message(role="user", content="hi")
    result = LLMResult[str](
        value="ok",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        cached=False,
        latency_ms=1.0,
        model="m",
    )
    assert result.value == "ok"
    with pytest.raises(ValueError, match="extra"):
        Message.model_validate({**message.model_dump(), "name": "x"})
    with pytest.raises(ValueError, match="role"):
        Message.model_validate({"role": "tool", "content": "x"})


BUILDERS = [getattr(factory, name) for name in factory.__all__ if name.startswith("build_")]


def test_factory_has_a_builder_per_interface() -> None:
    returned = {typing.get_type_hints(b)["return"] for b in BUILDERS}
    assert returned == set(EXPECTED)


def test_implemented_builders_return_their_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.embeddings import bm25, sentence_transformer
    from tests.unit.embeddings.test_wrappers import FakeFastEmbed, FakeSTModel

    cfg = load_config(environ={})
    monkeypatch.setattr(
        sentence_transformer, "_load_model", lambda c: FakeSTModel(dim=cfg.embedding.dimension)
    )
    monkeypatch.setattr(bm25, "_load_model", lambda c, d: FakeFastEmbed())
    embedder = factory.build_embedder(cfg)
    sparse = factory.build_sparse_encoder(cfg)
    assert isinstance(embedder, Embedder)
    assert isinstance(sparse, SparseEncoder)
    # Qdrant clients connect lazily, so building them needs no running server.
    assert isinstance(factory.build_retriever(cfg, embedder, sparse), RetrieverAdapter)
    assert isinstance(factory.build_index_writer(cfg), IndexWriter)
    assert isinstance(factory.build_cache(cfg, embedder), Cache)  # connects lazily
    assert isinstance(factory.build_task_store(cfg), TaskStore)  # connects lazily
    assert isinstance(factory.build_reranker(cfg, embedder), Reranker)  # loads lazily
    assert isinstance(factory.build_controller(cfg, embedder), Controller)
    assert isinstance(factory.build_evidence_manager(cfg, embedder, None), EvidenceManager)  # type: ignore[arg-type]
    assert isinstance(factory.build_monitor(cfg), TrajectoryMonitor)
    llm = factory.build_llm_client(cfg)
    assert isinstance(llm, LLMClient)  # no provider built yet
    assert isinstance(factory.build_planner(cfg, llm), Planner)
    ctx = RunContext(run_id="r", system="controller_on")
    assert isinstance(factory.build_trace_writer(cfg, ctx, Path("unused")), TraceWriter)

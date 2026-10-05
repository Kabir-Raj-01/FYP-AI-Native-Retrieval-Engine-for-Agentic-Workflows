"""RETRIEVAL_SPEC §6 integration tests: real BGE embedder + BM25 + local qdrant.exe.

Needs a running Qdrant at ``qdrant.url`` (SETUP §3) and the pinned models (downloaded on
first use). Indexes ``tests/fixtures/mini_corpus.jsonl`` into a throwaway collection.
Expected top-1 results were measured on 2026-10-05 (bge-base-en-v1.5 @ a5beb1e3, BM25 @
22b8d2af, Qdrant 1.19.1) and are unambiguous for these queries.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import numpy as np
import pytest
from qdrant_client import QdrantClient

from engine import factory
from engine.config import EngineConfig, load_config
from engine.embeddings.interfaces import Embedder
from engine.errors import BackendError
from engine.retrieval.interfaces import IndexPoint, RetrieverAdapter
from engine.state.models import RetrievalMode
from tests.fixture_data import FIXTURES_DIR

pytestmark = pytest.mark.integration

GREENFIELD_CHUNK = "b82699b39d909769"  # mini-d04 chunk 0: the gold chunk of mini:q2, hop 1
KX40_CHUNK = "c113c0baaee4eea7"  # mini-d03, the only chunk mentioning the KX-40
BRIGHTWING_CHUNK = "0888fc2c6f083709"  # mini-d08 chunk 0


def _cfg() -> EngineConfig:
    cfg = load_config(environ={k: v for k, v in os.environ.items() if k == "QDRANT_URL"})
    cfg.qdrant.collection = f"it_mini_{os.getpid()}"
    return cfg


@pytest.fixture(scope="module")
def live() -> Iterator[tuple[EngineConfig, RetrieverAdapter, Embedder]]:
    cfg = _cfg()
    client = QdrantClient(url=cfg.qdrant.url, timeout=cfg.qdrant.timeout_s)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant not reachable at {cfg.qdrant.url}: {exc}")
    embedder = factory.build_embedder(cfg)
    sparse = factory.build_sparse_encoder(cfg)
    writer = factory.build_index_writer(cfg)
    with (FIXTURES_DIR / "mini_corpus.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    texts = [r["text"] for r in rows]
    writer.ensure_collection()
    writer.upsert(
        [
            IndexPoint(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                text=r["text"],
                metadata=r["metadata"],
                dense=d.tolist(),
                sparse=s,
            )
            for r, d, s in zip(
                rows, embedder.embed_passages(texts), sparse.encode_documents(texts), strict=True
            )
        ]
    )
    try:
        yield cfg, factory.build_retriever(cfg, embedder, sparse), embedder
    finally:
        client.delete_collection(cfg.qdrant.collection)


def test_semantic_query_dense_top1(live: tuple[EngineConfig, RetrieverAdapter, Embedder]) -> None:
    _, adapter, _ = live
    top = adapter.search(
        "Which farming group won an award for using less water?", RetrievalMode.DENSE, k=3
    )
    assert top[0].chunk_id == GREENFIELD_CHUNK


def test_exact_entity_query_sparse_top1(
    live: tuple[EngineConfig, RetrieverAdapter, Embedder],
) -> None:
    _, adapter, _ = live
    top = adapter.search("KX-40 battery fault", RetrievalMode.SPARSE, k=3)
    assert top[0].chunk_id == KX40_CHUNK


def test_mixed_query_hybrid_top1(live: tuple[EngineConfig, RetrieverAdapter, Embedder]) -> None:
    _, adapter, _ = live
    top = adapter.search("When did Brightwing Air start flying?", RetrievalMode.HYBRID, k=3)
    assert top[0].chunk_id == BRIGHTWING_CHUNK


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_payload_and_dense_vector_round_trip(
    live: tuple[EngineConfig, RetrieverAdapter, Embedder], mode: RetrievalMode
) -> None:
    cfg, adapter, embedder = live
    top = adapter.search("Calder Freight cargo partnership", mode, k=2)[0]
    assert top.metadata["title"]
    assert top.vector is not None and len(top.vector) == cfg.embedding.dimension
    stored = np.asarray(top.vector)
    assert float(stored @ embedder.embed_passages([top.text])[0]) == pytest.approx(1.0, abs=1e-4)


def test_health(live: tuple[EngineConfig, RetrieverAdapter, Embedder]) -> None:
    assert live[1].health() is True


def test_unreachable_backend_raises_backend_error(
    live: tuple[EngineConfig, RetrieverAdapter, Embedder],
) -> None:
    cfg, _, embedder = live
    broken = cfg.model_copy(deep=True)
    broken.qdrant.url = "http://127.0.0.1:1"  # nothing listens on port 1
    broken.qdrant.timeout_s = 2
    adapter = factory.build_retriever(broken, embedder, factory.build_sparse_encoder(broken))
    with pytest.raises(BackendError):
        adapter.search("anything", RetrievalMode.DENSE, k=1)
    assert adapter.health() is False

"""QdrantIndexWriter + QdrantAdapter against Qdrant's in-memory client (no server, no models).

In-memory mode searches exactly (HNSW parameters have no effect) but runs the same client
API, including named dense/sparse vectors with IDF, filters and RRF fusion.
"""

from __future__ import annotations

import warnings
from typing import Any
from uuid import uuid5

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from data.ids import point_id_for
from engine.config import EngineConfig
from engine.errors import BackendError, ConfigError
from engine.retrieval.interfaces import IndexPoint
from engine.retrieval.qdrant_adapter import QdrantAdapter
from engine.retrieval.qdrant_index import QdrantIndexWriter, point_id
from engine.state.models import RetrievalMode
from tests.fakes import CorpusChunk, FakeEmbedder, FakeSparseEncoder

pytestmark = pytest.mark.filterwarnings(
    "ignore:Local mode:UserWarning", "ignore:Payload indexes:UserWarning"
)
DIM = 16


@pytest.fixture
def q_cfg(cfg: EngineConfig) -> EngineConfig:
    cfg.embedding.dimension = DIM
    cfg.qdrant.collection = "unit_test"
    cfg.qdrant.upsert_batch_size = 2  # exercise batching
    return cfg


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def points(chunks: list[CorpusChunk], embedder: FakeEmbedder) -> list[IndexPoint]:
    texts = [c.text for c in chunks]
    dense = embedder.embed_passages(texts)
    sparse = FakeSparseEncoder().encode_documents(texts)
    return [
        IndexPoint(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            text=c.text,
            metadata=dict(c.metadata),
            dense=[float(x) for x in d],
            sparse=s,
        )
        for c, d, s in zip(chunks, dense, sparse, strict=True)
    ]


@pytest.fixture
def indexed(
    q_cfg: EngineConfig, client: QdrantClient, mini_corpus: list[CorpusChunk]
) -> tuple[QdrantAdapter, FakeEmbedder]:
    embedder = FakeEmbedder(dimension=DIM)
    writer = QdrantIndexWriter(q_cfg, client)
    writer.ensure_collection()
    writer.upsert(points(mini_corpus, embedder))
    return QdrantAdapter(q_cfg, embedder, FakeSparseEncoder(), client), embedder


# ------------------------------------------------------------------------- index writer
def test_collection_is_created_with_named_dense_and_idf_sparse_vectors(
    q_cfg: EngineConfig, client: QdrantClient
) -> None:
    QdrantIndexWriter(q_cfg, client).ensure_collection()
    params = client.get_collection("unit_test").config.params
    assert isinstance(params.vectors, dict)
    dense = params.vectors[q_cfg.qdrant.dense_vector_name]
    assert (dense.size, dense.distance) == (DIM, models.Distance.COSINE)
    assert params.sparse_vectors is not None
    assert params.sparse_vectors[q_cfg.qdrant.sparse_vector_name].modifier == models.Modifier.IDF


def test_ensure_collection_is_idempotent(q_cfg: EngineConfig, client: QdrantClient) -> None:
    writer = QdrantIndexWriter(q_cfg, client)
    writer.ensure_collection()
    writer.ensure_collection()  # existing and compatible: no error, no re-creation


@pytest.mark.parametrize("problem", ["dimension", "no_idf"])
def test_incompatible_existing_collection_is_refused(
    q_cfg: EngineConfig, client: QdrantClient, problem: str
) -> None:
    size = DIM + 1 if problem == "dimension" else DIM
    modifier = None if problem == "no_idf" else models.Modifier.IDF
    client.create_collection(
        "unit_test",
        vectors_config={"dense": models.VectorParams(size=size, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=modifier)},
    )
    with pytest.raises(ConfigError, match=r"new qdrant\.collection name"):
        QdrantIndexWriter(q_cfg, client).ensure_collection()


def test_upsert_is_idempotent_and_ids_are_uuid5_of_chunk_id(
    q_cfg: EngineConfig, client: QdrantClient, mini_corpus: list[CorpusChunk]
) -> None:
    writer = QdrantIndexWriter(q_cfg, client)
    writer.ensure_collection()
    pts = points(mini_corpus, FakeEmbedder(dimension=DIM))
    writer.upsert(pts)
    writer.upsert(pts)  # same point ids: replaced, not duplicated
    assert client.count("unit_test", exact=True).count == len(mini_corpus)
    assert writer.existing_ids() == {c.chunk_id for c in mini_corpus}
    first = mini_corpus[0].chunk_id
    assert point_id(q_cfg, first) == str(point_id_for(first, q_cfg.qdrant.id_namespace))
    record = client.retrieve("unit_test", ids=[point_id(q_cfg, first)], with_payload=True)[0]
    assert record.payload is not None
    assert record.payload["chunk_id"] == first
    assert record.payload["title"] == mini_corpus[0].metadata["title"]


def test_existing_ids_of_an_empty_collection(q_cfg: EngineConfig, client: QdrantClient) -> None:
    writer = QdrantIndexWriter(q_cfg, client)
    writer.ensure_collection()
    assert writer.existing_ids() == set()


# ------------------------------------------------------------------------------ adapter
@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_search_returns_ranked_candidates_with_payload_and_vectors(
    indexed: tuple[QdrantAdapter, FakeEmbedder], mode: RetrievalMode
) -> None:
    adapter, _ = indexed
    results = adapter.search("KX-41 inspection drone Talvane", mode, k=5)
    assert 1 <= len(results) <= 5
    assert [c.rank for c in results] == list(range(len(results)))
    assert all(c.mode == mode for c in results)
    assert all(c.vector is not None and len(c.vector) == DIM for c in results)
    assert {"title", "source", "published_at"} <= results[0].metadata.keys()
    assert "text" not in results[0].metadata


def test_dense_search_finds_the_identical_text_first(
    indexed: tuple[QdrantAdapter, FakeEmbedder], mini_corpus: list[CorpusChunk]
) -> None:
    adapter, _ = indexed
    target = mini_corpus[3]  # FakeEmbedder: identical text -> identical vector
    assert adapter.search(target.text, RetrievalMode.DENSE, k=3)[0].chunk_id == target.chunk_id


def test_sparse_search_matches_exact_terms(
    indexed: tuple[QdrantAdapter, FakeEmbedder], mini_corpus: list[CorpusChunk]
) -> None:
    adapter, _ = indexed
    top = adapter.search("Brightwing", RetrievalMode.SPARSE, k=3)
    assert top
    assert all("Brightwing" in c.text for c in top)


def test_with_vectors_false_and_k_limit(indexed: tuple[QdrantAdapter, FakeEmbedder]) -> None:
    adapter, _ = indexed
    results = adapter.search("Linden Bay", RetrievalMode.HYBRID, k=2, with_vectors=False)
    assert len(results) == 2
    assert all(c.vector is None for c in results)


def test_filters_restrict_results(
    indexed: tuple[QdrantAdapter, FakeEmbedder], mini_corpus: list[CorpusChunk]
) -> None:
    adapter, _ = indexed
    doc = mini_corpus[0].doc_id
    for mode in RetrievalMode:
        results = adapter.search("drone", mode, k=10, filters={"doc_id": doc})
        assert results
        assert {c.doc_id for c in results} == {doc}


def test_k_below_one_is_rejected(indexed: tuple[QdrantAdapter, FakeEmbedder]) -> None:
    with pytest.raises(ValueError, match="k must be"):
        indexed[0].search("x", RetrievalMode.DENSE, k=0)


def test_health(q_cfg: EngineConfig, client: QdrantClient) -> None:
    adapter = QdrantAdapter(q_cfg, FakeEmbedder(dimension=DIM), FakeSparseEncoder(), client)
    assert adapter.health() is False  # collection missing
    QdrantIndexWriter(q_cfg, client).ensure_collection()
    assert adapter.health() is True


class _FailingClient:
    def __getattr__(self, name: str) -> Any:
        def fail(*args: object, **kwargs: object) -> None:
            raise ResponseHandlingException(ConnectionError("connection refused"))

        return fail


def test_client_failures_become_backend_errors(q_cfg: EngineConfig) -> None:
    broken: Any = _FailingClient()
    adapter = QdrantAdapter(q_cfg, FakeEmbedder(dimension=DIM), FakeSparseEncoder(), broken)
    with pytest.raises(BackendError, match="dense search failed"):
        adapter.search("x", RetrievalMode.DENSE, k=1)
    assert adapter.health() is False
    with pytest.raises(BackendError, match="collection check"):
        QdrantIndexWriter(q_cfg, broken).ensure_collection()


def test_malformed_payload_is_a_backend_error(q_cfg: EngineConfig, client: QdrantClient) -> None:
    QdrantIndexWriter(q_cfg, client).ensure_collection()
    embedder = FakeEmbedder(dimension=DIM)
    client.upsert(
        "unit_test",
        points=[
            models.PointStruct(
                id=str(uuid5(q_cfg.qdrant.id_namespace, "bad")),
                vector={"dense": embedder.embed_passages(["x"])[0].tolist()},
                payload={"text": "no ids here"},
            )
        ],
    )
    adapter = QdrantAdapter(q_cfg, embedder, FakeSparseEncoder(), client)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BackendError, match="payload"):
            adapter.search("x", RetrievalMode.DENSE, k=1)

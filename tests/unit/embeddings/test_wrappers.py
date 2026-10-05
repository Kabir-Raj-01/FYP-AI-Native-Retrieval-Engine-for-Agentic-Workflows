"""SentenceTransformerEmbedder and FastEmbedBM25Encoder with injected fake models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from engine.config import EngineConfig
from engine.embeddings.bm25 import FastEmbedBM25Encoder
from engine.embeddings.interfaces import Embedder, SparseEncoder, SparseVector
from engine.embeddings.sentence_transformer import SentenceTransformerEmbedder
from engine.errors import ConfigError

DIM = 4


class FakeSTModel:
    """Records encode() calls; returns un-normalised rows derived from text length."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def get_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        self.calls.append((list(sentences), kwargs))
        return np.array([[len(s), 1.0, 0.0, 0.0] for s in sentences], dtype=np.float64)


@pytest.fixture
def st_cfg(cfg: EngineConfig) -> EngineConfig:
    cfg.embedding.dimension = DIM
    cfg.embedding.lru_size = 2
    return cfg


def test_embedder_applies_query_prefix_and_requests_normalised_vectors(
    st_cfg: EngineConfig,
) -> None:
    model = FakeSTModel()
    embedder = SentenceTransformerEmbedder(st_cfg.embedding, model)
    out = embedder.embed_queries(["q1"])
    texts, kwargs = model.calls[0]
    assert texts == [st_cfg.embedding.query_prefix + "q1"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == st_cfg.embedding.batch_size
    assert out.dtype == np.float32
    assert out.shape == (1, DIM)
    assert isinstance(embedder, Embedder)
    assert embedder.dimension == DIM


def test_embedder_applies_passage_prefix_without_caching(st_cfg: EngineConfig) -> None:
    st_cfg.embedding.passage_prefix = "passage: "
    model = FakeSTModel()
    embedder = SentenceTransformerEmbedder(st_cfg.embedding, model)
    embedder.embed_passages(["p"])
    embedder.embed_passages(["p"])
    assert [c[0] for c in model.calls] == [["passage: p"], ["passage: p"]]


def test_query_lru_avoids_reencoding_and_evicts_oldest(st_cfg: EngineConfig) -> None:
    model = FakeSTModel()
    embedder = SentenceTransformerEmbedder(st_cfg.embedding, model)
    first = embedder.embed_queries(["a", "b", "a"])  # duplicates encoded once
    assert len(model.calls[0][0]) == 2
    assert np.array_equal(first[0], first[2])
    embedder.embed_queries(["a"])  # cached
    assert len(model.calls) == 1
    embedder.embed_queries(["c"])  # lru_size 2: evicts "b" (least recently used)
    embedder.embed_queries(["b"])
    assert [c[0] for c in model.calls][-1] == [st_cfg.embedding.query_prefix + "b"]


def test_more_distinct_queries_than_the_lru_holds(st_cfg: EngineConfig) -> None:
    # regression: rows must come from this call, not from an LRU that already evicted them
    model = FakeSTModel()
    embedder = SentenceTransformerEmbedder(st_cfg.embedding, model)  # lru_size 2
    out = embedder.embed_queries(["a", "bb", "ccc", "a"])
    assert out.shape == (4, DIM)
    assert np.array_equal(out[0], out[3])
    assert not np.array_equal(out[0], out[2])


def test_lru_size_zero_disables_the_cache(st_cfg: EngineConfig) -> None:
    st_cfg.embedding.lru_size = 0
    model = FakeSTModel()
    embedder = SentenceTransformerEmbedder(st_cfg.embedding, model)
    embedder.embed_queries(["a"])
    embedder.embed_queries(["a"])
    assert len(model.calls) == 2


def test_empty_inputs_give_empty_arrays_without_model_calls(st_cfg: EngineConfig) -> None:
    model = FakeSTModel()
    embedder = SentenceTransformerEmbedder(st_cfg.embedding, model)
    assert embedder.embed_queries([]).shape == (0, DIM)
    assert embedder.embed_passages([]).shape == (0, DIM)
    assert model.calls == []


def test_dimension_mismatch_is_a_config_error(st_cfg: EngineConfig) -> None:
    with pytest.raises(ConfigError, match=r"embedding\.dimension"):
        SentenceTransformerEmbedder(st_cfg.embedding, FakeSTModel(dim=DIM + 1))


@dataclass
class _SparseEmbedding:
    indices: np.ndarray
    values: np.ndarray


class FakeFastEmbed:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, documents: list[str], **kwargs: Any) -> Any:
        self.calls.append("embed")
        return iter(
            _SparseEmbedding(np.array([1, 2], np.int64), np.array([0.5, 1.5])) for _ in documents
        )

    def query_embed(self, query: list[str], **kwargs: Any) -> Any:
        self.calls.append("query_embed")
        return iter(_SparseEmbedding(np.array([7], np.int64), np.array([1.0])) for _ in query)


def test_bm25_uses_document_and_query_encodings(cfg: EngineConfig, tmp_path: Path) -> None:
    model = FakeFastEmbed()
    encoder = FastEmbedBM25Encoder(cfg.sparse, tmp_path, model)
    docs = encoder.encode_documents(["a", "b"])
    queries = encoder.encode_queries(["q"])
    assert docs == [SparseVector(indices=[1, 2], values=[0.5, 1.5])] * 2
    assert queries == [SparseVector(indices=[7], values=[1.0])]
    assert all(isinstance(i, int) for i in docs[0].indices)
    assert model.calls == ["embed", "query_embed"]
    assert isinstance(encoder, SparseEncoder)


def test_bm25_empty_inputs(cfg: EngineConfig, tmp_path: Path) -> None:
    model = FakeFastEmbed()
    encoder = FastEmbedBM25Encoder(cfg.sparse, tmp_path, model)
    assert encoder.encode_documents([]) == encoder.encode_queries([]) == []
    assert model.calls == []

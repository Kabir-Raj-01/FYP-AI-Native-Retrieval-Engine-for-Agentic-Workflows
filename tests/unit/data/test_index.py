"""data/index.py: resumable indexing through the IndexWriter interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from data.index import IndexReport, index_chunks
from engine.config import EngineConfig
from engine.errors import ConfigError
from engine.retrieval.qdrant_index import QdrantIndexWriter
from tests.fakes import CorpusChunk, FakeEmbedder, FakeSparseEncoder

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes:UserWarning")
DIM = 8


@pytest.fixture
def setup(
    cfg: EngineConfig, tmp_path: Path, mini_corpus: list[CorpusChunk]
) -> tuple[EngineConfig, Path, QdrantClient]:
    cfg.embedding.dimension = DIM
    cfg.qdrant.collection = "idx_test"
    cfg.qdrant.upsert_batch_size = 10
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "metadata": dict(c.metadata),
                }
            )
            + "\n"
            for c in mini_corpus
        ),
        encoding="utf-8",
    )
    return cfg, path, QdrantClient(":memory:")


def run(cfg: EngineConfig, path: Path, client: QdrantClient, embedder: FakeEmbedder) -> IndexReport:
    return index_chunks(
        cfg, path, embedder, FakeSparseEncoder(), QdrantIndexWriter(cfg, client), log=lambda m: None
    )


def test_indexes_every_chunk_in_batches(
    setup: tuple[EngineConfig, Path, QdrantClient], mini_corpus: list[CorpusChunk]
) -> None:
    cfg, path, client = setup
    embedder = FakeEmbedder(dimension=DIM)
    report = run(cfg, path, client, embedder)
    assert (report.chunks, report.already_indexed, report.upserted) == (
        len(mini_corpus),
        0,
        len(mini_corpus),
    )
    assert client.count("idx_test", exact=True).count == len(mini_corpus)
    passage_calls = [c for c in embedder.calls if c[0] == "passages"]
    assert len(passage_calls) == 3  # 27 chunks / batch of 10


def test_rerun_skips_indexed_chunks(setup: tuple[EngineConfig, Path, QdrantClient]) -> None:
    cfg, path, client = setup
    run(cfg, path, client, FakeEmbedder(dimension=DIM))
    embedder = FakeEmbedder(dimension=DIM)
    report = run(cfg, path, client, embedder)
    assert report.upserted == 0
    assert embedder.calls == []


def test_resume_after_partial_index(setup: tuple[EngineConfig, Path, QdrantClient]) -> None:
    cfg, path, client = setup
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    partial = path.with_name("partial.jsonl")
    partial.write_text("".join(lines[:5]), encoding="utf-8")
    run(cfg, partial, client, FakeEmbedder(dimension=DIM))
    report = run(cfg, path, client, FakeEmbedder(dimension=DIM))
    assert (report.already_indexed, report.upserted) == (5, len(lines) - 5)


def test_foreign_chunk_ids_in_the_collection_are_refused(
    setup: tuple[EngineConfig, Path, QdrantClient],
) -> None:
    cfg, path, client = setup
    run(cfg, path, client, FakeEmbedder(dimension=DIM))
    other = path.with_name("other.jsonl")
    other.write_text(
        path.read_text(encoding="utf-8").splitlines(keepends=True)[0], encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="different chunking"):
        run(cfg, other, client, FakeEmbedder(dimension=DIM))

"""Ingestion step 2: embed, sparse-encode and upsert chunks (DATA.md §3 steps 5-6).

Reads ``chunks.jsonl`` written by step 1 and fills ``qdrant.collection`` through the
``IndexWriter`` interface (components come from ``engine.factory``; ``data/`` never imports
a backend client or model library). Resumable: chunks already in the collection are
skipped, so an interrupted run continues where it stopped. A collection holding chunk ids
that are not in ``chunks.jsonl`` (an older chunking) is refused: an index change needs a new
collection name (ADR-001).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.config import EngineConfig
from engine.embeddings.interfaces import Embedder, SparseEncoder
from engine.errors import ConfigError
from engine.retrieval.interfaces import IndexPoint, IndexWriter

__all__ = ["IndexReport", "index_chunks", "read_chunks"]


@dataclass(frozen=True)
class IndexReport:
    collection: str
    chunks: int
    already_indexed: int
    upserted: int
    seconds: float


def read_chunks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_chunks(
    cfg: EngineConfig,
    chunks_path: Path,
    embedder: Embedder,
    sparse: SparseEncoder,
    writer: IndexWriter,
    *,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
) -> IndexReport:
    """Index every chunk of ``chunks_path`` that the collection does not hold yet.

    Raises:
        ConfigError: the collection has incompatible vector settings or foreign chunk ids.
        BackendError: the backend fails.
    """
    started = clock()
    rows = read_chunks(chunks_path)
    wanted = {r["chunk_id"] for r in rows}
    writer.ensure_collection()
    existing = writer.existing_ids()
    foreign = existing - wanted
    if foreign:
        raise ConfigError(
            f"collection {cfg.qdrant.collection!r} holds {len(foreign)} chunk ids not in "
            f"{chunks_path.name} (different chunking?); use a new qdrant.collection name"
        )
    todo = [r for r in rows if r["chunk_id"] not in existing]
    log(
        f"{cfg.qdrant.collection}: {len(rows)} chunks, {len(existing)} already indexed, "
        f"{len(todo)} to index"
    )

    batch = cfg.qdrant.upsert_batch_size
    done = 0
    for start in range(0, len(todo), batch):
        part = todo[start : start + batch]
        texts = [r["text"] for r in part]
        dense = embedder.embed_passages(texts)
        sparse_vectors = sparse.encode_documents(texts)
        writer.upsert(
            [
                IndexPoint(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    text=r["text"],
                    metadata=r["metadata"],
                    dense=[float(x) for x in d],
                    sparse=s,
                )
                for r, d, s in zip(part, dense, sparse_vectors, strict=True)
            ]
        )
        done += len(part)
        elapsed = clock() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (len(todo) - done) / rate if rate > 0 else 0.0
        log(f"  indexed {done}/{len(todo)} ({rate:.2f} chunks/s, ~{eta / 60:.0f} min left)")
    return IndexReport(cfg.qdrant.collection, len(rows), len(existing), done, clock() - started)

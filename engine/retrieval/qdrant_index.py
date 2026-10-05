"""IndexWriter for Qdrant (RETRIEVAL_SPEC §2; DATA.md §3 step 6)."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid5

from qdrant_client import QdrantClient, models

from engine.config import EngineConfig
from engine.errors import ConfigError
from engine.retrieval.interfaces import IndexPoint, IndexWriter
from engine.retrieval.qdrant_common import backend_errors, make_client

__all__ = ["QdrantIndexWriter", "point_id"]

# Payload fields with a keyword index (RETRIEVAL_SPEC §2: needed by filters).
_KEYWORD_INDEXED = ("chunk_id", "doc_id")


def point_id(cfg: EngineConfig, chunk_id: str) -> str:
    """UUIDv5(``qdrant.id_namespace``, chunk_id): Qdrant ids must be UUIDs or integers.
    Same formula as ``data.ids.point_id_for`` (DATA.md §3)."""
    return str(uuid5(cfg.qdrant.id_namespace, chunk_id))


class QdrantIndexWriter(IndexWriter):
    def __init__(self, cfg: EngineConfig, client: QdrantClient | None = None) -> None:
        self._cfg = cfg
        self._client = client if client is not None else make_client(cfg)
        self._collection = cfg.qdrant.collection

    def ensure_collection(self) -> None:
        q = self._cfg.qdrant
        with backend_errors("collection check"):
            exists = self._client.collection_exists(self._collection)
        if exists:
            self._verify_existing()
            return
        with backend_errors("collection creation"):
            self._client.create_collection(
                self._collection,
                vectors_config={
                    q.dense_vector_name: models.VectorParams(
                        size=self._cfg.embedding.dimension, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    q.sparse_vector_name: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                hnsw_config=models.HnswConfigDiff(m=q.hnsw.m, ef_construct=q.hnsw.ef_construct),
            )
            for field in _KEYWORD_INDEXED:
                self._client.create_payload_index(
                    self._collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def _verify_existing(self) -> None:
        q = self._cfg.qdrant
        with backend_errors("collection info"):
            params = self._client.get_collection(self._collection).config.params
        dense = (
            params.vectors.get(q.dense_vector_name) if isinstance(params.vectors, dict) else None
        )
        sparse = (params.sparse_vectors or {}).get(q.sparse_vector_name)
        problems = []
        if dense is None or dense.size != self._cfg.embedding.dimension:
            problems.append(
                f"dense vector {q.dense_vector_name!r} with size {self._cfg.embedding.dimension}"
            )
        elif dense.distance != models.Distance.COSINE:
            problems.append("cosine distance")
        if sparse is None or sparse.modifier != models.Modifier.IDF:
            problems.append(f"sparse vector {q.sparse_vector_name!r} with modifier=idf")
        if problems:
            raise ConfigError(
                f"collection {self._collection!r} exists without "
                + ", ".join(problems)
                + "; use a new qdrant.collection name (ADR-001) instead of overwriting"
            )

    def existing_ids(self) -> set[str]:
        ids: set[str] = set()
        offset = None
        while True:
            with backend_errors("scroll"):
                records, offset = self._client.scroll(
                    self._collection,
                    limit=self._cfg.qdrant.upsert_batch_size,
                    offset=offset,
                    with_payload=["chunk_id"],
                    with_vectors=False,
                )
            ids.update(str(r.payload["chunk_id"]) for r in records if r.payload)
            if offset is None:
                return ids

    def upsert(self, points: Sequence[IndexPoint]) -> None:
        q = self._cfg.qdrant
        size = q.upsert_batch_size
        for start in range(0, len(points), size):
            batch = [
                models.PointStruct(
                    id=point_id(self._cfg, p.chunk_id),
                    vector={
                        q.dense_vector_name: p.dense,
                        q.sparse_vector_name: models.SparseVector(
                            indices=p.sparse.indices, values=p.sparse.values
                        ),
                    },
                    payload={
                        "chunk_id": p.chunk_id,
                        "doc_id": p.doc_id,
                        "text": p.text,
                        **p.metadata,
                    },
                )
                for p in points[start : start + size]
            ]
            with backend_errors("upsert"):
                self._client.upsert(self._collection, points=batch, wait=True)

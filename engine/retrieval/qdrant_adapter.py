"""RetrieverAdapter for Qdrant: dense, sparse and hybrid (RRF) search (RETRIEVAL_SPEC §1-§3).

- dense:  the query embedded by the Embedder (query prefix), HNSW search on the dense vector
          with ``qdrant.search_ef``.
- sparse: the query encoded by the same SparseEncoder as at ingestion; BM25 with Qdrant-side
          IDF.
- hybrid: Query API prefetch of ``retrieval.hybrid_prefetch_k`` from both, fused with
          Reciprocal Rank Fusion (``retrieval.hybrid_fusion``).
Scores are not comparable across modes; nothing may threshold on them.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from engine.config import EngineConfig
from engine.embeddings.interfaces import Embedder, SparseEncoder
from engine.errors import BackendError
from engine.retrieval.interfaces import RetrieverAdapter, SearchFilters
from engine.retrieval.qdrant_common import backend_errors, make_client
from engine.state.models import Candidate, RetrievalMode

__all__ = ["QdrantAdapter"]

_CORE_PAYLOAD = ("chunk_id", "doc_id", "text")


def _filter(filters: SearchFilters | None) -> models.Filter | None:
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filters.items()
        ]
    )


class QdrantAdapter(RetrieverAdapter):
    def __init__(
        self,
        cfg: EngineConfig,
        embedder: Embedder,
        sparse_encoder: SparseEncoder,
        client: QdrantClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._client = client if client is not None else make_client(cfg)

    def search(
        self,
        query: str,
        mode: RetrievalMode,
        k: int,
        filters: SearchFilters | None = None,
        with_vectors: bool = True,
    ) -> list[Candidate]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        q = self._cfg.qdrant
        common: dict[str, Any] = {
            "limit": k,
            "query_filter": _filter(filters),
            "with_payload": True,
            "with_vectors": [q.dense_vector_name] if with_vectors else False,
        }
        if mode == RetrievalMode.DENSE:
            request: dict[str, Any] = {
                "query": self._dense_query(query),
                "using": q.dense_vector_name,
                "search_params": models.SearchParams(hnsw_ef=q.search_ef),
            }
        elif mode == RetrievalMode.SPARSE:
            request = {"query": self._sparse_query(query), "using": q.sparse_vector_name}
        else:
            prefetch_k = self._cfg.retrieval.hybrid_prefetch_k
            request = {
                "prefetch": [
                    models.Prefetch(
                        query=self._dense_query(query),
                        using=q.dense_vector_name,
                        limit=prefetch_k,
                        filter=common["query_filter"],
                    ),
                    models.Prefetch(
                        query=self._sparse_query(query),
                        using=q.sparse_vector_name,
                        limit=prefetch_k,
                        filter=common["query_filter"],
                    ),
                ],
                "query": models.FusionQuery(fusion=models.Fusion.RRF),
            }
        with backend_errors(f"{mode.value} search"):
            response = self._client.query_points(q.collection, **request, **common)
        return [
            self._candidate(p, rank, mode, with_vectors) for rank, p in enumerate(response.points)
        ]

    def health(self) -> bool:
        try:
            with backend_errors("health check"):
                return bool(self._client.collection_exists(self._cfg.qdrant.collection))
        except BackendError:
            return False

    def _dense_query(self, query: str) -> list[float]:
        return [float(x) for x in self._embedder.embed_queries([query])[0]]

    def _sparse_query(self, query: str) -> models.SparseVector:
        vector = self._sparse.encode_queries([query])[0]
        return models.SparseVector(indices=vector.indices, values=vector.values)

    def _candidate(
        self, point: models.ScoredPoint, rank: int, mode: RetrievalMode, with_vectors: bool
    ) -> Candidate:
        payload = point.payload or {}
        if any(key not in payload for key in _CORE_PAYLOAD):
            raise BackendError(f"point {point.id} lacks {_CORE_PAYLOAD} in its payload")
        vector = None
        if with_vectors:
            vectors = point.vector
            dense = (
                vectors.get(self._cfg.qdrant.dense_vector_name)
                if isinstance(vectors, dict)
                else None
            )
            if not isinstance(dense, list) or not all(isinstance(x, float | int) for x in dense):
                raise BackendError(f"point {point.id} came back without its dense vector")
            vector = [float(x) for x in dense if isinstance(x, float | int)]
        metadata = {k: v for k, v in payload.items() if k not in _CORE_PAYLOAD}
        return Candidate(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload["doc_id"]),
            text=str(payload["text"]),
            retrieval_score=float(point.score),
            rank=rank,
            mode=mode,
            metadata=metadata,
            vector=vector,
        )

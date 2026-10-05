"""SparseEncoder backed by fastembed's BM25 (RETRIEVAL_SPEC §3; ADR-001).

fastembed always downloads the latest snapshot of a model and cannot pin a revision, so the
pinned ``sparse.revision`` snapshot is fetched with ``huggingface_hub.snapshot_download``
into ``paths.model_cache_dir`` and handed to fastembed as ``specific_model_path``.
BM25 parameters come from ``sparse.k1``/``sparse.b``/``sparse.avg_len``. Document vectors
carry the term-frequency part; the IDF part is applied by Qdrant, so the collection's sparse
vector must use ``modifier=idf`` (fastembed's Bm25 docstring; LIBRARY_NOTES).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from engine.config import SparseConfig
from engine.embeddings.interfaces import SparseEncoder, SparseVector

__all__ = ["FastEmbedBM25Encoder"]

# Files of the Qdrant/bm25 repository fastembed reads (stop-word lists and its config).
_SNAPSHOT_PATTERNS = ["*.json", "*.txt"]


class _SparseModel(Protocol):
    def embed(self, documents: list[str], **kwargs: Any) -> Iterable[Any]: ...

    def query_embed(self, query: list[str], **kwargs: Any) -> Iterable[Any]: ...


def _load_model(cfg: SparseConfig, cache_dir: Path) -> _SparseModel:
    from fastembed import SparseTextEmbedding  # heavy import
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=cfg.model_name,
        revision=cfg.revision,
        cache_dir=str(cache_dir),
        allow_patterns=_SNAPSHOT_PATTERNS,
    )
    model: _SparseModel = SparseTextEmbedding(
        cfg.model_name,
        cache_dir=str(cache_dir),
        specific_model_path=snapshot,
        k=cfg.k1,
        b=cfg.b,
        avg_len=cfg.avg_len,
    )
    return model


def _to_sparse(embedding: Any) -> SparseVector:
    return SparseVector(
        indices=[int(i) for i in embedding.indices],
        values=[float(v) for v in embedding.values],
    )


class FastEmbedBM25Encoder(SparseEncoder):
    def __init__(
        self, cfg: SparseConfig, cache_dir: Path, model: _SparseModel | None = None
    ) -> None:
        self._model = model if model is not None else _load_model(cfg, cache_dir)

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [_to_sparse(e) for e in self._model.embed(texts)] if texts else []

    def encode_queries(self, texts: list[str]) -> list[SparseVector]:
        return [_to_sparse(e) for e in self._model.query_embed(texts)] if texts else []

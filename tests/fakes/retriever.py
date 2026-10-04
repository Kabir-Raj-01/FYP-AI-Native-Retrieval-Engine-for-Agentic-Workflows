"""FakeRetriever (TESTING §2): in-memory corpus, scripted or token-overlap results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from engine.embeddings.interfaces import Embedder
from engine.errors import BackendError
from engine.retrieval.interfaces import RetrieverAdapter, SearchFilters
from engine.state.models import Candidate, RetrievalMode

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Mapping[str, str | int | None] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchRequest:
    query: str
    mode: RetrievalMode
    k: int
    filters: SearchFilters | None
    with_vectors: bool


class FakeRetriever(RetrieverAdapter):
    """``scripted[(query, mode)]`` -> chunk ids returned in that order; any other query is
    scored by the fraction of query tokens found in the chunk (zero-score chunks are not
    returned). ``fail_on_call=n`` raises BackendError on the n-th search (1-based)."""

    def __init__(
        self,
        corpus: Iterable[CorpusChunk],
        *,
        embedder: Embedder | None = None,
        scripted: Mapping[tuple[str, RetrievalMode], Sequence[str]] | None = None,
        fail_on_call: int | None = None,
        healthy: bool = True,
    ) -> None:
        self._chunks = {c.chunk_id: c for c in corpus}
        self._embedder = embedder
        self._scripted = dict(scripted or {})
        self._fail_on_call = fail_on_call
        self._healthy = healthy
        self.requests: list[SearchRequest] = []

    def search(
        self,
        query: str,
        mode: RetrievalMode,
        k: int,
        filters: SearchFilters | None = None,
        with_vectors: bool = True,
    ) -> list[Candidate]:
        self.requests.append(SearchRequest(query, mode, k, filters, with_vectors))
        if self._fail_on_call is not None and len(self.requests) == self._fail_on_call:
            raise BackendError(f"fake backend failure on call {len(self.requests)}")
        ranked = self._rank(query, mode)
        ranked = [(c, s) for c, s in ranked if self._matches(c, filters)][:k]
        vectors = None
        if with_vectors and self._embedder is not None and ranked:
            vectors = self._embedder.embed_passages([c.text for c, _ in ranked])
        return [
            Candidate(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                retrieval_score=score,
                rank=rank,
                mode=mode,
                metadata=dict(c.metadata),
                vector=None if vectors is None else [float(x) for x in vectors[rank]],
            )
            for rank, (c, score) in enumerate(ranked)
        ]

    def health(self) -> bool:
        return self._healthy

    def _rank(self, query: str, mode: RetrievalMode) -> list[tuple[CorpusChunk, float]]:
        scripted = self._scripted.get((query, mode))
        if scripted is not None:
            n = len(scripted)
            return [(self._chunks[cid], float(n - i)) for i, cid in enumerate(scripted)]
        wanted = tokens(query)
        scored = [
            (chunk, len(wanted & tokens(chunk.text)) / len(wanted))
            for chunk in self._chunks.values()
            if wanted
        ]
        scored = [(c, s) for c, s in scored if s > 0]
        return sorted(scored, key=lambda cs: (-cs[1], cs[0].chunk_id))

    @staticmethod
    def _matches(chunk: CorpusChunk, filters: SearchFilters | None) -> bool:
        if not filters:
            return True
        fields = {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, **chunk.metadata}
        return all(fields.get(key) == value for key, value in filters.items())

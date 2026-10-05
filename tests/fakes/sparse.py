"""FakeSparseEncoder: term-count sparse vectors with stable hashed term ids (no model)."""

from __future__ import annotations

import re
import zlib
from collections import Counter

from engine.embeddings.interfaces import SparseEncoder, SparseVector

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _vector(text: str, *, binary: bool) -> SparseVector:
    counts = Counter(zlib.crc32(t.encode("utf-8")) for t in _TOKEN.findall(text.lower()))
    ids = sorted(counts)
    return SparseVector(indices=ids, values=[1.0 if binary else float(counts[i]) for i in ids])


class FakeSparseEncoder(SparseEncoder):
    """Documents get term counts, queries get 1.0 per term (like BM25 with server-side IDF)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        self.calls.append(("documents", list(texts)))
        return [_vector(t, binary=False) for t in texts]

    def encode_queries(self, texts: list[str]) -> list[SparseVector]:
        self.calls.append(("queries", list(texts)))
        return [_vector(t, binary=True) for t in texts]

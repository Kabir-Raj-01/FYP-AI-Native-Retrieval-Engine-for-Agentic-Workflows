"""Stable identifiers (DATA.md §3 step 4; RETRIEVAL_SPEC §2; ADR-001)."""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid5

__all__ = ["chunk_id_for", "doc_id_for", "point_id_for"]

# DATA.md §3 fixes chunk_id = sha1(...)[:16]; doc_id uses the same length (Week 8 plan).
_ID_HEX_CHARS = 16


def _sha1_prefix(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:_ID_HEX_CHARS]


def doc_id_for(url: str) -> str:
    """MultiHop-RAG has no document id; ``url`` is unique across the corpus (ADR-003)."""
    return _sha1_prefix(url)


def chunk_id_for(doc_id: str, char_start: int) -> str:
    """``sha1(f"{doc_id}:{char_start}")[:16]`` with ``char_start`` into the normalised text."""
    return _sha1_prefix(f"{doc_id}:{char_start}")


def point_id_for(chunk_id: str, namespace: UUID) -> UUID:
    """Qdrant point id: UUIDv5(``qdrant.id_namespace``, chunk_id) — point ids must be UUIDs."""
    return uuid5(namespace, chunk_id)

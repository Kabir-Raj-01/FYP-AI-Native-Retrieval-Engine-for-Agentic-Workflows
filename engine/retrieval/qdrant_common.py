"""Qdrant client construction and error translation shared by the adapter and index writer."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from engine.config import EngineConfig
from engine.errors import BackendError

__all__ = ["backend_errors", "make_client"]

# Exceptions the client raises for timeouts, connection failures and error responses.
_CLIENT_ERRORS = (ResponseHandlingException, UnexpectedResponse, httpx.HTTPError, OSError)


def make_client(cfg: EngineConfig) -> QdrantClient:
    """Client for ``qdrant.url`` with ``qdrant.timeout_s`` (127.0.0.1, never localhost)."""
    return QdrantClient(url=cfg.qdrant.url, timeout=cfg.qdrant.timeout_s)


@contextmanager
def backend_errors(action: str) -> Iterator[None]:
    """Re-raise client failures as BackendError (RETRIEVAL_SPEC §1)."""
    try:
        yield
    except _CLIENT_ERRORS as exc:
        raise BackendError(f"Qdrant {action} failed: {exc}") from exc

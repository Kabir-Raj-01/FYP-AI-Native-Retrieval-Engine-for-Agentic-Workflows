"""Engine facade: create_task · step · run · retrieve_direct (ARCHITECTURE §1, §4, §6).

This version implements the controller-OFF direct path (FR3). The controlled loop
(``create_task``/``step``/``run``: TaskStore, LangGraph graph, per-task lock) is TASKS
Weeks 8-9 work and raises NotImplementedError until then.

The direct path never touches task state, the cache, the planner or the controller, and
``Engine.from_config`` builds none of them — so it keeps working when controlled components
or the LLM are unavailable (FR3.4). Retriever and reranker are the same instances the
controlled path will use (FR3.3).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from engine import factory
from engine.config import EngineConfig
from engine.errors import ConfigError
from engine.monitor.logging import get_logger
from engine.rerank.interfaces import Reranker
from engine.retrieval.interfaces import RetrieverAdapter
from engine.state.models import DirectResult, RetrievalMode

__all__ = ["Engine"]


class Engine:
    def __init__(
        self,
        cfg: EngineConfig,
        *,
        retriever: RetrieverAdapter,
        reranker: Reranker | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._cfg = cfg
        self._retriever = retriever
        self._reranker = reranker
        self._clock = clock
        self._log = get_logger("engine.core")

    @classmethod
    def from_config(cls, cfg: EngineConfig) -> Engine:
        """Build the direct-path components with the factory (no planner, LLM or store)."""
        embedder = factory.build_embedder(cfg)
        sparse = factory.build_sparse_encoder(cfg)
        retriever = factory.build_retriever(cfg, embedder, sparse)
        try:
            reranker: Reranker | None = factory.build_reranker(cfg, embedder)
        except NotImplementedError:  # Reranker arrives with ADR-005 (Weeks 11-12)
            reranker = None
        return cls(cfg, retriever=retriever, reranker=reranker)

    @property
    def cfg(self) -> EngineConfig:
        return self._cfg

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None

    def health(self) -> bool:
        """True if the retrieval backend is reachable and the collection exists."""
        return self._retriever.health()

    def retrieve_direct(
        self,
        query: str,
        mode: RetrievalMode,
        k: int | None = None,
        rerank: bool = False,
    ) -> DirectResult:
        """One-shot retrieval through the shared adapter (ARCHITECTURE §6; ADR-012).

        Args:
            k: candidates to retrieve; ``None`` uses ``retrieval.default_k``.
            rerank: also score the candidates with the relevance-only reranker
                (``Reranker.rerank(query, candidates, state=None)``).

        Raises:
            ValueError: ``k`` outside ``[1, retrieval.max_k]``.
            ConfigError: ``rerank=True`` but no reranker is available.
            BackendError: the backend fails.
        """
        size = self._cfg.retrieval.default_k if k is None else k
        if not 1 <= size <= self._cfg.retrieval.max_k:
            raise ValueError(f"k must be in [1, {self._cfg.retrieval.max_k}], got {size}")
        if rerank and self._reranker is None:
            raise ConfigError("rerank requested but no reranker is available yet (ADR-005)")

        start = self._clock()
        candidates = self._retriever.search(query, mode, size)
        scored = None
        if rerank and self._reranker is not None:
            scored = self._reranker.rerank(query, candidates, None)
        latency_ms = (self._clock() - start) * 1000.0
        self._log.info(
            "retrieval.call",
            path="direct",
            mode=mode.value,
            k=size,
            query=query,
            n_returned=len(candidates),
            rerank=rerank,
            latency_ms=round(latency_ms, 2),
        )
        return DirectResult(candidates=candidates, scored=scored, latency_ms=latency_ms)

    # ----------------------------------------------------- controlled path (Weeks 8-9)
    def create_task(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("controlled path: TaskStore + graph (docs/TASKS.md Weeks 8-9)")

    def step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("controlled path: TaskStore + graph (docs/TASKS.md Weeks 8-9)")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("controlled path: TaskStore + graph (docs/TASKS.md Weeks 8-9)")

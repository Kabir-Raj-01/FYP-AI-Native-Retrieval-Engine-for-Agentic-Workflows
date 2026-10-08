"""Engine facade: create_task · step · run · retrieve_direct (ARCHITECTURE §1, §4, §6).

The direct path never touches task state, the cache, the planner or the controller, and
``Engine.from_config`` builds none of them eagerly — so it keeps working when controlled
components or the LLM are unavailable (FR3.4). The controlled components are built through
the factory on first use of ``create_task``/``step``/``run``. Retriever and reranker are the
same instances on both paths (FR3.3).

Controlled path (ARCHITECTURE §3-§4, §8): ``step`` loads S_t from TaskStore, runs one graph
iteration (``engine/graph``) and returns its TraceStep; ``run`` iterates until the task
stops, checking ``api.run_timeout_s`` between iterations. A per-task in-process lock
rejects a concurrent step/run on the same task with TaskConflict (API_CONTRACT: the second
caller gets 409); TaskStore's version check is the backstop. A failing iteration is never
persisted; after a non-retryable error the task is marked FAILED / ERROR directly in
TaskStore (no TraceStep, only an ``error`` event).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from engine import factory
from engine.config import EngineConfig
from engine.embeddings.interfaces import Embedder
from engine.errors import (
    BackendError,
    ConfigError,
    LLMBudgetExceeded,
    LLMError,
    RunTimeout,
    TaskConflict,
    TaskNotActive,
)
from engine.graph.graph import run_iteration
from engine.graph.nodes import Components, Context
from engine.monitor.interfaces import TraceWriter
from engine.monitor.logging import bound_contextvars, get_logger
from engine.planner.interfaces import Planner
from engine.rerank.interfaces import Reranker
from engine.retrieval.interfaces import RetrieverAdapter
from engine.state.models import (
    Budget,
    DirectResult,
    EvidenceBundle,
    RetrievalMode,
    RetrievalState,
    StopReason,
    TaskStatus,
    TraceStep,
)

__all__ = ["Engine"]

# After these the task stays ACTIVE and the caller may retry (ARCHITECTURE §8); any other
# exception from an iteration marks the task FAILED / ERROR. LLMBudgetExceeded is the
# per-run cap: it aborts the run (LLM_CLIENT_SPEC §8), but the task itself is not broken.
_RETRYABLE = (BackendError, LLMError, LLMBudgetExceeded, TaskConflict)


class Engine:
    def __init__(
        self,
        cfg: EngineConfig,
        *,
        retriever: RetrieverAdapter,
        reranker: Reranker | None = None,
        embedder: Embedder | None = None,
        traces: TraceWriter | None = None,
        controlled: Components | None = None,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """``controlled`` supplies every controlled-path component (tests); otherwise they
        are built through the factory on first use, from ``embedder`` and ``traces``."""
        self._cfg = cfg
        self._retriever = retriever
        self._reranker = reranker
        self._embedder = embedder
        self._traces = traces
        self._controlled = controlled
        self._clock = clock
        self._now = now
        self._log = get_logger("engine.core")
        self._guard = threading.Lock()
        # ponytail: one Lock per task id, never removed; prune if a server runs for weeks
        self._task_locks: dict[str, threading.Lock] = {}
        self._template: Components | None = None  # shared parts for for_run()
        self._llm: object | None = None  # the LLM client, once built (run totals)

    @classmethod
    def from_config(cls, cfg: EngineConfig, traces: TraceWriter | None = None) -> Engine:
        """Build the direct-path components with the factory (no planner, LLM or store).

        ``traces`` is where the controlled path writes TraceSteps and bundles; without it
        only the direct path is available.
        """
        embedder = factory.build_embedder(cfg)
        sparse = factory.build_sparse_encoder(cfg)
        retriever = factory.build_retriever(cfg, embedder, sparse)
        reranker = factory.build_reranker(cfg, embedder)  # the model loads on first use
        return cls(cfg, retriever=retriever, reranker=reranker, embedder=embedder, traces=traces)

    def for_run(self, traces: TraceWriter, state_db: Path) -> Engine:
        """An Engine for one eval sub-run: the same retriever, reranker, embedder, LLM client,
        planner, controller, evidence manager and monitor (so per-run LLM caps span the
        run), with its own TaskStore + cache at ``state_db`` and its own ``traces``."""
        with self._guard:
            if self._template is None:
                self._template = self._controlled or self._build_controlled(traces)
            base = self._template
        cfg = self._cfg.model_copy(deep=True)
        cfg.paths.state_db = state_db
        parts = replace(
            base,
            store=factory.build_task_store(cfg),
            cache=factory.build_cache(cfg, base.embedder),
            traces=traces,
        )
        child = Engine(
            self._cfg,
            retriever=self._retriever,
            reranker=self._reranker,
            embedder=self._embedder,
            traces=traces,
            controlled=parts,
            clock=self._clock,
            now=self._now,
        )
        child._llm = self._llm
        return child

    def llm_run_totals(self) -> dict[str, float]:
        """Real LLM requests and spend of this process's LLM client (run manifest)."""
        totals = getattr(self._llm, "run_totals", None)
        return dict(totals) if totals is not None else {"llm_calls": 0, "cost_usd": 0.0}

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
        latency_ms = search_ms = (self._clock() - start) * 1000.0
        self._log.info(
            "retrieval.call",
            path="direct",
            mode=mode.value,
            k=size,
            query=query,
            n_returned=len(candidates),
            rerank=rerank,
            latency_ms=round(search_ms, 2),  # search only (RETRIEVAL_SPEC §4)
        )
        scored = None
        rerank_ms = None
        if rerank and self._reranker is not None:
            scored = self._reranker.rerank(query, candidates, None)
            latency_ms = (self._clock() - start) * 1000.0
            rerank_ms = latency_ms - search_ms
        return DirectResult(
            candidates=candidates, scored=scored, latency_ms=latency_ms, rerank_latency_ms=rerank_ms
        )

    # -------------------------------------------------------------------- controlled path
    def default_budget(self) -> Budget:
        d = self._cfg.budget_defaults
        return Budget(
            max_steps=d.max_steps,
            max_retrieval_calls=d.max_retrieval_calls,
            max_llm_calls=d.max_llm_calls,
            max_tokens=d.max_tokens,
        )

    def create_task(
        self, goal: str, budget: Budget | None = None, task_id: str | None = None
    ) -> RetrievalState:
        """Create S_0 (``budget`` defaults to ``budget_defaults``). Raises TaskExists."""
        return self._parts().store.create(goal, budget or self.default_budget(), task_id)

    def state(self, task_id: str) -> RetrievalState:
        return self._parts().store.load(task_id)

    @property
    def planner(self) -> Planner:
        """The controlled path's planner (the fixed_iteration baseline uses the same one)."""
        return self._parts().planner

    def trace(self, task_id: str) -> list[TraceStep]:
        """Every TraceStep of ``task_id`` written so far (raises TaskNotFound first)."""
        self._parts().store.load(task_id)
        return self._parts().traces.read_steps(task_id)

    def step(self, task_id: str) -> TraceStep:
        """Run exactly one iteration and return its TraceStep.

        Raises:
            TaskNotFound, TaskNotActive: before anything runs.
            TaskConflict: another step/run on this task is in progress, or a stale save.
            BackendError, LLMError: the iteration failed; the task stays ACTIVE.
        """
        with self._task_lock(task_id):
            _, trace = self._iterate(self._load_active(task_id))
            return trace

    def run(self, task_id: str, timeout_s: float | None = None) -> EvidenceBundle:
        """Iterate until the task stops and return its bundle.

        ``timeout_s`` defaults to ``api.run_timeout_s`` (the HTTP ``/run``); the eval
        harness passes ``math.inf``.

        Raises:
            RunTimeout: the timeout passed; the state is the last completed step.
            (and as ``step``)
        """
        limit = self._cfg.api.run_timeout_s if timeout_s is None else timeout_s
        with self._task_lock(task_id):
            state = self._load_active(task_id)
            deadline = self._clock() + limit
            while state.status == TaskStatus.ACTIVE:
                if self._clock() >= deadline:
                    raise RunTimeout(f"task {task_id}: run exceeded {limit} s", task_id=task_id)
                state, _ = self._iterate(state)
            return self._bundle(state)

    def result(self, task_id: str) -> EvidenceBundle:
        """Bundle of a stopped (or failed) task. Raises TaskNotActive while it is active."""
        state = self._parts().store.load(task_id)
        if state.status == TaskStatus.ACTIVE:
            raise TaskNotActive(f"task {task_id} is still active", task_id=task_id)
        return self._bundle(state)

    def _parts(self) -> Components:
        with self._guard:
            if self._controlled is None:
                self._controlled = self._build_controlled(self._traces)
            return self._controlled

    def _build_controlled(self, traces: TraceWriter | None) -> Components:
        cfg = self._cfg
        if self._embedder is None or traces is None or self._reranker is None:
            raise ConfigError("the controlled path needs an embedder, a reranker and traces")
        llm = factory.build_llm_client(cfg)
        self._llm = llm
        return Components(
            store=factory.build_task_store(cfg),
            cache=factory.build_cache(cfg, self._embedder),
            planner=factory.build_planner(cfg, llm),
            controller=factory.build_controller(cfg, self._embedder),
            retriever=self._retriever,
            reranker=self._reranker,
            evidence=factory.build_evidence_manager(cfg, self._embedder, llm),
            monitor=factory.build_monitor(cfg),
            embedder=self._embedder,
            traces=traces,
        )

    @contextmanager
    def _task_lock(self, task_id: str) -> Iterator[None]:
        with self._guard:
            lock = self._task_locks.setdefault(task_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise TaskConflict(f"task {task_id}: a step or run is in progress", task_id=task_id)
        try:
            yield
        finally:
            lock.release()

    def _load_active(self, task_id: str) -> RetrievalState:
        state = self._parts().store.load(task_id)
        if state.status != TaskStatus.ACTIVE:
            raise TaskNotActive(f"task {task_id} is {state.status.value}", task_id=task_id)
        return state

    def _iterate(self, state: RetrievalState) -> tuple[RetrievalState, TraceStep]:
        parts = self._parts()
        ctx = Context(
            cfg=self._cfg,
            parts=parts,
            prev=state.model_copy(deep=True),  # nodes must not mutate it (LIBRARY_NOTES)
            clock=self._clock,
            now=self._now,
            started=self._clock(),
        )
        with bound_contextvars(
            run_id=parts.traces.context.run_id, task_id=state.task_id, step=ctx.n
        ):
            try:
                new = run_iteration(state, ctx)
            except Exception as exc:
                retryable = isinstance(exc, _RETRYABLE)
                self._log.error(
                    "error",
                    type=type(exc).__name__,
                    message=str(exc),
                    component="engine.core",
                    retryable=retryable,
                )
                if not retryable:
                    self._fail(ctx.prev)
                raise
            if new.status != TaskStatus.ACTIVE:
                bundle = self._bundle(new)
                parts.traces.write_bundle(bundle)
                self._log.info(
                    "task.stopped",
                    stop_reason=new.stop_reason,
                    summary=bundle.trace_summary.model_dump(),
                )
        if ctx.trace is None:  # persist always sets it
            raise RuntimeError(f"task {state.task_id}: iteration ended without a trace step")
        return new, ctx.trace

    def _fail(self, state: RetrievalState) -> None:
        """status = FAILED, stop_reason = ERROR, saved outside the graph (ADR-009 §4)."""
        failed = state.model_copy(
            update={
                "status": TaskStatus.FAILED,
                "stop_reason": StopReason.ERROR,
                "version": state.version + 1,
                "updated_at": self._now(),
            }
        )
        with suppress(TaskConflict):  # the iteration was saved after all; keep that state
            self._parts().store.save(failed, state.version)

    def _bundle(self, state: RetrievalState) -> EvidenceBundle:
        traces = self._parts().traces
        if state.stop_reason is None:
            raise TaskNotActive(f"task {state.task_id} has not stopped", task_id=state.task_id)
        return EvidenceBundle(
            task_id=state.task_id,
            system=traces.context.system,
            goal=state.goal,
            evidence=sorted(state.evidence, key=lambda e: -e.final_score),
            gaps=state.gaps,
            stop_reason=state.stop_reason,
            trace_summary=TraceWriter.summarize(traces.read_steps(state.task_id)),
        )

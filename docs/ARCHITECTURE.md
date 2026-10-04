# Architecture

> Status: Stable · Owner: All · Last updated: 2026-09-28 · Related: STATE_SPEC.md, ADR-002, ADR-009

## 1. Layers
```
┌──────────────────────────────────────────────────────────────────────┐
│ Host RAG / agent application  (owns final answer generation)         │
└───────────────┬──────────────────────────────────────────────────────┘
                │ HTTP (FastAPI)  or  RetrievalClient  or  Engine (in-process, eval)
┌───────────────▼──────────────────────────────────────────────────────┐
│ engine/api  — thin routers, request validation, error mapping         │
├──────────────────────────────────────────────────────────────────────┤
│ engine/core.Engine  — facade: create_task · step · run · retrieve_direct │
├───────────────┬──────────────────────────────────────────────────────┤
│ DIRECT path   │ CONTROLLED path (engine/graph, LangGraph)            │
│ adapter.search│  A. Pre-retrieval:  plan → select_action → cache gate │
│ (+ rerank for │  B. Retrieval & feedback: execute → rerank → evidence │
│  baselines)   │     → monitor → persist → (loop | END)                │
├───────────────┴──────────────────────────────────────────────────────┤
│ Components (each behind an interface; chosen in engine/factory.py)   │
│ TaskStore · Planner · Cache · Controller · RetrieverAdapter ·         │
│ Reranker · EvidenceManager · TrajectoryMonitor · Embedder · LLMClient │
├──────────────────────────────────────────────────────────────────────┤
│ Storage: SQLite (task state, cache, LLM response cache) · JSONL traces│
│ Backend: Qdrant (dense/HNSW + sparse/BM25 named vectors)             │
└──────────────────────────────────────────────────────────────────────┘
```

## 2. Components and interfaces
All interfaces live in `engine/<module>/interfaces.py`. All models come from
`engine/state/models.py` (STATE_SPEC.md).

| Component | Interface (method signatures) | Spec |
|---|---|---|
| TaskStore | `create(goal, budget, task_id=None) -> RetrievalState` (caller-supplied `task_id` per API_CONTRACT `CreateTaskRequest`; duplicate → `TaskExists`) · `load(task_id) -> RetrievalState` · `save(state: RetrievalState, expected_version: int, cache_entries: Sequence[CacheEntry] = ()) -> None` (atomic: commits the state save and every entry in `cache_entries` in one transaction; raises `TaskConflict` if the stored version ≠ `expected_version`, in which case none of it is written) | STATE_SPEC |
| Planner | `plan(state, mode: PlanMode) -> PlanOutcome` | PLANNER_SPEC |
| Cache | `lookup(task_id, query, need_type) -> CacheHit \| None` · `put(entry: CacheEntry) -> None` | CACHE_SPEC |
| Controller | `select_action(state, cache_hit) -> ActionDecision` (sets `plan_mode` when `action == TRANSFORM`, ADR-009 §3) | ACTION_POLICY |
| RetrieverAdapter | `search(query, mode, k, filters: SearchFilters \| None = None, with_vectors=True) -> list[Candidate]` · `health() -> bool` (`fetch` dropped until a caller needs it, ADR-009 "Also") | RETRIEVAL_SPEC |
| Reranker | `rerank(query, candidates, state: RetrievalState \| None) -> list[ScoredCandidate]` (`None` → relevance-only, used by the direct path/baselines, ADR-009 "Also") | RERANK_SPEC |
| EvidenceManager | `update(state, scored, step) -> EvidenceUpdate` | EVIDENCE_MANAGER_SPEC |
| TrajectoryMonitor | `observe(state, decision, update: EvidenceUpdate \| None) -> MonitorVerdict` (`None` on TRANSFORM/STOP steps) · `advance(state) -> AdvanceResult` (ADR-009 §4) | ACTION_POLICY §6 |
| Embedder | `embed_queries(texts) -> ndarray` · `embed_passages(texts) -> ndarray` | RETRIEVAL_SPEC §5 |
| LLMClient | `structured(role, messages, schema) -> LLMResult[T]` · `text(role, messages) -> LLMResult[str]` | LLM_CLIENT_SPEC |

**Dependency rule:** a module may import `engine.state.models`, `engine.config`, and
other modules' `interfaces.py` only. `engine/factory.py` is the single place that imports
concrete classes. `examples/` and host apps import `client/` only.

## 3. Execution model (ADR-002)
- **Synchronous core.** All components are synchronous. FastAPI runs sync endpoints in
  its threadpool. The eval harness parallelises across tasks with a thread pool
  (`eval.max_workers`). This keeps the code simple and debuggable.
- **Source of truth for state = TaskStore (SQLite).** LangGraph executes the loop with
  the `RetrievalState` Pydantic model as graph state but **without** a persistent
  LangGraph checkpointer. A `persist` node writes the state to TaskStore at the end of
  every iteration. There is exactly one persisted copy of S_t.
- **Per-task lock.** An in-process `threading.Lock` per `task_id`, held inside `Engine`
  for the duration of one `step`/`run` call, plus TaskStore's `version`-based optimistic
  check as a backstop (ADR-009 "Also"). The in-process lock is correct only because this
  service is deployed as one process (the API server's threadpool and the eval harness's
  thread pool are both in-process); the version check is what a future multi-process
  deployment would rely on, and what catches a lock bug today. `persist` sets
  `new_state.version = expected_version + 1` in memory before calling `save`, so no
  read-back is needed. A version conflict is mapped to HTTP 409.
- **SQLite in WAL mode** so parallel eval workers (different tasks) do not block.

## 4. The LangGraph graph (`engine/graph/`)
Nodes are plain functions `node(graph_state: GraphState, ctx) -> dict` returning partial
updates to `GraphState` (`state` and/or `scratch`); `ctx` holds the component instances
built by the factory. `GraphState`/`IterationScratch` are defined in STATE_SPEC §4
(ADR-009 §2) — `scratch` carries per-iteration data (cache hit, fresh candidates, scored
candidates, plan outcome, queued cache entries) between nodes and is discarded by
`persist`; it is never part of `RetrievalState` and never persisted.

```
START
  └─► maybe_plan ──► select_action ──► execute ──► score_and_update ──► monitor ──► persist
         ▲                                                                           │
         └──────────────────── continue (loop) ◄──────────────────────────────────────┤
                                                                     stop / step-mode ─► END
```
| Node | Does | Writes (see ownership matrix, STATE_SPEC §4) |
|---|---|---|
| `maybe_plan` | If `state.needs_initial_plan`: call `Planner.plan(state, INIT)`; create gaps; call `TrajectoryMonitor.advance(state)` to set `current_gap_id` (`current_subgoal`/`pending_query`/`current_need_type` follow automatically as computed properties of `current_gap_id`, ADR-009 §4 correction). Otherwise no-op. | gaps (create), current_gap_id, query_history, plan_events, budget.llm_calls_used/tokens_used, needs_initial_plan |
| `select_action` | `Cache.lookup` for the pending query (→ `scratch.cache_hit`), then `Controller.select_action` (sets `plan_mode` when choosing TRANSFORM) | last_decision |
| `execute` | Run the chosen action (table §5); a REWRITE writes only `gap.query`/`gap.need_type` on the current gap (`pending_query`/`current_need_type` follow automatically); on SPLIT, calls `TrajectoryMonitor.advance(state)` after creating children | retrieval_history, seen_candidate_ids, `scratch.candidates`, budget counters, gaps (REWRITE's `query`/`need_type`; SPLIT → superseded + children) |
| `score_and_update` | If the action produced candidates: `Reranker.rerank(query, scratch.candidates, state)` → `scratch.scored`, then `EvidenceManager.update` → `scratch.evidence_update`; queues a fresh `CacheEntry` in `scratch.queued_cache_entries` for fresh retrievals (not written to the DB yet) | evidence, gaps (resolve), last_update, last_result, budget.reranker_calls_used |
| `monitor` | `TrajectoryMonitor.observe` → `MonitorVerdict` carrying the full new values (continue/stop, stagnation bookkeeping, gap status changes, current gap) — applied verbatim, not recomputed | stagnation_count, transforms_in_streak, action_gain_ema, current_gap_id, status, stop_reason |
| `persist` | `budget.steps_used += 1`; `TaskStore.save(state, expected_version, cache_entries=scratch.queued_cache_entries)` — one atomic call commits both (ADR-009 §9 correction); append TraceStep (`step = budget.steps_used`, the just-incremented value); discard `scratch` | version, updated_at, budget.steps_used, cache_entries table |

Conditional edge after `persist`: `END` if `status != ACTIVE` or the graph was invoked in
single-step mode; otherwise back to `maybe_plan`. LangGraph's `recursion_limit` is set
from `budget.max_steps × nodes_per_iteration + margin` as a safety net only — the monitor
is the real stopping mechanism (see ACTION_POLICY §6.4's `steps_used + 1 >= max_steps`
check and ADR-009 §12: with `max_steps = N` the engine runs exactly `N` real iterations).

If a non-retryable error aborts an iteration, the `Engine` facade sets
`status = FAILED, stop_reason = ERROR` directly via `TaskStore` outside this graph
(ARCHITECTURE §8; ADR-009 §4) — no TraceStep is written for that transition, only an
`error` event.

`Engine.step(task_id)` runs exactly one iteration; `Engine.run(task_id)` loops until stop.

## 5. Action execution semantics
Every action's `budget.steps_used` increment happens uniformly in `persist` (§4), not
per action — the "steps +1" below is the eventual effect of every iteration, listed here
only for the other budget counters each action touches.
| Action | Backend call | Candidates produced | Budget effect |
|---|---|---|---|
| REUSE | none (cache hit) | the hit's stored candidates, reranked against the current pending query (ADR-009 §9) | steps +1 |
| DENSE | `search(pending_query, dense, retrieval.default_k)` | yes | steps +1, retrieval_calls +1 |
| SPARSE_HYBRID | `search(pending_query, sparse if need=lexical else hybrid, retrieval.default_k)` | yes | steps +1, retrieval_calls +1 |
| RERANK_EXPAND | `search(last_result.query, last_result.mode, min(last_result.depth + retrieval.expand_step_k, retrieval.max_k))`; only unseen candidates passed on | yes | steps +1, retrieval_calls +1 |
| TRANSFORM | `Planner.plan(state, decision.plan_mode)` | none | steps +1, llm_calls +attempts |
| STOP | none | none | steps +1 |

## 6. Direct (controller-OFF) path
`Engine.retrieve_direct(query, mode, k, rerank: bool)` calls the same `RetrieverAdapter`
instance; optionally applies `Reranker.rerank(query, candidates, state=None)`, which is
relevance-only mode (ADR-009 "Also"). No TaskStore, cache, planner or controller.
Baselines (BASELINES_SPEC.md) are built from this path plus, for fixed-iteration, the
same Planner.

## 7. Storage layout
| Store | Location (config key) | Content |
|---|---|---|
| Task state | `paths.state_db` | table `tasks(task_id PK, version, status, state_json, created_at, updated_at)` |
| Cache | `paths.state_db` | table `cache_entries(task_id, entry_id, entry_json)`, index on task_id |
| LLM response cache | `paths.llm_cache_db` | `llm_cache(key PK, response_json, usage_json, created_at)` |
| Traces / runs | `paths.runs_dir` | see OBSERVABILITY.md |
| Qdrant | `qdrant.url`, `qdrant.collection` | see RETRIEVAL_SPEC.md |

## 8. Error handling
- Components raise typed exceptions from `engine/errors.py`, all subclasses of
  `EngineError`: `BackendError`, `LLMError`, `LLMOutputError` (LLM_CLIENT_SPEC §5),
  `LLMBudgetExceeded`, `PlannerValidationError` (internal only — planner falls back),
  `TaskNotFound`, `TaskExists`, `TaskConflict`, `TaskNotActive`, `RunTimeout`,
  `ConfigError`, `StateInvariantError` (ADR-009 §5; programming error). The hierarchy is
  flat — `LLMOutputError` and `LLMBudgetExceeded` are not `LLMError` subclasses, so an
  `except LLMError` never swallows a schema failure or the per-run cap. Each class carries
  its API_CONTRACT error `code` and HTTP status; classes with no API_CONTRACT row map to
  500 `INTERNAL_ERROR`.
- A failure inside an iteration aborts the iteration **before** `persist`, so the stored
  state is the last fully completed step, and no `scratch.queued_cache_entries` are
  committed (no orphan cache entries — ADR-009 §9). The error is logged as an event and,
  via the API, mapped to the error codes in API_CONTRACT.md. The task stays `ACTIVE`
  (caller may retry) unless the error is non-retryable, in which case the `Engine`
  facade sets `status = FAILED`, `stop_reason = ERROR` directly via `TaskStore`, outside
  any graph node (STATE_SPEC §4 ownership matrix; ADR-009 §4).

## 9. Change log
| Date | Change | ADR |
|---|---|---|
| 2026-09-25 | Initial architecture; sync core; TaskStore as single source of truth | ADR-002 |
| 2026-09-28 | GraphState/scratch, TaskStore.save signature, plan_mode, ownership fixes, per-task lock, no fetch(), Reranker state\|None | ADR-009 |
| 2026-10-04 | §8 error list completed from the specs (`LLMOutputError`, `StateInvariantError`) plus `TaskExists`/`RunTimeout` for API_CONTRACT's `TASK_EXISTS`/`RUN_TIMEOUT`; flat hierarchy under `EngineError` (Week 7 plan review) | — |
| 2026-10-05 | §2: `TaskStore.create` gains `task_id=None`; `search` filters typed `SearchFilters = Mapping[str, str \| int \| None]`; `observe`'s `update` may be `None`; `advance` is a `TrajectoryMonitor` method. Signatures pinned by `tests/unit/test_interfaces.py` | needs ADR line (Week 7 checkpoint) |

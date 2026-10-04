# ADR-002: Synchronous core; TaskStore as the single source of truth for state

> Status: Accepted · Date: 2026-09-25 · Author(s): Samay · Reviewer: Vishal, Kabir

## Context
LangGraph can persist graph state with its own checkpointers, while the service also
needs task state keyed by `task_id` that survives restarts and is readable via the API.
Two persisted copies would drift. Separately, async code adds complexity for a team of
three with little benefit at benchmark scale.

## Decision
1. All components are synchronous. FastAPI sync endpoints run in its threadpool; the eval
   harness parallelises across tasks with a thread pool (`eval.max_workers`).
2. `TaskStore` (SQLite, WAL mode) is the only persisted copy of `RetrievalState`.
   LangGraph is compiled **without** a persistent checkpointer; a `persist` node saves to
   TaskStore at the end of every iteration.
3. Optimistic concurrency via `RetrievalState.version`; conflicting saves raise
   `TaskConflict` (HTTP 409).

## Alternatives considered
- LangGraph checkpointer (SQLite) as the source of truth — ties the API contract to
  LangGraph's internal checkpoint format; harder to query and to keep stable.
- Async core — better throughput, but LLM/Qdrant call volume here does not need it;
  higher bug risk.

## Consequences
ARCHITECTURE §3–4 and STATE_SPEC ownership matrix follow this. LangGraph is used for
explicit node/edge structure and conditional routing, not for persistence.

## Reversal condition
If eval wall-time becomes a blocker even with thread-level parallelism, revisit async for
the LLM client only.

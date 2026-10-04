# Observability: Traces, Events, Runs

> Status: Frozen for the TraceStep schema (ADR to change) · Owner: Vishal · Last updated: 2026-09-28
> TraceStep amended by ADR-009 (schema_version, step semantics, run_id/system via RunContext).

## 1. Three kinds of records
| Record | Granularity | Where | Purpose |
|---|---|---|---|
| **TraceStep** | one per step per task | `traces/<task_id>.jsonl` | metrics + trajectory analysis (schema in STATE_SPEC §6) |
| **Event** | fine-grained, many per step | `events.jsonl` (run) / stdout (server) | debugging: why a decision happened |
| **Manifest** | one per run | `manifest.json` | reproducibility |

## 2. TraceStep field semantics
- `schema_version` — 1; bumped only by a future ADR that changes this shape (ADR-009 §6).
- `run_id`, `system` — from the `RunContext` passed to the trace writer, not from task
  state (ADR-009 §6).
- `step` — 1-based index of the iteration, equal to `budget.steps_used + 1` at the time
  this step ran, i.e. the value `steps_used` will have once `persist` increments it
  (ADR-009 "Also"; `RetrievalState` has no separate `step` field); `action` — the action
  executed; `rule` — the controller rule that fired.
- `utility_scores` — G, C, R, L, U for every eligible non-STOP action (empty for
  baselines and fixed-policy steps).
- `candidate_ids` — everything returned by the action (post-filter for RERANK_EXPAND);
  `new_candidate_ids` — those not seen earlier in the task.
- `accepted_ids`, `rejected` (id → reason code), `evicted_ids`, `resolved_gap_ids`,
  `evidence_gain` — from the EvidenceUpdate (0.0 and empty for steps without candidates).
- `plan_events` — planner events produced during this step (INIT in step 1, TRANSFORM).
- `latency_ms` — per stage (`plan, select, execute, rerank, evidence, monitor, total`).
- `llm_calls`, `tokens` — consumed in this step; `budget_after` — cumulative.
- `stop_reason` — set only on the final step.
Baselines fill the same fields (e.g. one-shot: one step, action DENSE/SPARSE_HYBRID).
Vectors are never written to traces.

## 3. Event log
Event names (e.g. `cache.hit`, `llm.call`) are **not** config keys, even where they share a
prefix with a config section. Config keys are listed only in CONFIG_SCHEMA.md.

JSON lines with `ts, level, event, run_id, task_id, step, data`. Logger: `structlog`
configured in `engine/monitor/logging.py`.
| Event | data |
|---|---|
| `task.created` / `task.stopped` | budget / stop_reason, summary |
| `planner.accepted` / `planner.rejected` / `planner.fallback` | mode, attempt, reason, result |
| `cache.hit` / `cache.miss` / `cache.put` / `cache.evict` | query, similarity, match, entry_id |
| `controller.decision` | eligible, scores, chosen, rule |
| `retrieval.call` | mode, k, query, n_returned, latency_ms |
| `rerank.done` | per-candidate relevance/gap/novelty/redundancy/final |
| `evidence.accepted` / `evidence.rejected` / `evidence.evicted` | chunk_id, reason, scores |
| `gap.created` / `gap.resolved` / `gap.superseded` / `gap.exhausted` / `gap.verification` | gap_id, by, coverage |
| `monitor.verdict` | continue, stop_reason, stagnation_count, advanced_to |
| `llm.call` | role, model, tokens, cost_usd, cached, latency_ms, attempt |
| `error` | type, message, component, retryable |
Never logged: API keys, full prompts with keys, raw environment. Prompts may be logged at
DEBUG level only.

## 4. Run folder
`eval/runs/<run_id>/` with `run_id = YYYYMMDD-HHMMSS_<experiment>_<git7>`:
```
manifest.json
config.resolved.yaml
events.jsonl
b<B>/                       # one sub-folder per retrieval budget sub-run
  traces/<task_id>.jsonl
  bundles/<task_id>.json
  answers.jsonl             # {task_id, answer, generator tokens}
  per_task.csv              # metrics per task
  metrics.json              # aggregates
report/                     # tables and figures (eval/report.py)
```
`eval/runs/` is gitignored; final reported runs are archived (zip) and their run ids
listed in the thesis appendix.

## 5. Manifest (`manifest.json`)
`run_id, experiment, system, dataset, split, created_at, git_commit, git_dirty (must be
false for reportable runs), config_hash, cli_overrides, reportable (bool), python_version,
package_versions (from importlib.metadata), qdrant_server_version, collection,
split_hash, gold_map_hash, prompt_hashes {template: sha256}, models {role: model},
seeds, retrieval_budgets, repeats, cache_namespace, n_tasks, aborted (bool + reason),
totals {llm_calls, tokens, cost_usd, wall_time_s}`.

## 6. Tools
- `scripts/show_trace.py <run_id> <task_id>` — pretty-prints a trajectory (actions,
  queries, accepted evidence, gap changes, stop reason).
- `scripts/latency_profile.py` — runs each action type on dev-subset queries, writes
  median latency per action to `configs/latency_profile.yaml`.

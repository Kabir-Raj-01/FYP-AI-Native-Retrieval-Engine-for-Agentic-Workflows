# Observability: Traces, Events, Runs

> Status: Frozen for the TraceStep schema (ADR to change) · Owner: Vishal · Last updated: 2026-10-05
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
  The direct-path baselines record `execute` (search), `rerank` and `total` (ADR-012 A1).
  With `rerank.score_cache_enabled`, a cached score costs ~0 ms in `rerank`: quote latency
  from `scripts/latency_profile.py` or a single-worker cold run, not from quality runs.
- `llm_calls`, `tokens` — consumed in this step; `budget_after` — cumulative.
- `stop_reason` — set only on the final step.
Baselines fill the same fields (e.g. one-shot: one step, action DENSE/SPARSE_HYBRID).
Vectors are never written to traces.

**Writer** (`engine/monitor/interfaces.py:TraceWriter`, JSONL implementation
`engine/monitor/trace.py`, `factory.build_trace_writer(cfg, ctx, root)`), shared by the
engine's `persist` node and every baseline:
- Bound to one `RunContext`; producers create records with `writer.new_step(**fields)`,
  which stamps `run_id`/`system` (ADR-009 §6), and `write_step` refuses a step of another
  run or system.
- Per task, steps must arrive as 1, 2, 3, … (`step == previous + 1`, continuing an existing
  file); a duplicate or a gap is refused, which catches double writes from retries.
- `TraceSummary` (for the bundle) = `TraceWriter.summarize(steps)`: `steps`,
  `retrieval_calls`, `llm_calls`, `tokens`, `reranker_calls` from the last step's
  `budget_after`; `cache_hits` = steps with `cache_hit`; `total_latency_ms` = sum of
  `latency_ms["total"]`; `actions` = the action sequence.
- Unknown `schema_version` on read is an error, never a silent misread.

## 3. Event log
Event names (e.g. `cache.hit`, `llm.call`) are **not** config keys, even where they share a
prefix with a config section. Config keys are listed only in CONFIG_SCHEMA.md.

JSON lines with `ts, level, event, run_id, task_id, step, data`. Logger: `structlog`
configured in `engine/monitor/logging.py`. `run_id`, `task_id` and `step` are null when
unbound (e.g. the API server's direct path); everything else, including the logger's
`component`, goes under `data`. Producers bind the ids with
`engine.monitor.logging.bound_contextvars(...)`: the eval harness binds `run_id` and
`task_id` around each task (inside the worker thread — bindings are per thread), and the
code that runs a step binds `step` (one-shot baselines: 1).
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
`eval/runs/<run_id>/` with `run_id = YYYYMMDD-HHMMSS_<experiment>_<git7>` (`-2`, `-3`, …
appended when two runs start in the same second):
```
manifest.json
config.resolved.yaml
events.jsonl
b<B>/                       # one sub-folder per retrieval budget sub-run (TraceWriter root)
  traces/<task_id>.jsonl
  bundles/<task_id>.json
  answers.jsonl             # {task_id, answer, generator tokens}
  per_task.csv              # metrics per task
  metrics.json              # aggregates
report/                     # tables and figures (eval/report.py)
```
`eval/runs/` is gitignored; final reported runs are archived (zip) and their run ids
listed in the thesis appendix.
`<task_id>` in file names is percent-encoded (`multihop:0001` → `multihop%3A0001.jsonl`):
Windows forbids `:` in file names; `urllib.parse.unquote` restores the id. Files are UTF-8
with LF line endings. Run-id creation, `events.jsonl` (the event log written to a file),
`config.resolved.yaml` and the manifest belong to the eval harness (`eval/run.py`). The API
server writes its traces and bundles under `paths.runs_dir/server/` (`traces/`, `bundles/`)
with `run_id = "server"`, `system = "controller_on"`; `GET /v1/tasks/{id}/trace` reads them
back (decided 2026-10-05 with the `/tasks` routes, user-approved; owner to confirm).

## 5. Manifest (`manifest.json`)
`run_id, experiment, system, dataset, split, created_at, git_commit, git_dirty (must be
false for reportable runs), config_hash, cli_overrides, reportable (bool), python_version,
package_versions (from importlib.metadata), qdrant_server_version, collection,
split_hash, gold_map_hash, prompt_hashes {template: sha256}, models {role: model},
seeds, retrieval_budgets, repeats, cache_namespace, n_tasks, aborted (bool + reason),
totals {llm_calls, tokens, cost_usd, wall_time_s}`, plus `reportable_reasons` (why
`reportable` is false: dirty tree, `--set`, `--max-tasks`, a missing component such as the
reranker) and `notes` (`experiment.notes`). `models` holds `embedding` and `sparse` as
`name@revision`, and the reranker/planner/generator models (null until they exist).
Written by `eval/run.py` even when the run aborts.

## 6. Tools
- `scripts/show_trace.py <run_id> <task_id>` — pretty-prints a trajectory (actions,
  queries, accepted evidence, gap changes, stop reason).
- `scripts/latency_profile.py` — runs each action type on dev-subset queries, writes
  median latency per action to `configs/latency_profile.yaml` (format and definitions:
  ACTION_POLICY §4). Each query uses its own throwaway task in a temporary database, so the
  REUSE timing reflects a small per-task cache. The generated file is committed with the
  commit it was measured on (`git_commit` in the file).

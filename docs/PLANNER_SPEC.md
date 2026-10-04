# Query Planner Specification

> Status: Stable · Owner: Vishal · Last updated: 2026-10-04 · Related: LLM_CLIENT_SPEC.md, SECURITY.md, ADR-009

## 1. Role and hard limits
The planner turns the task goal into a small set of **gaps** (sub-questions) with
retrieval queries, and rewrites or splits a gap when retrieval stagnates.
- It is **not** a source of evidence. It never writes `evidence` or resolves gaps.
- It is **bounded**: at most `planner.max_subgoals` subgoals per call, at most
  `planner.max_gaps` live (open + resolved) gaps per task, at most `planner.max_attempts`
  LLM calls per planning invocation.
- It is invoked only (a) once at task start (`INIT`) and (b) when the controller chooses
  TRANSFORM. Never on every step.

Interface: `Planner.plan(state: RetrievalState, mode: PlanMode) -> PlanOutcome`. For
TRANSFORM, `mode` comes from `ActionDecision.plan_mode` — the **controller** decides
REWRITE vs SPLIT deterministically (using the same current-gap failed-query-count and
`max_gaps`-room conditions listed in §2 below), at selection time, per ADR-009 §3; the
planner itself never re-derives which mode it was asked for.

## 2. Modes
| Mode | When | Output constraint | Effect on state (applied by the calling node) |
|---|---|---|---|
| INIT | first iteration (`needs_initial_plan`) | 1..max_subgoals subgoals | each subgoal → new OPEN gap (`g0, g1, …`); `maybe_plan` then calls `TrajectoryMonitor.advance(state)` (ADR-009 §4) to set `current_gap_id = g0.gap_id` — `pending_query`/`current_need_type` then read directly from `g0` as computed properties (ADR-009 §4 correction), no separate write |
| REWRITE | controller set `plan_mode = REWRITE`: current gap has < 2 failed queries, or SPLIT would exceed max_gaps | exactly 1 subgoal | current gap keeps its description; only `gap.query` and `gap.need_type` are written — `pending_query`/`current_need_type` follow automatically (ADR-009 §4 correction) |
| SPLIT | controller set `plan_mode = SPLIT`: current gap has ≥ 2 failed queries and room for children | 2..max_subgoals subgoals | current gap → SUPERSEDED; children created with `parent_gap_id`; `execute` then calls `TrajectoryMonitor.advance(state)` to set current gap = first child |

"Failed query" = a query for this gap whose retrieval step had evidence gain below
`monitor.stagnation_threshold`.

A SUPERSEDED gap counts as resolved when all its children are RESOLVED, and as
exhausted when all children are closed and at least one is EXHAUSTED (computed by the
monitor).

## 3. Inputs given to the LLM (and nothing else)
- Goal text.
- For REWRITE/SPLIT: the current gap description, its previous queries (from
  `query_history`), and the list of other open gaps.
- Evidence summaries: up to `planner.max_evidence_summaries` accepted items, highest
  `final_score` first. `summary` = the item's text truncated to
  `planner.evidence_summary_chars` characters at the last sentence boundary (deterministic,
  no LLM summarisation — computed by the EvidenceManager when the item is accepted).
- Summaries are wrapped as untrusted data (SECURITY.md §3).

## 4. Output schema
The LLM is asked for JSON matching `PlanLLMOutput` (STATE_SPEC §3 — no `mode` field, no
length/count constraints in the schema, ADR-009 §10), via `LLMClient.structured(
role="planner", schema=PlanLLMOutput, ...)` at `llm.planner.temperature` (0.0).
`referenced_entities` must list every named entity, number, date, identifier or quoted
string used in any subgoal description or query. `Planner.plan` combines the parsed
`PlanLLMOutput` with the requested `mode` to build `PlanResult` after validation (§5).

## 5. Validation (`engine/planner/validator.py`)
Checks, in order; the first failure produces the reason code. This validator is also
where the count limits removed from the model schema (ADR-009 §10) are enforced, from
`planner.*` config keys.
| Code | Check |
|---|---|
| `SCHEMA` | Parses into `PlanLLMOutput` (LLMClient already validated JSON; re-validate here). |
| `COUNT` | Subgoal count within the mode's bounds (§2) and the gap cap (`planner.max_gaps`, computed against the current state, not a fixed schema bound). |
| `EMPTY_OR_DUPLICATE` | No empty query; no query identical (normalised, CACHE_SPEC §2) to a previous query for the same gap (REWRITE/SPLIT). |
| `UNGROUNDED_ENTITY` | Every item in `referenced_entities` fuzzy-matches (`rapidfuzz.partial_ratio` ≥ `planner.entity_match_threshold`, case-insensitive) text in the goal, the gap descriptions given as input, or the evidence summaries given as input. |
| `UNDECLARED_ENTITY` | A deterministic extractor (`engine/planner/entities.py`: capitalised multi-word spans not at sentence start, 4-digit years, numbers with units, quoted strings, alphanumeric IDs) run over all subgoal text finds a span that is neither in `referenced_entities` nor groundable as above. Catches models that omit entities from the list. |

`MODE_MISMATCH` is removed (ADR-009 §10): `PlanLLMOutput` carries no `mode` field for
the LLM to get wrong; `PlanResult.mode` is always exactly the mode the planner was
called with. On success, `PlanResult.rationale` is built by truncating
`PlanLLMOutput.rationale` to `planner.rationale_max_chars` — truncated, never rejected,
since it is logged only and never used for control.

## 6. Retry and fallback (exact)
```
for attempt in 1..planner.max_attempts:            # default 2
    result = llm.structured(planner prompt; on attempt ≥ 2 add the rejection reason
                            and a reminder to use only provided entities)
    if llm transport failure after its own retries → event REJECTED(reason="LLM_ERROR"); continue
    if validate(result) ok → event ACCEPTED(attempt); return
    else → event REJECTED(attempt, reason)
event FALLBACK; return deterministic fallback
```
Deterministic fallback (no LLM call):
| Mode | Fallback plan |
|---|---|
| INIT | one subgoal: description = goal, query = goal, need_type = heuristic(goal) |
| REWRITE / SPLIT | one subgoal (treated as REWRITE): description = current gap description; query = the gap description if it differs from all previous queries of this gap, else the goal; if both were already tried → the planner returns the fallback and the monitor marks the gap EXHAUSTED on the next stagnant step |

`heuristic(text)` → `lexical` if the extractor finds any entity/year/ID/quoted span and the
text is short (≤ 12 tokens) or the span is a quoted string/ID; `semantic` if the text starts
with why/how/what caused/explain and has no extracted span; else `mixed`.

LLM transport retries (network, 5xx, rate limit) are handled inside LLMClient and do
**not** consume planner attempts; every call that reaches the provider counts toward
`budget.llm_calls_used`.

## 7. Prompt templates
`engine/planner/prompts/{init,rewrite,split}.j2` (Jinja2), versioned in git. The
manifest records a hash of each template. A template change is an experimental-condition
change: re-run affected baselines and note it in `TASKS.md`/ADR if results are reported.
Template structure: system message = role, rules, JSON schema, untrusted-data warning;
user message = goal, gaps, previous queries, `<evidence>` block.

## 8. Tests
- Hallucinated entity → `UNGROUNDED_ENTITY`, retry happens, second attempt used.
- Entity present in a query but missing from `referenced_entities` → `UNDECLARED_ENTITY`.
- Two invalid responses → FALLBACK; exactly `max_attempts` LLM calls made.
- INIT with 5 subgoals when `max_subgoals=3` → `COUNT`.
- REWRITE returning a previously used query → `EMPTY_OR_DUPLICATE`.
- `heuristic()` on ≥ 6 labelled examples.
- Prompt rendering test: evidence summaries appear only inside the `<evidence>` block.

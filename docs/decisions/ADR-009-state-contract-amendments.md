# ADR-009: State contract amendments (pre-implementation review of `engine/state/models.py`)

> Status: Proposed · Date: 2026-09-28 · Author(s): Kabir Raj · Reviewer: Samay, Vishal

## Context
A documentation-only review of CLAUDE.md, ARCHITECTURE.md and STATE_SPEC.md (before
writing `engine/state/models.py`) found contradictions and gaps that make the loop as
specified unbuildable: fields with no writer that can produce them, a graph that has
nowhere to carry per-iteration data (cache hit, fresh candidates, scores) between nodes,
missing fields on component result models needed to satisfy the field-ownership matrix,
a Pydantic vector-exclusion behaviour that would silently drop vectors from persistence,
duplicate/derivable fields that can drift apart, and several unresolved questions about
REUSE, budget exhaustion and stagnation counting. This ADR is the single decision record
for all resulting changes to the state contract and the specs that describe it, so that
`engine/state/models.py` can be implemented against a consistent set of documents.

## Decision

### 1. Process
One amendment ADR (this one) followed by edits to the affected docs in the same PR.
"Implementation must match STATE_SPEC exactly" (STATE_SPEC.md header) is read as: match
after this ADR's changes are applied, and additions STATE_SPEC's prose already implies
(Budget helper methods, an invariant-checking function) are not contract changes needing
a further ADR — only changes to model shape, ownership or semantics need one.

### 2. Transient inter-node data: `GraphState`
LangGraph's state gains a wrapper:
```python
class IterationScratch(BaseModel):
    cache_hit: CacheHit | None = None
    candidates: list[Candidate] = []            # from execute
    scored: list[ScoredCandidate] = []           # from score_and_update (rerank output)
    evidence_update: EvidenceUpdate | None = None
    plan_outcome: PlanOutcome | None = None
    queued_cache_entries: list[CacheEntry] = []  # written by score_and_update, committed by persist

class GraphState(BaseModel):
    state: RetrievalState
    scratch: IterationScratch = IterationScratch()
```
`scratch` is never persisted and never appears in `TaskStore`, the API, or traces.
`persist` discards it (returns a fresh `IterationScratch()` for the next loop iteration,
or the graph simply ends). This is the only field LangGraph carries beyond
`RetrievalState`.

### 3. `ActionDecision.plan_mode`
`ActionDecision` gains `plan_mode: PlanMode | None = None`. When `action == TRANSFORM`,
the controller sets it deterministically from PLANNER_SPEC §2's REWRITE/SPLIT conditions
(current gap's failed-query count and remaining room under `planner.max_gaps`) — the
same rule the planner would otherwise have to re-derive. `execute` passes `plan_mode`
straight to `Planner.plan(state, plan_mode)`.

### 4. Ownership: component results carry full new values
- `MonitorVerdict` gains `stagnation_count: int`, `transforms_in_streak: int`,
  `action_gain_ema: dict[Action, float]`, `gap_status_changes: dict[str, GapStatus]`,
  `current_gap_id: str | None`. The `monitor` node applies these verbatim instead of
  recomputing them.
- `EvidenceUpdate` gains `gap_resolutions: dict[str, str]` (gap_id → evidence_id) and
  `item_changes: dict[str, ItemChange]` where
  `ItemChange = {pinned: bool | None, supports_gap_ids: list[str] | None}` for evidence
  items whose flags changed without being newly accepted this step.
- A monitor-owned `advance(state) -> AdvanceResult` function (not a graph node — a plain
  function in `engine/monitor/`, callable by any node) computes the next current gap:
  `AdvanceResult = {current_gap_id, reset_stagnation: bool}`. `current_subgoal`,
  `pending_query` and `current_need_type` are **not** part of `AdvanceResult`: they are
  computed properties derived from `current_gap_id` (§4a below), so setting it is the
  only write `advance()` needs to make. `maybe_plan` calls it after INIT to set the
  first current gap; `execute`'s SPLIT branch calls it after creating children;
  `monitor` calls it on gap resolution/exhaustion (ACTION_POLICY §6.3). This replaces
  the STATE_SPEC ownership table's implicit "TrajectoryMonitor sets current_gap_id" for
  the INIT/SPLIT cases, which no monitor step exists to run at that point.

### 4a. Correction: `current_subgoal`, `pending_query`, `current_need_type` are derived
This supersedes the version of this decision originally recorded under "Also" (below,
struck through). `current_subgoal`, `pending_query` and `current_need_type` are **not**
stored `RetrievalState` fields; they are `@computed_field @property` values derived from
`state.gaps[current_gap_id]` (`current_subgoal = gap.description`,
`pending_query = gap.query`, `current_need_type = gap.need_type`; all `""`/`MIXED` when
`current_gap_id` is `None`). A REWRITE writes `gap.query` and `gap.need_type` only,
never a duplicate top-level field — the two representations can no longer drift apart,
because there is only one representation. `@computed_field` keeps them present in a
plain `model_dump()` output (API responses, the debug state endpoint), matching prior
behaviour for external consumers even though nothing is stored under those names
anymore. STATE_SPEC §4's `RetrievalState` model, ownership matrix, and `AdvanceResult`;
ARCHITECTURE §4's node table; ACTION_POLICY §6.3; and PLANNER_SPEC §2 are updated
accordingly.

**Persistence correction (verified against Pydantic 2.13):** `RetrievalState` has
`extra="forbid"`, and `model_validate()` rejects a payload that contains a
`@computed_field`'s key, because a computed field has no corresponding `__init__`
parameter — it is output-only. So `dump_with_vectors(state)` (used for the TaskStore
write) must **exclude** `current_subgoal`, `pending_query`, `current_need_type` from its
output (`model_dump(context={"with_vectors": True}, exclude={"current_subgoal",
"pending_query", "current_need_type"})`), and `load_with_vectors` validates that reduced
JSON back into a `RetrievalState` — the three properties recompute themselves from the
restored `gaps`/`current_gap_id` on load, so nothing is lost. Only `dump_with_vectors` /
`load_with_vectors` (persistence) are affected; a plain `model_dump()` (API responses,
traces, `EvidenceBundle`) is unaffected because those consumers only ever *read* the
dump, never round-trip it back through `model_validate`. TESTING.md §4 gains a required
round-trip test for this.
- The field-ownership matrix gains a row: `status = FAILED, stop_reason = ERROR` is set
  by the `Engine` facade on the non-retryable error path (ARCHITECTURE §8), outside any
  graph node, immediately followed by a `TaskStore.save` of just those two fields (not a
  full `persist` run — no TraceStep is written for this transition, only an `error`
  event, per OBSERVABILITY §3).
- `LastResult` moves from `execute`'s ownership to `score_and_update`'s: it needs
  `top_relevances`, which only exist after reranking. `execute` writes `scratch.candidates`
  and `retrieval_history`/`seen_candidate_ids`; `score_and_update` writes `last_result`
  (state) from `scratch.scored`.

### 5. Invariants
No STATE_SPEC invariant from §5 is enforced as a Pydantic `field_validator` /
`model_validator` on `RetrievalState`: several are legitimately false mid-iteration
(e.g. invariant 6, between `execute`'s SPLIT and `monitor`'s `advance()` call), and
LangGraph reconstructs the Pydantic model at every node boundary, so a model-level
validator would raise on valid intermediate states. Instead, `persist` calls
`check_invariants(prev: RetrievalState, new: RetrievalState, cfg: EngineConfig) -> None`
(raises `StateInvariantError`, a programming-error signal, not a domain error) once per
completed iteration, before saving. `RetrievalState.model_config` keeps
`validate_assignment=True` (catches type/range errors on any direct field set during
development/tests) — this does not conflict with invariant 9 ("mutated only via returned
partial updates in nodes"), which is a process rule enforced by code review and the
import-boundary test, not by Pydantic.

### 6. `schema_version`; no `run_id`/`system` in state; no external_ref; no per-task filters
- `RetrievalState`, `CacheEntry` and `TraceStep` each gain `schema_version: int = 1`, so
  a later ADR that changes shape can migrate or reject old rows instead of silently
  misreading them.
- `run_id` and `system` are **not** state fields. `Engine` and the trace writer take a
  `RunContext = {run_id: str, system: str}` as a call parameter (constructed by the
  caller — `eval/run.py` for experiments, a fixed context for the API server), and stamp
  it onto each `TraceStep` and `EvidenceBundle` at write time.
- No `external_ref` field: `task_id` is already caller-suppliable
  (`CreateTaskRequest.task_id`); a benchmark id is just used as `task_id` directly.
- No per-task search filters field: nothing in the current specs issues filtered
  searches from within the controlled loop; add this only when a concrete need appears.

### 7. `EvidenceItem.metadata`; no partial bundles
`EvidenceItem` gains `metadata: dict[str, str | int | None] = {}`, copied from the
source `Candidate.metadata` at acceptance time — EVAL_PROTOCOL §3 requires `(title,
date)` in the generator's per-item citation format, and only `EvidenceItem` (not
`Candidate`) survives into the `EvidenceBundle`. Partial bundles for an in-flight task
remain out of scope for this ADR; backlogged behind a future cancel endpoint
(AGENT_INTEGRATION_SPEC / API_CONTRACT, not decided here).

### 8. `Budget`
```python
class Budget(BaseModel):
    max_steps: int = Field(ge=1)
    max_retrieval_calls: int = Field(ge=1)
    max_llm_calls: int = Field(ge=0)          # 0 = no-planner/no-LLM ablation
    max_tokens: int = Field(ge=1)
    steps_used: int = 0
    retrieval_calls_used: int = 0
    llm_calls_used: int = 0
    tokens_used: int = 0
    reranker_calls_used: int = 0

    def consume(self, *, steps=0, retrieval_calls=0, llm_calls=0, tokens=0,
                reranker_calls=0) -> "Budget":
        """Returns a new Budget with the given deltas added. Never mutates self."""

    def pressure(self) -> float:
        """Max utilisation ratio over caps with max_* > 0; caps that are 0 are
        skipped (a 0 cap means the resource is unused this run, not maximally
        pressured)."""
```
`max_steps ≥ 1` and `max_retrieval_calls ≥ 1` (a task must be able to take at least one
step and, per ACTION_POLICY §2 step 8, always retrieves at least once before STOP on low
utility is eligible). `max_llm_calls = 0` is valid (disables TRANSFORM and INIT planning
falls back deterministically every time — ablation.planner=false already implies this in
practice, but a config that sets the cap to 0 directly must not crash `pressure()`).

### 9. REUSE semantics
- `RetrievalRecord` and `LastResult` for a REUSE step use `mode = entry.mode`,
  `k = len(entry.candidates)`, `query = pending_query` (not the entry's original
  normalised query — the reranker and gap-coverage signals score against the *current*
  pending query, per CACHE_SPEC §5's "feeds through the normal rerank → evidence path").
  `LastResult.depth = len(entry.candidates)` (same value as `k`) — there is no
  backend-search depth for a cache hit, so it is treated as already-at-that-depth,
  consistent with how `current_depth`/`last_result.depth` gate RERANK_EXPAND.
- `EvidenceItem.source_action = Action.REUSE` for items accepted from a REUSE step.
- `RetrievalRecord.cache_hit` means "this record's candidates came from the cache"
  (true only for REUSE records). A `Cache.lookup` hit that exists but was **not** chosen
  by the controller (e.g. utility favoured a fresh search) is not reflected in any
  `RetrievalRecord` — it is logged only as the `cache.hit` event (OBSERVABILITY §3),
  which already carries `similarity`/`match`/`entry_id`.
- Steps with no candidates (TRANSFORM, STOP) set `last_update = None`.
  `last_result` is never reset except by `advance()` on gap advancement (§4 above) —
  in particular a TRANSFORM or STOP step leaves the prior `last_result` intact, since
  RERANK_EXPAND on the next step still needs it.
- Cache writes: `score_and_update` appends fresh `CacheEntry` objects to
  `scratch.queued_cache_entries` rather than calling `Cache.put` directly. `persist`
  passes them to `TaskStore.save`'s `cache_entries` parameter (corrected signature, "Also"
  below), which commits the state save and every queued entry (`cache_entries` table
  inserts/replacements) in one atomic transaction. If the iteration fails before
  `persist`, `save` is never called and no cache entry is written — this closes the
  "orphan cache entry after a failed step" gap identified in the original review (a retry
  can no longer REUSE a result from a step that was never actually saved).

### 10. LLM output schema split
A new model, `PlanLLMOutput` (lives in `engine/state/models.py` alongside the other
shared models, since `engine/llm` and `engine/planner` both need it):
```python
class PlanLLMOutput(BaseModel):
    subgoals: list[SubgoalPlan] = Field(min_length=1)   # no upper bound in the schema
    referenced_entities: list[str] = []
    rationale: str = ""                                  # no max_length in the schema
```
This is what `LLMClient.structured(role="planner", ..., schema=PlanLLMOutput)` asks the
provider for — no `mode` field (the planner already knows the mode it requested; asking
the LLM to echo it back only creates a `MODE_MISMATCH` failure mode for no benefit) and
no length/count constraints baked into the schema (CLAUDE.md rule 4: no literal
thresholds in code — a Pydantic `Field(max_length=...)` on a model definition is such a
literal). `Planner.plan` builds `PlanResult` from `PlanLLMOutput` plus the requested
`mode`, and the validator (`engine/planner/validator.py`, PLANNER_SPEC §5) enforces
subgoal-count bounds and the `max_gaps` cap by reading `planner.*` config, not from
model field constraints. `SubgoalPlan.description`/`query` keep no length constraints in
the model either; if truncation/rejection of long text is wanted, it is a config-driven
validator check, not a schema constraint — `PlanResult.rationale` in particular is
**truncated** to `planner.rationale_max_chars` (new config key) by the validator, never
rejected, since it is logged only and never used for control.
`LLMResult[T]` and `Message` are defined in `engine/llm/interfaces.py` (already the case
per LLM_CLIENT_SPEC §3), not in `engine/state/models.py` — they are LLM-transport types,
not shared task-state models, and nothing outside `engine/llm` and `engine/planner`
needs them.

### 11. `MonitorVerdict.continue_` → `should_continue`
Renamed; the `Field(alias="continue")` is dropped. `continue` is a reserved word, so the
alias could never be supplied as a keyword argument to the constructor, making every
non-alias construction site awkward for no reason (JSON serialisation compatibility was
the only reason for the alias, and nothing external parses this model as JSON).

### 12. `budget.steps_used` timing vs. the monitor's `MAX_STEPS` check
`persist` increments `budget.steps_used` (and `state.step` is dropped — see "Also",
below) **after** `monitor` runs. ACTION_POLICY §6.4's stop check is therefore written
against the value `steps_used` **will** have once this iteration is persisted, not its
current value:
```
steps_used + 1 >= max_steps  →  MAX_STEPS
```
(ACTION_POLICY §6.4 is amended to this exact expression.) This makes the semantics "the
iteration in progress counts towards the cap it is checking" — the monitor, running near
the end of the very iteration that is about to make `steps_used` reach `max_steps`, is
the thing that stops the task, and that iteration is the last real (non-wasted) one: with
`max_steps = 3`, the engine runs exactly 3 real iterations and ends with
`steps_used == 3`, not 2 real iterations plus a wasted STOP-only third.
`Controller.select_action`'s own guard (ACTION_POLICY §2 step 1,
`budget.steps_used >= max_steps`) is **not** changed to `max_steps - 1`: it stays
exactly as written, checking the value already persisted from the *previous* iteration.
In normal operation the monitor's fix above means the task is always already `STOPPED`
by the time `steps_used == max_steps`, so `select_action` is never invoked again — the
controller's guard is defensive-only, covering a caller that invokes `Engine.step` on a
task the monitor should already have stopped (e.g. a race, or a bug elsewhere). It must
never fire in the normal single-caller flow; `tests/unit/test_state_invariants.py` and
the new `max_steps` test (TESTING.md §4, below) both assert this by construction.

## Also (smaller changes bundled into the same PR)
- Remove `current_depth` from `RetrievalState`; use `state.last_result.depth` (0/absent
  when there is no last result). Remove `state.step`; `budget.steps_used` is the sole
  step counter, incremented by `persist`. Within one iteration, any record that needs
  "the step number this iteration will end up as" uses `n = budget.steps_used + 1`
  (`TraceStep.step`, `QueryRecord.step`, `RetrievalRecord.step`, `PlanEvent.step`,
  `EvidenceItem.added_at_step`, gap `created_at_step`/`closed_at_step` all use this `n`,
  computed once per iteration and threaded through — not `steps_used` itself, which is
  still the pre-iteration value when they're written).
- ~~`current_subgoal`, `pending_query`, `current_need_type` are derived from the current
  gap (`gaps[current_gap_id]`) rather than duplicated fields that can drift: ...
  `RetrievalState` keeps `pending_query` and `current_need_type` as plain fields for now
  ... documented as *authoritative only when written together with the gap update that
  produces them*~~ — **superseded by §4a**: they are `@computed_field` properties, not
  stored fields at all, so there is no "keep in sync" discipline needed. See §4a.
- `QueryRecord.source` drops the `"gap_default"` value (only `"planner"` and
  `"fallback"` remain — every query recorded in `query_history` is now traceable to a
  specific `PlanOutcome`, including the deterministic fallback plans from PLANNER_SPEC
  §6, which already produce a `PlanResult` and so a `QueryRecord` with source
  `"fallback"`; there is no path that writes a query to history without going through
  the planner).
- `query_history` records queries the **planner proposed** (one per accepted subgoal,
  written by `maybe_plan`/`execute`'s TRANSFORM branch); `retrieval_history` records
  queries actually **issued** to the backend or cache (written by `execute`). These can
  differ in count (a REWRITE proposes one query; it may or may not be issued this step
  depending on the controller's next decision) and this is intentional, not a bug.
- `Reranker.rerank(query, candidates, state: RetrievalState | None) -> list[ScoredCandidate]`.
  `state=None` means relevance-only mode (no gap coverage, no novelty against E_t, no
  MMR — `final = relevance`, matching RERANK_SPEC §3's `ablation.state_aware_rerank`
  behaviour) — used by `Engine.retrieve_direct` and the baselines, which have no task
  state.
- `RetrieverAdapter.fetch()` is dropped from the interface until a concrete caller needs
  it (nothing in ARCHITECTURE, RETRIEVAL_SPEC's own test list aside, currently calls it;
  its own integration test was the only user).
- Concurrency: an in-process per-`task_id` lock inside `Engine` (`threading.Lock` per
  task, held for the duration of one `step`/`run` call) is added **in addition to** the
  existing `version`-based optimistic check in `TaskStore.save`. The lock is a
  single-process optimisation (correct only because ARCHITECTURE §3 already scopes this
  service to one process per deployment — the eval harness's thread pool and the API
  server's threadpool are both in-process); the version check remains the correctness
  backstop for any future multi-process deployment and for detecting a lock bug.
  ~~`TaskStore.save` signature becomes `save(state: RetrievalState, expected_version: int) -> None`~~
  — **corrected (§9)**: `save` also takes the queued cache entries, so the state save and
  the cache writes commit atomically in one call: `save(state: RetrievalState,
  expected_version: int, cache_entries: Sequence[CacheEntry] = ()) -> None` (raises
  `TaskConflict` if the stored version does not equal `expected_version`, in which case
  neither the state nor any cache entry is written). `persist` sets
  `new_state.version = expected_version + 1` in memory *before* calling `save`, so the
  state object returned to the caller (and used as the next loop's input) already
  matches what is now in the database — no read-back required. §9's description of
  `score_and_update` queuing entries and `persist` committing them "in the same SQLite
  transaction as the state save" means via this one call, not two separate writes
  sharing a manually-managed transaction.
- Vector persistence: `Candidate.vector`, `EvidenceItem.vector` and `CacheEntry`'s
  per-candidate vectors use a `field_serializer` keyed off a `SerializationContext`
  (`with_vectors: bool`, default `False`) passed via `model_dump(context=...)`, not
  `Field(exclude=True)` (which unconditionally drops the field regardless of `include`).
  Default dumps (API responses, traces, `EvidenceBundle`) omit vectors; `TaskStore.save`
  and `Cache.put`'s underlying write call `model_dump(context={"with_vectors": True})`.
  This lives in a new `engine/state/serialization.py` module documented in CLAUDE.md's
  repo layout, exporting `dump_with_vectors(model) -> dict` and
  `load_with_vectors(cls, data) -> model` helpers so call sites never construct the
  context dict by hand.
- `StrEnum` (not `(str, Enum)`) for every enum in §1, `extra="forbid"` on every model,
  UTC-aware `datetime` fields, and explicit `= None` defaults on every `X | None` field
  without one (Pydantic v2 treats a bare `X | None` annotation as required unless a
  default is given).
- `engine/errors.py` and `engine/state/serialization.py` are added to CLAUDE.md's repo
  layout listing under `engine/`.
- Rename the controller's per-action signal variables in code (`gain`, `cost`,
  `redundancy_signal`, `latency_signal` — not `G`, `C`, `R`, `L`) to avoid colliding with
  S_t's `G` (goal) and `R_t` (retrieval history). The formula `U = α·gain − β·cost −
  γ·redundancy_signal − δ·latency_signal` is unchanged; only the identifiers in
  `engine/controller/signals.py` and `ActionScores`'s field *names* stay `G`/`C`/`R`/`L`
  in the model only where STATE_SPEC already froze that shape (`ActionScores`) — code
  that computes them uses the longer names, then constructs `ActionScores(G=gain, C=cost,
  R=redundancy_signal, L=latency_signal, U=...)` at the boundary.

## Alternatives considered
- **Per-node graph state instead of `GraphState`/scratch** — passing `cache_hit`,
  `candidates`, etc. as extra top-level fields on `RetrievalState` itself, since
  LangGraph state is just the Pydantic model. Rejected: it would put transient,
  per-iteration data in the same model that TaskStore persists and the API serialises,
  forcing every one of those consumers to special-case "ignore these fields" instead of
  the wrapper simply not reaching them.
- **STOP always consumes a step; fix via controller guard at `max_steps - 1`** (the
  original plan before this ADR) — rejected per the user's explicit correction: it burns
  the task's last allowed step on a no-op STOP instead of a real action, undercounting
  useful work by one for every task that runs to the step cap.
- **Keep `MonitorVerdict.continue_` with the `Field(alias="continue")`** — rejected
  because the alias cannot be used as a Python keyword argument, so every construction
  site either fights the alias or uses `model_construct`/dict unpacking, for a
  compatibility need (JSON key literally named `continue`) nothing in this codebase has.
- **Validate STATE_SPEC invariants as Pydantic validators** — rejected; several are
  false mid-iteration by design (see Decision §5), and LangGraph's per-node state
  reconstruction means a `model_validator(mode="after")` would run on every node
  boundary, not just at `persist`.

## Consequences
Implementing `engine/state/models.py` can now proceed against a single consistent
contract. Docs updated in this PR: STATE_SPEC.md, ARCHITECTURE.md, CLAUDE.md,
ACTION_POLICY.md, PLANNER_SPEC.md, CACHE_SPEC.md, RETRIEVAL_SPEC.md, RERANK_SPEC.md,
LLM_CLIENT_SPEC.md, API_CONTRACT.md, CONFIG_SCHEMA.md, OBSERVABILITY.md, TESTING.md,
GLOSSARY.md. A new config key `planner.rationale_max_chars` is added to
`configs/base.yaml`/CONFIG_SCHEMA.md in the same PR as the code that reads it (not this
docs-only PR). Two new required test categories (TESTING.md §4): the `max_steps` exact
iteration-count test, and an explicit "no orphan cache entries after a failed step" test.

## Evidence
None — this is a design review, not a calibrated/measured decision. No dev-split numbers
apply.

## Reversal condition
Revisit the `GraphState`/`IterationScratch` split if implementing `engine/graph/` finds
that LangGraph's partial-update-return model doesn't compose cleanly with a
nested-and-discarded scratch object (e.g. if partial updates must be flat top-level
fields). Revisit the `steps_used + 1 >= max_steps` monitor check if a future controller
change makes the monitor's post-execute check not the last opportunity to stop
before persist (e.g. if a node is added after monitor).

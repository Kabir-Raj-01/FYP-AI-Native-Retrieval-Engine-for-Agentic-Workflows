# State and Shared Data Models

> Status: Frozen (contract — change only via ADR) · Owner: Samay · Last updated: 2026-10-04
> Implementation: `engine/state/models.py`. This document and that file must match exactly
> (as amended by ADR-009, and ADR-012 for `DirectResult`). Vector persistence uses `engine/state/serialization.py`
> (ADR-009 "Also"), not `Field(exclude=True)`. ADR-009 is still Proposed: it must be
> Accepted by its reviewers before `engine/state/models.py` is merged (TASKS.md Week 7).

S_t = (G, g_t, E_t, Q_t, R_t, B_t) plus control bookkeeping.
Mapping: G = `goal`, g_t = `current_subgoal` (+ `current_gap_id`), E_t = `evidence`,
Q_t = `query_history`, R_t = `retrieval_history`, B_t = `budget`.

## 1. Enums
All enums are `StrEnum` (not `(str, Enum)`) — ADR-009 "Also".
```python
from enum import StrEnum

class NeedType(StrEnum):
    SEMANTIC = "semantic"; LEXICAL = "lexical"; MIXED = "mixed"

class RetrievalMode(StrEnum):
    DENSE = "dense"; SPARSE = "sparse"; HYBRID = "hybrid"

class Action(StrEnum):
    REUSE = "REUSE"; DENSE = "DENSE"; SPARSE_HYBRID = "SPARSE_HYBRID"
    TRANSFORM = "TRANSFORM"; RERANK_EXPAND = "RERANK_EXPAND"; STOP = "STOP"

class GapStatus(StrEnum):
    OPEN = "open"            # still needed
    RESOLVED = "resolved"    # covered by accepted evidence
    SUPERSEDED = "superseded"  # replaced by child gaps after a SPLIT
    EXHAUSTED = "exhausted"  # given up after stagnation; task continues with other gaps

class TaskStatus(StrEnum):
    ACTIVE = "active"; STOPPED = "stopped"; FAILED = "failed"

class StopReason(StrEnum):
    ALL_GAPS_RESOLVED = "ALL_GAPS_RESOLVED"
    STAGNATION = "STAGNATION"                  # remaining gaps exhausted
    LOW_MARGINAL_UTILITY = "LOW_MARGINAL_UTILITY"
    MAX_STEPS = "MAX_STEPS"
    MAX_RETRIEVAL_CALLS = "MAX_RETRIEVAL_CALLS"
    MAX_LLM_CALLS = "MAX_LLM_CALLS"
    MAX_TOKENS = "MAX_TOKENS"
    ONE_SHOT_COMPLETE = "ONE_SHOT_COMPLETE"            # baselines only
    FIXED_ITERATIONS_COMPLETE = "FIXED_ITERATIONS_COMPLETE"  # baseline only
    ERROR = "ERROR"

class PlanMode(StrEnum):
    INIT = "init"; REWRITE = "rewrite"; SPLIT = "split"

class PlanStatus(StrEnum):
    ACCEPTED = "accepted"; REJECTED = "rejected"; FALLBACK = "fallback"
```

## 2. Retrieval and evidence models
Vector fields below (`vector`, and `CacheEntry.query_vector`) are omitted from
`model_dump()` by default via a `field_serializer` gated on a serialization context, not
`Field(exclude=True)` (ADR-009 "Also"); `engine/state/serialization.py`'s
`dump_with_vectors` / `load_with_vectors` opt in for TaskStore and Cache writes. On
`RetrievalState` specifically, `dump_with_vectors` also **excludes** the three
`@computed_field` properties (`current_subgoal`, `pending_query`, `current_need_type`,
§4) — with `extra="forbid"`, `model_validate` rejects a payload containing computed-field
keys, since they are not settable `__init__` parameters (verified against Pydantic
2.13), so `load_with_vectors` must be given JSON without them (ADR-009 §4a correction).
```python
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, computed_field

class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str                 # stable ID from DATA.md
    doc_id: str
    text: str
    retrieval_score: float        # raw backend score (not comparable across modes)
    rank: int                     # 0-based rank in the backend result
    mode: RetrievalMode
    metadata: dict[str, str | int | None] = {}   # title, source, published_at, ...
    vector: list[float] | None = None  # dense vector, for similarity signals; see note above

class ScoredCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate: Candidate
    relevance: float        # [0,1] cross-encoder
    gap_coverage: float     # [0,1] max over open gaps
    novelty: float          # [0,1] vs E_t
    redundancy: float       # [0,1] vs higher-ranked selected candidates (MMR)
    final_score: float      # RERANK_SPEC formula
    covered_gap_id: str | None = None  # gap with the max coverage

class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str        # == chunk_id (one evidence item per chunk)
    chunk_id: str
    doc_id: str
    text: str
    summary: str            # deterministic truncation, see PLANNER_SPEC §3
    relevance: float
    novelty: float
    final_score: float
    source_action: Action   # ADR-009 §9: REUSE for items accepted from a REUSE step
    query: str
    metadata: dict[str, str | int | None] = {}  # copied from Candidate.metadata at acceptance (ADR-009 §7); title/date for EVAL_PROTOCOL §3
    added_at_step: int
    supports_gap_ids: list[str] = []
    pinned: bool = False    # True if it resolved a gap → never evicted
    vector: list[float] | None = None

class EvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gap_id: str             # "g0", "g1", ... unique within task
    description: str        # natural-language information need
    query: str              # planner-proposed retrieval query for this gap
    need_type: NeedType
    status: GapStatus = GapStatus.OPEN
    parent_gap_id: str | None = None   # set for children of a SPLIT
    resolved_by: str | None = None     # evidence_id
    created_at_step: int
    closed_at_step: int | None = None
```

## 3. History, budget, planning, decisions
Every model below has `model_config = ConfigDict(extra="forbid")` (omitted from the
snippets for brevity). `step` fields hold `n = budget.steps_used + 1` at the time the
record is created within an iteration (ADR-009 "Also") — the step number this iteration
will have once `persist` runs, not `steps_used` itself.
```python
class QueryRecord(BaseModel):
    step: int
    gap_id: str
    query: str
    need_type: NeedType
    source: str             # "planner" | "fallback" — every query is planner-proposed
                             # (including deterministic fallback plans); "gap_default" removed (ADR-009)

class RetrievalRecord(BaseModel):
    step: int
    action: Action          # DENSE | SPARSE_HYBRID | RERANK_EXPAND | REUSE
    mode: RetrievalMode | None = None
    query: str
    k: int
    candidate_ids: list[str]
    new_candidate_ids: list[str]   # not seen earlier in this task
    cache_hit: bool          # True iff this record's candidates came from the cache (REUSE only);
                              # a lookup hit that was not chosen is logged only as event cache.hit (ADR-009 §9)
    latency_ms: float

class LastResult(BaseModel):
    """Compact view of the most recent candidate-producing action (for RERANK_EXPAND).
    Written by score_and_update (needs top_relevances, which only exist post-rerank),
    not execute (ADR-009 §4). Never reset except by monitor.advance() on gap
    advancement (ADR-009 §9) — a TRANSFORM/STOP step leaves the prior value intact."""
    step: int
    query: str               # ADR-009 §9: for REUSE, the current pending query, not the entry's original query
    mode: RetrievalMode | None = None
    depth: int                # ADR-009 §9: for REUSE, len(entry.candidates) (no backend-search depth applies)
    candidate_ids: list[str]
    top_relevances: list[float]    # relevance of top rerank.top_n_output, descending

class Budget(BaseModel):
    max_steps: int = Field(ge=1)
    max_retrieval_calls: int = Field(ge=1)
    max_llm_calls: int = Field(ge=0)   # 0 = no-LLM ablation
    max_tokens: int = Field(ge=1)
    steps_used: int = 0
    retrieval_calls_used: int = 0
    llm_calls_used: int = 0
    tokens_used: int = 0
    reranker_calls_used: int = 0   # tracked, not capped

    def consume(self, *, steps: int = 0, retrieval_calls: int = 0, llm_calls: int = 0,
                tokens: int = 0, reranker_calls: int = 0) -> "Budget":
        """Returns a new Budget with the given deltas added; never mutates self."""

    def pressure(self) -> float:
        """Max utilisation ratio, in [0,1], over caps with max_* > 0; a zero cap is
        skipped rather than causing a division by zero (ADR-009 §8)."""

class PlanEvent(BaseModel):
    step: int
    mode: PlanMode
    attempt: int
    status: PlanStatus
    reason: str | None = None      # reason code, PLANNER_SPEC §5

class SubgoalPlan(BaseModel):
    description: str
    query: str
    need_type: NeedType
    # No Field(min_length=..., max_length=...) — literal length limits are a config
    # rule, not a schema constraint (CLAUDE.md rule 4). Enforced by the planner
    # validator from `planner.*` config keys (PLANNER_SPEC §5), not here.

class PlanLLMOutput(BaseModel):
    """What is actually asked of the LLM (LLMClient.structured(schema=PlanLLMOutput)).
    No `mode` (the planner already knows the mode it requested) and no length/count
    constraints (ADR-009 §10)."""
    subgoals: list[SubgoalPlan] = Field(min_length=1)
    referenced_entities: list[str] = []
    rationale: str = ""

class PlanResult(BaseModel):
    """Built by Planner.plan from PlanLLMOutput + the requested mode, after validation.
    rationale is truncated (never rejected) to planner.rationale_max_chars by the
    validator (ADR-009 §10)."""
    mode: PlanMode
    subgoals: list[SubgoalPlan] = Field(min_length=1)   # bounds checked by validator, PLANNER_SPEC §5
    referenced_entities: list[str] = []
    rationale: str = ""     # logged only, never used for control

class PlanOutcome(BaseModel):
    result: PlanResult          # accepted plan or the deterministic fallback
    events: list[PlanEvent]
    llm_calls: int
    tokens: int

class ActionScores(BaseModel):
    G: float; C: float; R: float; L: float; U: float
    # Field names stay G/C/R/L/U (this shape is frozen); the code that computes them
    # uses longer identifiers (gain, cost, redundancy_signal, latency_signal) to avoid
    # colliding with S_t's G (goal) and R_t (retrieval history) — ADR-009 "Also".

class ActionDecision(BaseModel):
    action: Action
    eligible: list[Action]
    scores: dict[Action, ActionScores]    # eligible non-STOP actions only
    stop_reason: StopReason | None = None
    plan_mode: PlanMode | None = None     # set iff action == TRANSFORM (ADR-009 §3)
    rule: str                             # which rule fired, e.g. "utility_argmax", "budget_guard"

class CacheEntry(BaseModel):
    schema_version: int = 1
    task_id: str
    entry_id: str
    normalized_query: str
    query_vector: list[float]
    need_type: NeedType
    mode: RetrievalMode
    candidates: list[Candidate]           # stored with vectors
    accepted_ids: list[str]
    created_at_step: int

class CacheHit(BaseModel):
    entry: CacheEntry
    similarity: float                     # 1.0 for exact normalized match
    match: str                            # "exact" | "semantic"

class ItemChange(BaseModel):
    pinned: bool | None = None
    supports_gap_ids: list[str] | None = None

class EvidenceUpdate(BaseModel):
    accepted: list[EvidenceItem]
    rejected: dict[str, str]              # chunk_id -> reason code
    evicted: list[str]
    resolved_gap_ids: list[str]
    gap_resolutions: dict[str, str] = {}  # gap_id -> evidence_id (ADR-009 §4)
    item_changes: dict[str, ItemChange] = {}  # evidence_id -> changed flags, for items
                                               # not newly accepted this step (ADR-009 §4)
    evidence_gain: float                  # [0,1]

class MonitorVerdict(BaseModel):
    should_continue: bool                 # renamed from continue_ (ADR-009 §11); no alias
    stop_reason: StopReason | None = None
    advanced_to_gap: str | None = None
    exhausted_gap: str | None = None
    stagnation_count: int                 # ADR-009 §4: full new values, applied verbatim
    transforms_in_streak: int
    action_gain_ema: dict[Action, float]
    gap_status_changes: dict[str, GapStatus] = {}
    current_gap_id: str | None = None
    note: str = ""

class AdvanceResult(BaseModel):
    """Return type of the monitor-owned advance() function (ADR-009 §4) — not a graph
    node; a plain function in engine/monitor/, callable by maybe_plan (after INIT),
    execute (after a SPLIT) and monitor (on gap resolution/exhaustion). Only
    current_gap_id is applied to state: current_subgoal, pending_query and
    current_need_type are computed fields derived from the gap it names (ADR-009 §4
    correction), so setting current_gap_id alone is sufficient — there is nothing else
    to write."""
    current_gap_id: str | None
    reset_stagnation: bool
```

## 4. RetrievalState
```python
class RetrievalState(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    schema_version: int = 1
    task_id: str
    version: int = 0
    status: TaskStatus = TaskStatus.ACTIVE
    stop_reason: StopReason | None = None
    # No `step` field — budget.steps_used is the sole step counter (ADR-009 "Also").

    goal: str                                        # G (immutable)
    gaps: list[EvidenceGap] = []
    current_gap_id: str | None = None
    # current_subgoal, pending_query, current_need_type are NOT stored fields — they are
    # computed properties derived from `gaps[current_gap_id]` (ADR-009 §4 correction: a
    # REWRITE writes gap.query/gap.need_type only, never a separate top-level field, so
    # the two can never drift apart). @computed_field so they still appear in a plain
    # model_dump()/API responses/the debug state endpoint exactly as before — but
    # dump_with_vectors (persistence) excludes them, and load_with_vectors is given JSON
    # without them, because a computed-field key in the payload makes model_validate
    # fail under extra="forbid" (ADR-009 §4a correction).
    needs_initial_plan: bool = True

    evidence: list[EvidenceItem] = []                # E_t
    query_history: list[QueryRecord] = []            # Q_t: queries the planner proposed
    retrieval_history: list[RetrievalRecord] = []    # R_t: queries actually issued
    budget: Budget                                   # B_t

    seen_candidate_ids: list[str] = []               # every chunk returned so far
    last_result: LastResult | None = None
    # No `current_depth` field — use last_result.depth (0/absent when no last_result).
    stagnation_count: int = 0
    transforms_in_streak: int = 0
    action_gain_ema: dict[Action, float] = {}        # missing key → controller.prior_action_gain
    last_decision: ActionDecision | None = None
    last_update: EvidenceUpdate | None = None        # None on steps without candidates (ADR-009 §9)
    plan_events: list[PlanEvent] = []

    created_at: datetime                              # UTC-aware
    updated_at: datetime                               # UTC-aware

    @property
    def current_gap(self) -> EvidenceGap | None:
        return next((g for g in self.gaps if g.gap_id == self.current_gap_id), None)

    @computed_field
    @property
    def current_subgoal(self) -> str:                # g_t = current gap's description
        g = self.current_gap
        return g.description if g else ""

    @computed_field
    @property
    def pending_query(self) -> str:                  # current gap's query
        g = self.current_gap
        return g.query if g else ""

    @computed_field
    @property
    def current_need_type(self) -> NeedType:          # current gap's need_type
        g = self.current_gap
        return g.need_type if g else NeedType.MIXED
```

### `GraphState` (LangGraph state; not persisted)
```python
class IterationScratch(BaseModel):
    """Transient, per-iteration data passed between graph nodes. Never persisted;
    discarded after persist. ADR-009 §2."""
    cache_hit: CacheHit | None = None
    candidates: list[Candidate] = []             # from execute
    scored: list[ScoredCandidate] = []            # from score_and_update (rerank output)
    evidence_update: EvidenceUpdate | None = None
    plan_outcome: PlanOutcome | None = None
    queued_cache_entries: list[CacheEntry] = []   # written by score_and_update, passed by
                                                   # persist to TaskStore.save's
                                                   # cache_entries argument (ADR-009 §9;
                                                   # correction below) for an atomic commit

class GraphState(BaseModel):
    state: RetrievalState
    scratch: IterationScratch = IterationScratch()
```

### `RunContext` (not state — passed to `Engine` and the trace writer)
```python
class RunContext(BaseModel):
    run_id: str
    system: str                # "controller_on" | baseline name
```
Stamped onto `TraceStep`/`EvidenceBundle` at write time (ADR-009 §6). `eval/run.py`
constructs one per experiment sub-run; the API server uses a fixed context.

### Field-ownership matrix (binding)
Only the listed writer may change a field. Everyone may read everything.
`current_subgoal`, `pending_query`, `current_need_type` are not listed below — they are
computed properties with no writer of their own; they change only as a side effect of
whoever writes `current_gap_id` or the current gap's `query`/`need_type` (ADR-009 §4
correction).
| Field(s) | Writer |
|---|---|
| task_id, goal, created_at, budget.max_* | TaskStore.create (immutable afterwards) |
| version, updated_at | `persist` node / TaskStore.save |
| gaps (create, supersede, and a REWRITE's `gap.query`/`gap.need_type` update), query_history, plan_events, needs_initial_plan | Planner via `maybe_plan` / TRANSFORM execution |
| gaps (resolve), evidence | EvidenceManager |
| gaps (exhaust), current_gap_id, stagnation_count, transforms_in_streak, action_gain_ema, status, stop_reason | TrajectoryMonitor (`monitor` node's `MonitorVerdict`, and the shared `advance()` function it and `maybe_plan`/`execute`'s SPLIT branch call — ADR-009 §4; `advance()` writes only `current_gap_id`, per the corrected `AdvanceResult`) |
| status = FAILED, stop_reason = ERROR | `Engine` facade, on the non-retryable error path outside any graph node (ARCHITECTURE §8; ADR-009 §4) |
| last_decision, ActionDecision.plan_mode | `select_action` node |
| retrieval_history, seen_candidate_ids | `execute` node |
| last_result | `score_and_update` node (needs post-rerank relevances — ADR-009 §4) |
| last_update | `score_and_update` node |
| budget.*_used | the node that consumed the resource, via `Budget.consume(...)` |

## 5. Invariants (each has a unit test in `tests/unit/test_state_invariants.py`)
None of these are Pydantic validators on `RetrievalState` — several are legitimately
false mid-iteration. `persist` calls `check_invariants(prev, new, cfg) -> None`
(`engine/state/invariants.py`) once per completed iteration, before saving (ADR-009 §5).
1. `budget.steps_used ≤ max_steps`, `retrieval_calls_used ≤ max_retrieval_calls`,
   `llm_calls_used ≤ max_llm_calls`; tokens may exceed `max_tokens` by at most one call's
   usage, after which the monitor stops with MAX_TOKENS.
2. `evidence` has unique `evidence_id`s; `len(evidence) ≤ evidence.max_items`.
3. `retrieval_history`, `query_history`, `plan_events` are append-only.
4. A gap with status RESOLVED has `resolved_by` equal to the id of a `pinned` item in
   `evidence`.
5. `status == ACTIVE ⇔ stop_reason is None`. `stop_reason` is set exactly once.
6. If `status == ACTIVE` and `needs_initial_plan is False`, then `current_gap_id` refers
   to a gap with status OPEN. (Checked only at `persist`, i.e. end-of-iteration; may be
   false between `execute`'s SPLIT and `monitor`'s `advance()` call within one iteration.)
7. At most `planner.max_gaps` gaps with status OPEN or RESOLVED exist at once.
8. `version` increases by exactly 1 per successful save.
9. State is mutated only inside graph nodes via returned partial updates; no component
   stores a reference to the state between calls. (Process rule; enforced by review and
   the import-boundary test, not by `check_invariants`.)
10. `budget.steps_used == max_steps` only on a completed iteration whose `monitor` step
    set `stop_reason = MAX_STEPS` (ACTION_POLICY §6.4's `steps_used + 1 >= max_steps`
    check, evaluated before `persist` increments the counter — ADR-009 §12); with
    `max_steps = N`, exactly `N` real iterations run, never `N-1` real iterations plus a
    wasted STOP-only `N`-th.

## 6. Terminal and trace models
```python
class EvidenceBundle(BaseModel):
    task_id: str
    system: str                       # "controller_on" | baseline name — from RunContext
    goal: str
    evidence: list[EvidenceItem]      # sorted by final_score desc
    gaps: list[EvidenceGap]
    stop_reason: StopReason
    trace_summary: "TraceSummary"
    # No partial-bundle variant for now (ADR-009 §7) — a bundle always represents a
    # stopped task; backlogged behind a future cancel endpoint.

class TraceSummary(BaseModel):
    steps: int
    retrieval_calls: int
    cache_hits: int
    llm_calls: int
    tokens: int
    reranker_calls: int
    total_latency_ms: float
    actions: list[Action]             # action sequence

class TraceStep(BaseModel):          # semantics: OBSERVABILITY.md §2
    schema_version: int = 1
    run_id: str                       # from RunContext, not state
    task_id: str
    system: str                       # from RunContext, not state
    step: int                         # = budget.steps_used + 1 at the time this step ran
    action: Action
    rule: str
    eligible_actions: list[Action]
    utility_scores: dict[Action, ActionScores]
    query: str | None = None
    mode: RetrievalMode | None = None
    k: int | None = None
    cache_hit: bool
    candidate_ids: list[str]
    new_candidate_ids: list[str]
    accepted_ids: list[str]
    rejected: dict[str, str]
    evicted_ids: list[str]
    resolved_gap_ids: list[str]
    evidence_gain: float
    current_gap_id: str | None = None
    plan_events: list[PlanEvent]      # events produced in this step
    stagnation_count: int
    latency_ms: dict[str, float]      # plan, select, execute, rerank, evidence, monitor, total
    llm_calls: int
    tokens: int
    budget_after: Budget
    stop_reason: StopReason | None = None
    timestamp: datetime                # UTC-aware
```

### `DirectResult` (ADR-012, Proposed)
Return type of `Engine.retrieve_direct` (controller-OFF path, ARCHITECTURE §6); never part
of task state.
```python
class DirectResult(BaseModel):
    candidates: list[Candidate]                  # backend order, rank 0..k-1, with vectors
    scored: list[ScoredCandidate] | None = None  # rerank=True: relevance-only reranker output
    latency_ms: float                            # wall clock: search (+ rerank)
```

`vector` fields (`Candidate.vector`, `EvidenceItem.vector`, `CacheEntry.query_vector`)
are omitted from API responses, traces and bundles by default (a `field_serializer`
gated on a serialization context, not `Field(exclude=True)` — see §2's note) but are
persisted in TaskStore/cache via `engine/state/serialization.py`'s `dump_with_vectors` /
`load_with_vectors` helpers, which pass `model_dump(context={"with_vectors": True})`.
For `RetrievalState`, `dump_with_vectors` additionally excludes the three
`@computed_field` properties (`current_subgoal`, `pending_query`, `current_need_type`)
before writing to TaskStore, and `load_with_vectors` validates the resulting JSON (which
never contained those keys) back into a `RetrievalState` — a plain `model_dump()` (API,
traces, `EvidenceBundle`) still includes them (§2, §4).

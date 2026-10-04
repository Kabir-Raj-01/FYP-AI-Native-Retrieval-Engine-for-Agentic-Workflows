"""All shared data models (docs/STATE_SPEC.md, as amended by ADR-009).

This file and STATE_SPEC.md must match exactly; change either only through an ADR.

Vector fields (``Candidate.vector``, ``EvidenceItem.vector``, ``CacheEntry.query_vector``)
are serialized as ``None`` unless the serialization context requests them
(``{"with_vectors": True}``). Use ``engine.state.serialization.dump_with_vectors`` /
``load_with_vectors`` for persistence instead of building the context by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    computed_field,
    field_serializer,
)

__all__ = [
    "WITH_VECTORS",
    "Action",
    "ActionDecision",
    "ActionScores",
    "AdvanceResult",
    "Budget",
    "CacheEntry",
    "CacheHit",
    "Candidate",
    "EvidenceBundle",
    "EvidenceGap",
    "EvidenceItem",
    "EvidenceUpdate",
    "GapStatus",
    "GraphState",
    "ItemChange",
    "IterationScratch",
    "LastResult",
    "MonitorVerdict",
    "NeedType",
    "PlanEvent",
    "PlanLLMOutput",
    "PlanMode",
    "PlanOutcome",
    "PlanResult",
    "PlanStatus",
    "QueryRecord",
    "RetrievalMode",
    "RetrievalRecord",
    "RetrievalState",
    "RunContext",
    "ScoredCandidate",
    "StopReason",
    "SubgoalPlan",
    "TaskStatus",
    "TraceStep",
    "TraceSummary",
]

# Serialization-context key that opts vector fields into a dump (ADR-009 "Also").
WITH_VECTORS = "with_vectors"


def _vectors_requested(info: SerializationInfo) -> bool:
    context = info.context
    return isinstance(context, Mapping) and bool(context.get(WITH_VECTORS, False))


# =============================================================================== §1 Enums
class NeedType(StrEnum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    MIXED = "mixed"


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class Action(StrEnum):
    REUSE = "REUSE"
    DENSE = "DENSE"
    SPARSE_HYBRID = "SPARSE_HYBRID"
    TRANSFORM = "TRANSFORM"
    RERANK_EXPAND = "RERANK_EXPAND"
    STOP = "STOP"


class GapStatus(StrEnum):
    OPEN = "open"  # still needed
    RESOLVED = "resolved"  # covered by accepted evidence
    SUPERSEDED = "superseded"  # replaced by child gaps after a SPLIT
    EXHAUSTED = "exhausted"  # given up after stagnation; task continues with other gaps


class TaskStatus(StrEnum):
    ACTIVE = "active"
    STOPPED = "stopped"
    FAILED = "failed"


class StopReason(StrEnum):
    ALL_GAPS_RESOLVED = "ALL_GAPS_RESOLVED"
    STAGNATION = "STAGNATION"  # remaining gaps exhausted
    LOW_MARGINAL_UTILITY = "LOW_MARGINAL_UTILITY"
    MAX_STEPS = "MAX_STEPS"
    MAX_RETRIEVAL_CALLS = "MAX_RETRIEVAL_CALLS"
    MAX_LLM_CALLS = "MAX_LLM_CALLS"
    MAX_TOKENS = "MAX_TOKENS"
    ONE_SHOT_COMPLETE = "ONE_SHOT_COMPLETE"  # baselines only
    FIXED_ITERATIONS_COMPLETE = "FIXED_ITERATIONS_COMPLETE"  # baseline only
    ERROR = "ERROR"


class PlanMode(StrEnum):
    INIT = "init"
    REWRITE = "rewrite"
    SPLIT = "split"


class PlanStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FALLBACK = "fallback"


# ============================================================ §2 Retrieval and evidence
class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str  # stable ID from DATA.md
    doc_id: str
    text: str
    retrieval_score: float  # raw backend score (not comparable across modes)
    rank: int  # 0-based rank in the backend result
    mode: RetrievalMode
    metadata: dict[str, str | int | None] = {}  # title, source, published_at, ...
    vector: list[float] | None = None  # dense vector, for similarity signals

    @field_serializer("vector")
    def _serialize_vector(
        self, vector: list[float] | None, info: SerializationInfo
    ) -> list[float] | None:
        return vector if _vectors_requested(info) else None


class ScoredCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: Candidate
    relevance: float  # [0,1] cross-encoder
    gap_coverage: float  # [0,1] max over open gaps
    novelty: float  # [0,1] vs E_t
    redundancy: float  # [0,1] vs higher-ranked selected candidates (MMR)
    final_score: float  # RERANK_SPEC formula
    covered_gap_id: str | None = None  # gap with the max coverage


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str  # == chunk_id (one evidence item per chunk)
    chunk_id: str
    doc_id: str
    text: str
    summary: str  # deterministic truncation, see PLANNER_SPEC §3
    relevance: float
    novelty: float
    final_score: float
    source_action: Action  # ADR-009 §9: REUSE for items accepted from a REUSE step
    query: str
    metadata: dict[str, str | int | None] = {}  # copied from Candidate.metadata (ADR-009 §7)
    added_at_step: int
    supports_gap_ids: list[str] = []
    pinned: bool = False  # True if it resolved a gap -> never evicted
    vector: list[float] | None = None

    @field_serializer("vector")
    def _serialize_vector(
        self, vector: list[float] | None, info: SerializationInfo
    ) -> list[float] | None:
        return vector if _vectors_requested(info) else None


class EvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gap_id: str  # "g0", "g1", ... unique within task
    description: str  # natural-language information need
    query: str  # planner-proposed retrieval query for this gap
    need_type: NeedType
    status: GapStatus = GapStatus.OPEN
    parent_gap_id: str | None = None  # set for children of a SPLIT
    resolved_by: str | None = None  # evidence_id
    created_at_step: int
    closed_at_step: int | None = None


# ============================================== §3 History, budget, planning, decisions
# `step` fields hold n = budget.steps_used + 1 at the time the record is created.
class QueryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    gap_id: str
    query: str
    need_type: NeedType
    source: str  # "planner" | "fallback" (ADR-009: "gap_default" removed)


class RetrievalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    action: Action  # DENSE | SPARSE_HYBRID | RERANK_EXPAND | REUSE
    mode: RetrievalMode | None = None
    query: str
    k: int
    candidate_ids: list[str]
    new_candidate_ids: list[str]  # not seen earlier in this task
    cache_hit: bool  # True iff this record's candidates came from the cache (REUSE only)
    latency_ms: float


class LastResult(BaseModel):
    """Compact view of the most recent candidate-producing action (for RERANK_EXPAND).

    Written by score_and_update (needs top_relevances, which only exist post-rerank), not
    execute (ADR-009 §4). Never reset except by monitor.advance() on gap advancement
    (ADR-009 §9) — a TRANSFORM/STOP step leaves the prior value intact.
    """

    model_config = ConfigDict(extra="forbid")

    step: int
    query: str  # ADR-009 §9: for REUSE, the current pending query
    mode: RetrievalMode | None = None
    depth: int  # ADR-009 §9: for REUSE, len(entry.candidates)
    candidate_ids: list[str]
    top_relevances: list[float]  # relevance of top rerank.top_n_output, descending


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(ge=1)
    max_retrieval_calls: int = Field(ge=1)
    max_llm_calls: int = Field(ge=0)  # 0 = no-LLM ablation
    max_tokens: int = Field(ge=1)
    steps_used: int = 0
    retrieval_calls_used: int = 0
    llm_calls_used: int = 0
    tokens_used: int = 0
    reranker_calls_used: int = 0  # tracked, not capped

    def consume(
        self,
        *,
        steps: int = 0,
        retrieval_calls: int = 0,
        llm_calls: int = 0,
        tokens: int = 0,
        reranker_calls: int = 0,
    ) -> Budget:
        """Returns a new Budget with the given deltas added; never mutates self."""
        return self.model_copy(
            update={
                "steps_used": self.steps_used + steps,
                "retrieval_calls_used": self.retrieval_calls_used + retrieval_calls,
                "llm_calls_used": self.llm_calls_used + llm_calls,
                "tokens_used": self.tokens_used + tokens,
                "reranker_calls_used": self.reranker_calls_used + reranker_calls,
            }
        )

    def pressure(self) -> float:
        """Max utilisation ratio, in [0,1], over caps with max_* > 0; a zero cap is
        skipped rather than causing a division by zero (ADR-009 §8)."""
        pairs = (
            (self.steps_used, self.max_steps),
            (self.retrieval_calls_used, self.max_retrieval_calls),
            (self.llm_calls_used, self.max_llm_calls),
            (self.tokens_used, self.max_tokens),
        )
        ratios = [used / cap for used, cap in pairs if cap > 0]
        return min(1.0, max(ratios, default=0.0))


class PlanEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    mode: PlanMode
    attempt: int
    status: PlanStatus
    reason: str | None = None  # reason code, PLANNER_SPEC §5


class SubgoalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    query: str
    need_type: NeedType
    # No length constraints here: they are config rules enforced by the planner validator
    # (PLANNER_SPEC §5; CLAUDE.md rule 4).


class PlanLLMOutput(BaseModel):
    """What is actually asked of the LLM (LLMClient.structured(schema=PlanLLMOutput)).

    No `mode` (the planner already knows the mode it requested) and no length/count
    constraints (ADR-009 §10).
    """

    model_config = ConfigDict(extra="forbid")

    subgoals: list[SubgoalPlan] = Field(min_length=1)
    referenced_entities: list[str] = []
    rationale: str = ""


class PlanResult(BaseModel):
    """Built by Planner.plan from PlanLLMOutput + the requested mode, after validation.

    rationale is truncated (never rejected) to planner.rationale_max_chars by the validator
    (ADR-009 §10).
    """

    model_config = ConfigDict(extra="forbid")

    mode: PlanMode
    subgoals: list[SubgoalPlan] = Field(min_length=1)  # bounds checked by the validator
    referenced_entities: list[str] = []
    rationale: str = ""  # logged only, never used for control


class PlanOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: PlanResult  # accepted plan or the deterministic fallback
    events: list[PlanEvent]
    llm_calls: int
    tokens: int


class ActionScores(BaseModel):
    # Field names stay G/C/R/L/U (frozen shape); code computing them uses gain, cost,
    # redundancy_signal, latency_signal (ADR-009 "Also").
    model_config = ConfigDict(extra="forbid")

    G: float
    C: float
    R: float
    L: float
    U: float


class ActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    eligible: list[Action]
    scores: dict[Action, ActionScores]  # eligible non-STOP actions only
    stop_reason: StopReason | None = None
    plan_mode: PlanMode | None = None  # set iff action == TRANSFORM (ADR-009 §3)
    rule: str  # which rule fired, e.g. "utility_argmax", "budget_guard"


class CacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    task_id: str
    entry_id: str
    normalized_query: str
    query_vector: list[float]
    need_type: NeedType
    mode: RetrievalMode
    candidates: list[Candidate]  # stored with vectors
    accepted_ids: list[str]
    created_at_step: int

    @field_serializer("query_vector")
    def _serialize_query_vector(
        self, query_vector: list[float], info: SerializationInfo
    ) -> list[float] | None:
        return query_vector if _vectors_requested(info) else None


class CacheHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry: CacheEntry
    similarity: float  # 1.0 for exact normalized match
    match: str  # "exact" | "semantic"


class ItemChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pinned: bool | None = None
    supports_gap_ids: list[str] | None = None


class EvidenceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: list[EvidenceItem]
    rejected: dict[str, str]  # chunk_id -> reason code
    evicted: list[str]
    resolved_gap_ids: list[str]
    gap_resolutions: dict[str, str] = {}  # gap_id -> evidence_id (ADR-009 §4)
    item_changes: dict[str, ItemChange] = {}  # evidence_id -> changed flags (ADR-009 §4)
    evidence_gain: float  # [0,1]


class MonitorVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_continue: bool  # renamed from continue_ (ADR-009 §11); no alias
    stop_reason: StopReason | None = None
    advanced_to_gap: str | None = None
    exhausted_gap: str | None = None
    stagnation_count: int  # ADR-009 §4: full new values, applied verbatim
    transforms_in_streak: int
    action_gain_ema: dict[Action, float]
    gap_status_changes: dict[str, GapStatus] = {}
    current_gap_id: str | None = None
    note: str = ""


class AdvanceResult(BaseModel):
    """Return type of the monitor-owned advance() function (ADR-009 §4).

    Only current_gap_id is applied to state: current_subgoal, pending_query and
    current_need_type are computed fields derived from the gap it names (ADR-009 §4
    correction).
    """

    model_config = ConfigDict(extra="forbid")

    current_gap_id: str | None
    reset_stagnation: bool


# ======================================================================= §4 RetrievalState
class RetrievalState(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    schema_version: int = 1
    task_id: str
    version: int = 0
    status: TaskStatus = TaskStatus.ACTIVE
    stop_reason: StopReason | None = None
    # No `step` field — budget.steps_used is the sole step counter (ADR-009 "Also").

    goal: str  # G (immutable)
    gaps: list[EvidenceGap] = []
    current_gap_id: str | None = None
    # current_subgoal, pending_query, current_need_type are computed properties derived
    # from gaps[current_gap_id] (ADR-009 §4 correction); persistence excludes them (§4a).
    needs_initial_plan: bool = True

    evidence: list[EvidenceItem] = []  # E_t
    query_history: list[QueryRecord] = []  # Q_t: queries the planner proposed
    retrieval_history: list[RetrievalRecord] = []  # R_t: queries actually issued
    budget: Budget  # B_t

    seen_candidate_ids: list[str] = []  # every chunk returned so far
    last_result: LastResult | None = None
    # No `current_depth` field — use last_result.depth (0/absent when no last_result).
    stagnation_count: int = 0
    transforms_in_streak: int = 0
    action_gain_ema: dict[Action, float] = {}  # missing key -> controller.prior_action_gain
    last_decision: ActionDecision | None = None
    last_update: EvidenceUpdate | None = None  # None on steps without candidates
    plan_events: list[PlanEvent] = []

    created_at: AwareDatetime  # UTC-aware
    updated_at: AwareDatetime  # UTC-aware

    @property
    def current_gap(self) -> EvidenceGap | None:
        return next((g for g in self.gaps if g.gap_id == self.current_gap_id), None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_subgoal(self) -> str:  # g_t = current gap's description
        g = self.current_gap
        return g.description if g else ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pending_query(self) -> str:  # current gap's query
        g = self.current_gap
        return g.query if g else ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_need_type(self) -> NeedType:  # current gap's need_type
        g = self.current_gap
        return g.need_type if g else NeedType.MIXED


class IterationScratch(BaseModel):
    """Transient, per-iteration data passed between graph nodes. Never persisted;
    discarded after persist. ADR-009 §2."""

    model_config = ConfigDict(extra="forbid")

    cache_hit: CacheHit | None = None
    candidates: list[Candidate] = []  # from execute
    scored: list[ScoredCandidate] = []  # from score_and_update (rerank output)
    evidence_update: EvidenceUpdate | None = None
    plan_outcome: PlanOutcome | None = None
    queued_cache_entries: list[CacheEntry] = []  # committed by persist via TaskStore.save


class GraphState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: RetrievalState
    scratch: IterationScratch = IterationScratch()


class RunContext(BaseModel):
    """Not state — passed to Engine and the trace writer (ADR-009 §6)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    system: str  # "controller_on" | baseline name


# ========================================================== §6 Terminal and trace models
class TraceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int
    retrieval_calls: int
    cache_hits: int
    llm_calls: int
    tokens: int
    reranker_calls: int
    total_latency_ms: float
    actions: list[Action]  # action sequence


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    system: str  # "controller_on" | baseline name — from RunContext
    goal: str
    evidence: list[EvidenceItem]  # sorted by final_score desc
    gaps: list[EvidenceGap]
    stop_reason: StopReason
    trace_summary: TraceSummary
    # No partial-bundle variant for now (ADR-009 §7).


class TraceStep(BaseModel):  # semantics: OBSERVABILITY.md §2
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str  # from RunContext, not state
    task_id: str
    system: str  # from RunContext, not state
    step: int  # = budget.steps_used + 1 at the time this step ran
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
    plan_events: list[PlanEvent]  # events produced in this step
    stagnation_count: int
    latency_ms: dict[str, float]  # plan, select, execute, rerank, evidence, monitor, total
    llm_calls: int
    tokens: int
    budget_after: Budget
    stop_reason: StopReason | None = None
    timestamp: AwareDatetime  # UTC-aware

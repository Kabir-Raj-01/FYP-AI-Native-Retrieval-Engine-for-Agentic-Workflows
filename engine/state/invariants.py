"""End-of-iteration state invariants (STATE_SPEC §5; ADR-009 §5).

``persist`` calls :func:`check_invariants` once per completed iteration, after it has
applied its own in-memory updates (``budget.steps_used + 1``, ``version + 1``) and before
``TaskStore.save``. These are deliberately not Pydantic validators: several are
legitimately false between nodes within one iteration.

Invariant 9 (state mutated only via node partial updates) is a process rule enforced by
review and the import-boundary test, not here.

Interpretations agreed at the Week 7 plan review:
- inv 1, tokens: the cap may be exceeded only by the iteration that crossed it (the previous
  state was under the cap), and that iteration must have stopped the task.
- inv 8: ``new.version == prev.version + 1`` (persist bumps the version before saving).
- inv 10: when ``steps_used`` reaches ``max_steps`` the task is stopped with MAX_STEPS or
  with a stop reason the monitor checks before MAX_STEPS (ACTION_POLICY §6.4 order).
- additionally, the fields the ownership matrix makes immutable after TaskStore.create
  (task_id, goal, created_at, budget.max_*) must not change.
"""

from __future__ import annotations

from engine.config import EngineConfig
from engine.errors import StateInvariantError
from engine.state.models import GapStatus, RetrievalState, StopReason, TaskStatus

__all__ = ["check_invariants"]

# Stop reasons the monitor evaluates before (or as) MAX_STEPS, ACTION_POLICY §6.4.
_STOPS_AT_STEP_CAP = frozenset(
    {StopReason.ALL_GAPS_RESOLVED, StopReason.STAGNATION, StopReason.MAX_STEPS}
)


def check_invariants(prev: RetrievalState, new: RetrievalState, cfg: EngineConfig) -> None:
    """Raise StateInvariantError listing every violated invariant; return None if all hold.

    Args:
        prev: the state as loaded at the start of the iteration.
        new: the state ``persist`` is about to save.
        cfg: the engine configuration (``evidence.max_items``, ``planner.max_gaps``).
    """
    violations: list[str] = []
    violations += _budget_caps(prev, new)
    violations += _evidence(new, cfg)
    violations += _append_only(prev, new)
    violations += _resolved_gaps(new)
    violations += _status(prev, new)
    violations += _current_gap(new)
    violations += _live_gap_count(new, cfg)
    violations += _version(prev, new)
    violations += _step_cap(new)
    violations += _immutable(prev, new)
    if violations:
        raise StateInvariantError(
            f"task {new.task_id}: " + "; ".join(violations), task_id=new.task_id
        )


def _budget_caps(prev: RetrievalState, new: RetrievalState) -> list[str]:
    b = new.budget
    out: list[str] = []
    if b.steps_used > b.max_steps:
        out.append(f"[inv 1] steps_used {b.steps_used} > max_steps {b.max_steps}")
    if b.retrieval_calls_used > b.max_retrieval_calls:
        out.append(
            f"[inv 1] retrieval_calls_used {b.retrieval_calls_used} > "
            f"max_retrieval_calls {b.max_retrieval_calls}"
        )
    if b.llm_calls_used > b.max_llm_calls:
        out.append(f"[inv 1] llm_calls_used {b.llm_calls_used} > max_llm_calls {b.max_llm_calls}")
    if b.tokens_used > b.max_tokens:
        if prev.budget.tokens_used >= b.max_tokens:
            out.append("[inv 1] tokens_used exceeded max_tokens in more than one iteration")
        if new.status == TaskStatus.ACTIVE:
            out.append("[inv 1] tokens_used > max_tokens but the task is still active")
    return out


def _evidence(new: RetrievalState, cfg: EngineConfig) -> list[str]:
    out: list[str] = []
    ids = [e.evidence_id for e in new.evidence]
    if len(ids) != len(set(ids)):
        out.append("[inv 2] duplicate evidence_id in evidence")
    if len(ids) > cfg.evidence.max_items:
        out.append(f"[inv 2] {len(ids)} evidence items > evidence.max_items")
    return out


def _append_only(prev: RetrievalState, new: RetrievalState) -> list[str]:
    out: list[str] = []
    histories = (
        ("retrieval_history", prev.retrieval_history, new.retrieval_history),
        ("query_history", prev.query_history, new.query_history),
        ("plan_events", prev.plan_events, new.plan_events),
    )
    for name, before, after in histories:
        if len(after) < len(before) or list(after[: len(before)]) != list(before):
            out.append(f"[inv 3] {name} is not append-only")
    return out


def _resolved_gaps(new: RetrievalState) -> list[str]:
    pinned = {e.evidence_id for e in new.evidence if e.pinned}
    return [
        f"[inv 4] gap {g.gap_id} is RESOLVED but resolved_by={g.resolved_by!r} is not a "
        "pinned evidence item"
        for g in new.gaps
        if g.status == GapStatus.RESOLVED and g.resolved_by not in pinned
    ]


def _status(prev: RetrievalState, new: RetrievalState) -> list[str]:
    out: list[str] = []
    if (new.status == TaskStatus.ACTIVE) != (new.stop_reason is None):
        out.append(
            f"[inv 5] status {new.status.value} inconsistent with stop_reason {new.stop_reason}"
        )
    if prev.stop_reason is not None and new.stop_reason != prev.stop_reason:
        out.append("[inv 5] stop_reason changed after it was set")
    return out


def _current_gap(new: RetrievalState) -> list[str]:
    if new.status != TaskStatus.ACTIVE or new.needs_initial_plan:
        return []
    gap = new.current_gap
    if gap is None or gap.status != GapStatus.OPEN:
        return [f"[inv 6] active task's current_gap_id {new.current_gap_id!r} is not an OPEN gap"]
    return []


def _live_gap_count(new: RetrievalState, cfg: EngineConfig) -> list[str]:
    live = sum(1 for g in new.gaps if g.status in (GapStatus.OPEN, GapStatus.RESOLVED))
    if live > cfg.planner.max_gaps:
        return [f"[inv 7] {live} OPEN/RESOLVED gaps > planner.max_gaps {cfg.planner.max_gaps}"]
    return []


def _version(prev: RetrievalState, new: RetrievalState) -> list[str]:
    if new.version != prev.version + 1:
        return [f"[inv 8] version {prev.version} -> {new.version}, expected +1"]
    return []


def _step_cap(new: RetrievalState) -> list[str]:
    b = new.budget
    if b.steps_used != b.max_steps:
        return []
    if new.status != TaskStatus.STOPPED or new.stop_reason not in _STOPS_AT_STEP_CAP:
        return [
            f"[inv 10] steps_used reached max_steps but status={new.status.value}, "
            f"stop_reason={new.stop_reason}"
        ]
    return []


def _immutable(prev: RetrievalState, new: RetrievalState) -> list[str]:
    out: list[str] = []
    for name in ("task_id", "goal", "created_at"):
        if getattr(prev, name) != getattr(new, name):
            out.append(f"[immutable] {name} changed")
    for name in ("max_steps", "max_retrieval_calls", "max_llm_calls", "max_tokens"):
        if getattr(prev.budget, name) != getattr(new.budget, name):
            out.append(f"[immutable] budget.{name} changed")
    return out

"""Bounded LLM query planner (PLANNER_SPEC §1-§6; ADR-009 §3, §10).

``plan`` makes at most ``planner.max_attempts`` LLM calls, each only while the task's LLM
budget allows, validates each reply (``validator.py``) and otherwise returns the
deterministic fallback. Transport and output failures become REJECTED attempts
(``LLM_ERROR`` / ``SCHEMA``); only ``LLMBudgetExceeded`` (the per-run cap) propagates.
Pure with respect to the state: the calling node applies the PlanOutcome.

Accounting: every attempt that was sent counts one LLM call (a failed one too — it may
have reached the provider); tokens are those the client reports for successful replies.
"""

from __future__ import annotations

from engine.config import EngineConfig
from engine.errors import LLMError, LLMOutputError
from engine.llm.interfaces import LLMClient, Message
from engine.monitor.logging import get_logger
from engine.planner.entities import heuristic
from engine.planner.interfaces import Planner
from engine.planner.prompting import render
from engine.planner.validator import PlanRejectedError, previous_queries, validate
from engine.state.models import (
    GapStatus,
    PlanEvent,
    PlanLLMOutput,
    PlanMode,
    PlanOutcome,
    PlanResult,
    PlanStatus,
    RetrievalState,
    SubgoalPlan,
)
from engine.text import normalize

__all__ = ["LLMPlanner"]

_RETRY_NOTE = (
    "Your previous answer was rejected ({reason}). Answer again; use only entities that "
    "appear in the goal, the gaps or the evidence above."
)


class LLMPlanner(Planner):
    def __init__(self, cfg: EngineConfig, llm: LLMClient) -> None:
        self._cfg = cfg
        self._llm = llm
        self._log = get_logger("engine.planner")

    def plan(self, state: RetrievalState, mode: PlanMode) -> PlanOutcome:
        n = state.budget.steps_used + 1
        events: list[PlanEvent] = []
        calls = tokens = 0
        ablation = self._cfg.ablation
        # without gap tracking INIT is exactly one gap, the goal (ACTION_POLICY §6)
        if ablation.planner and (mode != PlanMode.INIT or ablation.gap_tracking):
            values, sources = self._inputs(state, mode)
            reason: str | None = None
            for attempt in range(1, self._cfg.planner.max_attempts + 1):
                if state.budget.llm_calls_used + calls >= state.budget.max_llm_calls:
                    break
                note = _RETRY_NOTE.format(reason=reason) if reason else ""
                messages: list[Message] = render(mode, retry_note=note, **values)
                calls += 1
                try:
                    reply = self._llm.structured("planner", messages, PlanLLMOutput)
                    tokens += reply.input_tokens + reply.output_tokens
                    result = validate(reply.value, mode, state, sources, self._cfg)
                except LLMOutputError:
                    reason = "SCHEMA"
                except LLMError:
                    reason = "LLM_ERROR"
                except PlanRejectedError as exc:
                    reason = exc.reason
                else:
                    events.append(self._event(n, mode, attempt, PlanStatus.ACCEPTED, None))
                    self._log.info(
                        "planner.accepted",
                        mode=mode.value,
                        attempt=attempt,
                        result=result.model_dump(),
                    )
                    return PlanOutcome(result=result, events=events, llm_calls=calls, tokens=tokens)
                events.append(self._event(n, mode, attempt, PlanStatus.REJECTED, reason))
                self._log.info("planner.rejected", mode=mode.value, attempt=attempt, reason=reason)
        fallback = self._fallback(state, mode)
        attempt = len(events) + 1
        events.append(self._event(n, mode, attempt, PlanStatus.FALLBACK, None))
        self._log.info("planner.fallback", mode=mode.value, result=fallback.model_dump())
        return PlanOutcome(result=fallback, events=events, llm_calls=calls, tokens=tokens)

    @staticmethod
    def _event(
        n: int, mode: PlanMode, attempt: int, status: PlanStatus, reason: str | None
    ) -> PlanEvent:
        return PlanEvent(step=n, mode=mode, attempt=attempt, status=status, reason=reason)

    def _inputs(self, state: RetrievalState, mode: PlanMode) -> tuple[dict[str, object], list[str]]:
        """Template values, and the texts entity grounding may match against (§3, §5)."""
        p = self._cfg.planner
        top = sorted(state.evidence, key=lambda e: -e.final_score)[: p.max_evidence_summaries]
        evidence = [{"id": e.evidence_id, "summary": e.summary} for e in top]
        gap = state.current_gap
        others = [
            g.description
            for g in state.gaps
            if g.status == GapStatus.OPEN and (gap is None or g.gap_id != gap.gap_id)
        ]
        values: dict[str, object] = {
            "goal": state.goal,
            "evidence": evidence,
            "max_subgoals": p.max_subgoals,
        }
        sources = [state.goal, *(e.summary for e in top)]
        if mode != PlanMode.INIT and gap is not None:
            previous = [q.query for q in state.query_history if q.gap_id == gap.gap_id]
            values |= {
                "gap": gap.description,
                "previous": previous or [gap.query],
                "other_gaps": others,
            }
            sources += [gap.description, *others]
        return values, sources

    def _fallback(self, state: RetrievalState, mode: PlanMode) -> PlanResult:
        """PLANNER_SPEC §6: INIT -> the goal as one subgoal; REWRITE/SPLIT -> one subgoal
        on the current gap (its description as the query if untried, else the goal)."""
        gap = state.current_gap
        if mode == PlanMode.INIT or gap is None:
            sg = SubgoalPlan(
                description=state.goal, query=state.goal, need_type=heuristic(state.goal, self._cfg)
            )
        else:
            tried = previous_queries(state)
            query = gap.description if normalize(gap.description) not in tried else state.goal
            sg = SubgoalPlan(
                description=gap.description, query=query, need_type=heuristic(query, self._cfg)
            )
        return PlanResult(mode=mode, subgoals=[sg], rationale="deterministic fallback")

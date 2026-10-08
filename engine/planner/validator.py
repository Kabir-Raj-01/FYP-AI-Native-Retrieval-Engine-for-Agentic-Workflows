"""Planner output validation (PLANNER_SPEC §5; ADR-009 §10).

Checks in order, the first failure giving the reason code: SCHEMA, COUNT,
EMPTY_OR_DUPLICATE, UNGROUNDED_ENTITY, UNDECLARED_ENTITY. Count bounds come from
``planner.*`` config, never from the model schema. On success the PlanResult carries the
requested mode and a rationale truncated to ``planner.rationale_max_chars``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import ValidationError
from rapidfuzz import fuzz

from engine.config import EngineConfig
from engine.errors import PlannerValidationError
from engine.planner.entities import extract
from engine.state.models import GapStatus, PlanLLMOutput, PlanMode, PlanResult, RetrievalState
from engine.text import normalize

__all__ = ["PlanRejectedError", "validate"]


class PlanRejectedError(PlannerValidationError):
    """Carries the PLANNER_SPEC §5 reason code in ``reason``."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.reason = code


def _grounded(text: str, sources: Sequence[str], threshold: int) -> bool:
    t = text.lower()
    return any(fuzz.partial_ratio(t, s.lower()) >= threshold for s in sources if s)


def _words(text: str) -> list[str]:
    return re.findall(
        r"[a-z0-9][\w-]*", text.lower().replace("'s", "").replace(chr(0x2019) + "s", "")
    )


def _words_grounded(text: str, sources: Sequence[str]) -> bool:
    """Every word of a name span occurs in the sources: keyword-style queries juxtapose
    separately grounded names ("The Verge Apple Google Search"), which is not a new entity."""
    vocab = {w for s in sources for w in _words(s)}
    return all(w in vocab for w in _words(text))


def previous_queries(state: RetrievalState) -> set[str]:
    """Normalised queries already proposed or used for the current gap."""
    gap = state.current_gap
    if gap is None:
        return set()
    queries = {q.query for q in state.query_history if q.gap_id == gap.gap_id} | {gap.query}
    return {normalize(q) for q in queries}


def validate(
    output: PlanLLMOutput,
    mode: PlanMode,
    state: RetrievalState,
    sources: Sequence[str],
    cfg: EngineConfig,
) -> PlanResult:
    """``sources``: the goal, gap descriptions and evidence summaries given to the LLM.

    Raises:
        PlanRejectedError: with ``reason`` set to the first failing check.
    """
    p = cfg.planner
    try:  # SCHEMA: LLMClient already validated; re-validate a copy here
        out = PlanLLMOutput.model_validate(output.model_dump())
    except ValidationError as exc:
        raise PlanRejectedError("SCHEMA", str(exc)) from exc

    n = len(out.subgoals)  # COUNT: the mode's bounds and the live-gap cap
    live = sum(1 for g in state.gaps if g.status in (GapStatus.OPEN, GapStatus.RESOLVED))
    lo, hi, after = {
        PlanMode.INIT: (1, p.max_subgoals, live + n),
        PlanMode.REWRITE: (1, 1, live),
        PlanMode.SPLIT: (2, p.max_subgoals, live - 1 + n),
    }[mode]
    if not lo <= n <= hi or after > p.max_gaps:
        raise PlanRejectedError("COUNT", f"{n} subgoals for {mode.value} ({live} live gaps)")

    used = previous_queries(state) if mode != PlanMode.INIT else set()
    for sg in out.subgoals:  # EMPTY_OR_DUPLICATE
        q = normalize(sg.query)
        if not q or q in used:
            raise PlanRejectedError("EMPTY_OR_DUPLICATE", repr(sg.query))

    for entity in out.referenced_entities:  # UNGROUNDED_ENTITY
        if not _grounded(entity, sources, p.entity_match_threshold):
            raise PlanRejectedError("UNGROUNDED_ENTITY", repr(entity))

    declared = [e.lower() for e in out.referenced_entities]
    spans = [
        s for sg in out.subgoals for field in (sg.description, sg.query) for s in extract(field)
    ]
    for span in spans:  # UNDECLARED_ENTITY
        text = span.text.lower()
        listed = any(text in d or d in text for d in declared)
        grounded = _grounded(span.text, sources, p.entity_match_threshold) or (
            span.kind == "name" and _words_grounded(span.text, sources)
        )
        if not listed and not grounded:
            raise PlanRejectedError("UNDECLARED_ENTITY", repr(span.text))

    return PlanResult(
        mode=mode,
        subgoals=out.subgoals,
        referenced_entities=out.referenced_entities,
        rationale=out.rationale[: p.rationale_max_chars],
    )

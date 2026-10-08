"""Deterministic entity extractor and need heuristic (PLANNER_SPEC §5 UNDECLARED_ENTITY, §6).

``extract`` finds: capitalised multi-word spans not at a sentence start, 4-digit years,
numbers with units, quoted strings and alphanumeric IDs (letters and digits mixed, e.g.
"KX-41", "COVID-19"). ``heuristic`` maps a text to a NeedType.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from engine.config import EngineConfig
from engine.state.models import NeedType

__all__ = ["Span", "extract", "heuristic"]


@dataclass(frozen=True)
class Span:
    kind: str  # "name" | "year" | "number" | "quoted" | "id"
    text: str


# words may hold inner dots ("U.S"), but a name never runs over a line break or a full stop
_WORD = r"[A-Z][\w'&-]*(?:\.[\w'&-]+)*"
_NAME = re.compile(rf"\b{_WORD}(?:[ \t]+{_WORD})+")
_YEAR = re.compile(r"\b(?:1[89]\d\d|20\d\d)\b")
_NUMBER = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|bn|m|k))?)"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent|km/h|km|kg|mph|million|billion|bn|years?|days?|"
    r"hours?|minutes?|weeks?|months?|metres?|meters?|miles?|tonnes?|tons?|people|units?)\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"\"([^\"]{2,})\"|“([^”]{2,})”")
_ID = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)$")
_SEMANTIC_OPENERS = ("why", "how", "what caused", "explain")


def extract(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _NAME.finditer(text):
        words = m.group(0).split()
        if _SENTENCE_START.search(text[: m.start()]):
            words = words[1:]  # the first word is capitalised because it starts a sentence
        if len(words) >= 2:
            spans.append(Span("name", " ".join(words)))
    spans += [Span("year", m.group(0)) for m in _YEAR.finditer(text)]
    spans += [Span("number", m.group(0)) for m in _NUMBER.finditer(text)]
    spans += [Span("quoted", m.group(1) or m.group(2)) for m in _QUOTED.finditer(text)]
    spans += [Span("id", m.group(0)) for m in _ID.finditer(text) if not _YEAR.fullmatch(m.group(0))]
    return spans


def heuristic(text: str, cfg: EngineConfig) -> NeedType:
    """``lexical`` if there is an entity/year/ID/quoted span and the text is short (at
    most ``planner.heuristic_max_tokens`` tokens) or the span is a quoted string / ID;
    ``semantic`` if it opens with why / how / what caused / explain and has no span; else
    ``mixed`` (PLANNER_SPEC §6)."""
    spans = extract(text)
    short = len(text.split()) <= cfg.planner.heuristic_max_tokens
    if spans and (short or any(s.kind in ("quoted", "id") for s in spans)):
        return NeedType.LEXICAL
    if not spans and text.strip().lower().startswith(_SEMANTIC_OPENERS):
        return NeedType.SEMANTIC
    return NeedType.MIXED

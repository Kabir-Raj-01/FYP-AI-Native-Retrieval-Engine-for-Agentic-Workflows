"""engine/planner/entities.py (PLANNER_SPEC §5 extractor, §6 heuristic)."""

from __future__ import annotations

import pytest

from engine.config import EngineConfig
from engine.planner.entities import extract, heuristic
from engine.state.models import NeedType


def kinds(text: str) -> set[tuple[str, str]]:
    return {(s.kind, s.text) for s in extract(text)}


def test_extractor_finds_each_kind() -> None:
    found = kinds(
        'Where did the Talvane Robotics Lab test the KX-41 in 2026 at 60 km/h for "Harbour Watch"?'
    )
    assert ("name", "Talvane Robotics Lab") in found
    assert ("id", "KX-41") in found
    assert ("year", "2026") in found
    assert ("number", "60 km/h") in found
    assert ("quoted", "Harbour Watch") in found
    assert ("id", "2026") not in found  # a year is not an ID


def test_sentence_initial_capitals_are_not_names() -> None:
    assert kinds("Which company built it?") == set()
    assert ("name", "Linden Bay") in kinds(
        "The tram in Linden Bay opened. Linden Bay Council voted."
    )
    assert ("name", "Bay Council") in kinds("Linden Bay Council voted.")  # first word dropped


@pytest.mark.parametrize(
    ("text", "need"),
    [
        ("KX-41 drone maker", NeedType.LEXICAL),  # ID
        (
            'reports mentioning "Harbour Watch" and the long list of other things here',
            NeedType.LEXICAL,
        ),
        ("who leads the Talvane Robotics Lab", NeedType.LEXICAL),  # short with a name
        ("why did the tram expansion stall", NeedType.SEMANTIC),
        ("Explain how the cooperative prices its fruit", NeedType.SEMANTIC),
        ("what caused the delay in harbour dredging", NeedType.SEMANTIC),
        ("which fruit does the farmers cooperative grow", NeedType.MIXED),
        (
            "how did the Linden Bay Council vote on the tram line extension after the long and "
            "contested public consultation",
            NeedType.MIXED,  # has a name but is long
        ),
    ],
)
def test_need_heuristic(cfg: EngineConfig, text: str, need: NeedType) -> None:
    assert heuristic(text, cfg) == need

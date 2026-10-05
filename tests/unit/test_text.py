"""engine/text.py: normalize() (CACHE_SPEC §2, as made precise at the Week 13 plan)."""

from __future__ import annotations

import pytest

from engine.text import normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello, World!", "hello world"),
        ("State-of-the-art   models", "state of the art models"),
        ("GPT-4o vs Llama-3.5", "gpt-4o vs llama-3.5"),  # digit tokens keep inner punctuation
        ("COVID-19 cases rose 3.5% in (2023).", "covid-19 cases rose 3.5 in 2023"),
        ("U.S. tariffs", "u s tariffs"),
        ("  tabs\tand\nnewlines  ", "tabs and newlines"),
        ("Café", "café"),
        ("Café", "café"),  # NFC: e + combining acute == é
        ("company's results", "company s results"),
        ("", ""),
        ("?!...", ""),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_is_idempotent() -> None:
    for raw in ["Q3 2024: revenue (up 12.5%)!", "Who's the CEO of A.B.C.-Corp?"]:
        assert normalize(normalize(raw)) == normalize(raw)


def test_equivalent_queries_normalise_identically() -> None:
    assert normalize("Who acquired Voltrion?") == normalize("  who ACQUIRED voltrion ")

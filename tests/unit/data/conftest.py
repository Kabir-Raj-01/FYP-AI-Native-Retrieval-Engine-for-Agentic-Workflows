"""Shared helpers for data/ tests."""

from __future__ import annotations

import re

import pytest

from data.chunking import Span

_WORD = re.compile(r"\S+")


class WhitespaceTokenizer:
    """One token per whitespace-separated word (a deterministic stand-in for BGE)."""

    def offsets(self, text: str) -> list[Span]:
        return [(m.start(), m.end()) for m in _WORD.finditer(text)]


@pytest.fixture
def ws_tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer()

"""Text normalisation, sentence spans and sentence-aware chunking (DATA.md §3 steps 2-3).

All functions work on character spans of the *normalised* document text, so a chunk is
always the exact slice ``text[char_start:char_end]`` and ``chunk_id`` can be derived from
``char_start`` (DATA.md §3 step 4).

Chunking (ADR-001 values: ``chunking.chunk_size_tokens``, ``chunking.chunk_overlap_tokens``,
``chunking.tokenizer``): sentences are packed greedily while the chunk's token count — the
tokenizer run on the candidate chunk text itself — stays within the limit. The next chunk
starts with the previous chunk's trailing whole sentences, as many as fit in the overlap
budget. A sentence longer than a whole chunk is split into token windows.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

__all__ = ["Span", "Tokenizer", "chunk_spans", "normalize_text", "sentence_spans"]

Span = tuple[int, int]  # [start, end) character offsets

_WHITESPACE = re.compile(r"\s+")
# A sentence ends at . ! or ?, plus any closing quotes/brackets, when whitespace and then a
# capital letter, a digit or an opening quote/bracket follow. Deliberately simple and
# deterministic; imperfect on abbreviations ("U.S. President"), which only moves chunk
# boundaries, never drops text.
# (\u201d \u2019 = closing curly quotes; \u201c \u2018 = opening curly quotes)
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*(?=\s+[\"'\u201c\u2018(\[]?[A-Z0-9])")


class Tokenizer(Protocol):
    """Character offsets of each token of ``text``, without special tokens."""

    def offsets(self, text: str) -> list[Span]: ...


def normalize_text(text: str) -> str:
    """Unicode NFC, every whitespace run collapsed to one space, trimmed (DATA.md §3 step 2).

    No lower-casing and no boilerplate stripping: every MultiHop-RAG gold fact matches its
    article verbatim after exactly this normalisation (verified 2026-10-05).
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def sentence_spans(text: str) -> list[Span]:
    """Sentence spans of normalised ``text``; whitespace between sentences is excluded."""
    spans: list[Span] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.end()))
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.offsets(text))


def _token_windows(
    text: str, span: Span, tokenizer: Tokenizer, max_tokens: int, overlap_tokens: int
) -> list[Span]:
    """Split one over-long sentence into windows of at most ``max_tokens`` tokens."""
    start, end = span
    offsets = tokenizer.offsets(text[start:end])
    step = max(1, max_tokens - overlap_tokens)
    windows: list[Span] = []
    for first in range(0, len(offsets), step):
        last = min(first + max_tokens, len(offsets)) - 1
        windows.append((start + offsets[first][0], start + offsets[last][1]))
        if last == len(offsets) - 1:
            break
    return windows


def chunk_spans(
    text: str, tokenizer: Tokenizer, max_tokens: int, overlap_tokens: int
) -> list[Span]:
    """Chunk spans of normalised ``text`` (see module docstring); ``[]`` for empty text.

    Raises:
        ValueError: ``max_tokens < 1`` or ``overlap_tokens`` not in ``[0, max_tokens)``.
    """
    if max_tokens < 1 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError(
            f"need max_tokens >= 1 and 0 <= overlap < max ({max_tokens=}, {overlap_tokens=})"
        )
    units: list[Span] = []
    for span in sentence_spans(text):
        if _count(tokenizer, text[span[0] : span[1]]) <= max_tokens:
            units.append(span)
        else:
            units.extend(_token_windows(text, span, tokenizer, max_tokens, overlap_tokens))

    chunks: list[Span] = []
    i = 0
    while i < len(units):
        j = i + 1  # a chunk always takes at least one unit
        while j < len(units) and _count(tokenizer, text[units[i][0] : units[j][1]]) <= max_tokens:
            j += 1
        chunks.append((units[i][0], units[j - 1][1]))
        if j == len(units):
            break
        # Overlap: trailing whole units of this chunk that fit the overlap budget; never all
        # of them, so every chunk starts later than the previous one.
        k = j
        while k - 1 > i and _count(tokenizer, text[units[k - 1][0] : units[j - 1][1]]) <= (
            overlap_tokens
        ):
            k -= 1
        i = k
    return chunks

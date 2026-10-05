"""data/chunking.py: normalisation, sentence spans, chunk packing and overlap."""

from __future__ import annotations

import itertools

import pytest

from data.chunking import chunk_spans, normalize_text, sentence_spans
from tests.unit.data.conftest import WhitespaceTokenizer


def sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


# ------------------------------------------------------------------------ normalisation
def test_normalize_collapses_whitespace_and_trims() -> None:
    assert normalize_text("  a\tb\n\n c\u00a0d  ") == "a b c d"


def test_normalize_applies_nfc_and_keeps_case() -> None:
    decomposed = "Café OPEN"  # e + combining acute
    assert normalize_text(decomposed) == "Café OPEN"


# ------------------------------------------------------------------------- sentences
def test_sentences_split_on_terminal_punctuation_before_capital_or_digit() -> None:
    text = "First one. Second! Third? 4th starts with a digit. done"
    assert sentences(text) == ["First one.", "Second!", "Third?", "4th starts with a digit. done"]


def test_closing_quotes_and_brackets_stay_with_their_sentence() -> None:
    text = 'He said "stop." Then (it ended.) "Quoted" next.'
    assert sentences(text) == ['He said "stop."', "Then (it ended.)", '"Quoted" next.']


def test_no_split_before_lowercase_or_inside_numbers() -> None:
    assert sentences("Pay 3.5 dollars. e.g. this stays") == ["Pay 3.5 dollars. e.g. this stays"]


def test_sentence_spans_cover_text_without_dropping_characters() -> None:
    text = 'A "b." C d! E (f.) G'
    spans = sentence_spans(text)
    rebuilt = " ".join(text[s:e] for s, e in spans)
    assert rebuilt == text


def test_empty_and_unterminated_text() -> None:
    assert sentence_spans("") == []
    assert sentences("no terminal punctuation") == ["no terminal punctuation"]


# ---------------------------------------------------------------------------- chunks
def make_text(n_sentences: int, words: int) -> str:
    return " ".join(
        "W " + " ".join(f"s{i}w{j}" for j in range(words - 1)) + "." for i in range(n_sentences)
    )


def test_every_chunk_respects_the_token_limit(ws_tokenizer: WhitespaceTokenizer) -> None:
    text = make_text(40, 7)
    for s, e in chunk_spans(text, ws_tokenizer, max_tokens=20, overlap_tokens=8):
        assert len(ws_tokenizer.offsets(text[s:e])) <= 20


def test_chunks_cover_every_sentence(ws_tokenizer: WhitespaceTokenizer) -> None:
    text = make_text(40, 7)
    chunks = chunk_spans(text, ws_tokenizer, max_tokens=20, overlap_tokens=8)
    for s, e in sentence_spans(text):
        assert any(cs <= s and e <= ce for cs, ce in chunks)


def test_chunk_boundaries_are_sentence_boundaries(ws_tokenizer: WhitespaceTokenizer) -> None:
    text = make_text(30, 5)
    starts = {s for s, _ in sentence_spans(text)}
    ends = {e for _, e in sentence_spans(text)}
    for s, e in chunk_spans(text, ws_tokenizer, max_tokens=12, overlap_tokens=5):
        assert s in starts
        assert e in ends


def test_overlap_is_whole_trailing_sentences_within_budget(
    ws_tokenizer: WhitespaceTokenizer,
) -> None:
    text = make_text(12, 5)  # 5 tokens per sentence
    chunks = chunk_spans(text, ws_tokenizer, max_tokens=15, overlap_tokens=6)
    assert chunks[0] == (0, sentence_spans(text)[2][1])  # 3 sentences = 15 tokens
    # next chunk starts at the 3rd sentence: one sentence (5 tokens) fits the 6-token overlap
    assert chunks[1][0] == sentence_spans(text)[2][0]
    for (_, prev_end), (start, _) in itertools.pairwise(chunks):
        overlap = text[start:prev_end] if start < prev_end else ""
        assert len(ws_tokenizer.offsets(overlap)) <= 6


def test_zero_overlap_gives_disjoint_chunks(ws_tokenizer: WhitespaceTokenizer) -> None:
    chunks = chunk_spans(make_text(12, 5), ws_tokenizer, max_tokens=15, overlap_tokens=0)
    for (_, prev_end), (start, _) in itertools.pairwise(chunks):
        assert start > prev_end


def test_chunk_starts_strictly_increase(ws_tokenizer: WhitespaceTokenizer) -> None:
    # overlap budget larger than any one sentence still makes progress
    chunks = chunk_spans(make_text(20, 4), ws_tokenizer, max_tokens=9, overlap_tokens=8)
    starts = [s for s, _ in chunks]
    assert starts == sorted(set(starts))


def test_short_text_is_one_chunk(ws_tokenizer: WhitespaceTokenizer) -> None:
    text = "One short. Two short."
    assert chunk_spans(text, ws_tokenizer, max_tokens=100, overlap_tokens=10) == [(0, len(text))]


def test_long_sentence_is_split_into_token_windows(ws_tokenizer: WhitespaceTokenizer) -> None:
    text = "Short one. " + " ".join(f"w{i}" for i in range(50)) + ". Tail sentence."
    chunks = chunk_spans(text, ws_tokenizer, max_tokens=20, overlap_tokens=5)
    assert all(len(ws_tokenizer.offsets(text[s:e])) <= 20 for s, e in chunks)
    covered = set()
    for s, e in chunks:
        covered.update(range(s, e))
    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())


def test_empty_text_has_no_chunks(ws_tokenizer: WhitespaceTokenizer) -> None:
    assert chunk_spans("", ws_tokenizer, max_tokens=10, overlap_tokens=2) == []


def test_chunking_is_deterministic(ws_tokenizer: WhitespaceTokenizer) -> None:
    text = make_text(25, 6)
    assert chunk_spans(text, ws_tokenizer, 17, 6) == chunk_spans(text, ws_tokenizer, 17, 6)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_limits_rejected(
    ws_tokenizer: WhitespaceTokenizer, size: int, overlap: int
) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_spans("A b.", ws_tokenizer, size, overlap)

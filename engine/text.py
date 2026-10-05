"""Shared text utilities (CACHE_SPEC §2).

``normalize`` is used for exact cache matching and for the planner's and controller's
duplicate-query checks, so it lives outside any component package.
"""

from __future__ import annotations

import unicodedata

__all__ = ["normalize"]


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def _normalize_token(token: str) -> str:
    if any(ch.isdigit() for ch in token):
        # Tokens with digits keep their inner punctuation ("3.5", "covid-19", "gpt-4o");
        # only leading/trailing punctuation goes ("(2023)," -> "2023").
        start, end = 0, len(token)
        while start < end and _is_punct(token[start]):
            start += 1
        while end > start and _is_punct(token[end - 1]):
            end -= 1
        return token[start:end]
    return "".join(" " if _is_punct(ch) else ch for ch in token)


def normalize(query: str) -> str:
    """NFC -> lowercase -> punctuation replaced by a space (except inside tokens with
    digits) -> whitespace collapsed -> trimmed. ``"State-of-the-art COVID-19 (2023)!"`` ->
    ``"state of the art covid-19 2023"``.
    """
    text = unicodedata.normalize("NFC", query).lower()
    return " ".join(" ".join(_normalize_token(t) for t in text.split()).split())

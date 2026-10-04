"""Loaders for tests/fixtures (docs/TESTING.md §3).

mini_corpus.jsonl: one chunk per line — ``chunk_id, doc_id, text, metadata`` with
``metadata = {title, source, published_at, char_start, char_end}``; ``chunk_id`` follows
DATA.md §3 (``sha1(f"{doc_id}:{char_start}")[:16]``).
mini_tasks.jsonl: one task per line in the DATA.md §1 ``Task`` shape (``dataset = "mini"``);
the null question has ``gold_groups = []``. All content is fictional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.fakes.retriever import CorpusChunk

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class MiniTask:
    task_id: str
    dataset: str
    question: str
    answer: str | None
    question_type: str | None
    gold_groups: list[list[str]]
    split: str
    meta: dict[str, Any] = field(default_factory=dict)


def _read_jsonl(name: str) -> list[dict[str, Any]]:
    path = FIXTURES_DIR / name
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_mini_corpus() -> list[CorpusChunk]:
    return [CorpusChunk(**row) for row in _read_jsonl("mini_corpus.jsonl")]


def load_mini_tasks() -> list[MiniTask]:
    return [MiniTask(**row) for row in _read_jsonl("mini_tasks.jsonl")]


def load_injection_doc() -> str:
    return (FIXTURES_DIR / "injection_doc.txt").read_text(encoding="utf-8")

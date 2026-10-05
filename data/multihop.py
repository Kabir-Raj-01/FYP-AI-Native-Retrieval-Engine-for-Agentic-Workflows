"""MultiHop-RAG raw-file loader (DATA.md §2; ADR-003 field names).

Texts are returned normalised (``data.chunking.normalize_text``), so chunk spans and gold
facts are matched in the same representation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.chunking import normalize_text
from data.ids import doc_id_for

__all__ = [
    "CORPUS_FILE",
    "DATASET",
    "QUERIES_FILE",
    "RawDoc",
    "RawEvidence",
    "RawQuery",
    "load_corpus",
    "load_queries",
    "task_id_for",
]

DATASET = "multihop"
# File names and fields of the dataset format (ADR-003); checksums live in data.sources.
CORPUS_FILE = "corpus.json"
QUERIES_FILE = "MultiHopRAG.json"


@dataclass(frozen=True)
class RawDoc:
    doc_id: str
    url: str
    title: str
    source: str
    published_at: str
    author: str | None
    category: str | None
    text: str  # normalised `body`


@dataclass(frozen=True)
class RawEvidence:
    url: str
    title: str
    fact: str  # normalised


@dataclass(frozen=True)
class RawQuery:
    task_id: str
    question: str
    answer: str
    question_type: str  # inference_query | comparison_query | temporal_query | null_query
    evidence: list[RawEvidence]


def task_id_for(index: int) -> str:
    """``multihop:<index>``: the queries file has no ids, so the 0-based position in the
    checksum-pinned file is the stable original id (DATA.md §1; Week 8 plan)."""
    return f"{DATASET}:{index:04d}"


def _read_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    return data


def _optional(value: Any) -> str | None:
    return None if value in (None, "", "None") else str(value)


def load_corpus(raw: Path) -> list[RawDoc]:
    return [
        RawDoc(
            doc_id=doc_id_for(item["url"]),
            url=item["url"],
            title=normalize_text(item["title"]),
            source=item["source"],
            published_at=item["published_at"],
            author=_optional(item.get("author")),
            category=_optional(item.get("category")),
            text=normalize_text(item["body"]),
        )
        for item in _read_list(raw / CORPUS_FILE)
    ]


def load_queries(raw: Path) -> list[RawQuery]:
    return [
        RawQuery(
            task_id=task_id_for(i),
            question=item["query"],
            answer=item["answer"],
            question_type=item["question_type"],
            evidence=[
                RawEvidence(
                    url=e["url"], title=normalize_text(e["title"]), fact=normalize_text(e["fact"])
                )
                for e in item["evidence_list"]
            ],
        )
        for i, item in enumerate(_read_list(raw / QUERIES_FILE))
    ]

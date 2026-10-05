"""Map gold evidence facts to chunk ids (DATA.md §4).

For each evidence entry: find its source document (url, else title); find the fact in the
normalised document text — every exact occurrence, else the best fuzzy alignment
(``rapidfuzz.fuzz.partial_ratio_alignment`` >= ``data.gold_fuzzy_threshold``); the entry's
group is every chunk that fully contains an occurrence, or — for an occurrence that
straddles a chunk boundary — every chunk it overlaps. Any member of a group satisfies the
entry (DATA.md §4). Entries that cannot be placed go to the unmapped list, not the map.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

from data.chunking import Span
from data.multihop import RawDoc, RawQuery

__all__ = ["ChunkRef", "GoldResult", "UnmappedEntry", "build_gold_map", "chunks_for_span"]


@dataclass(frozen=True)
class ChunkRef:
    chunk_id: str
    start: int
    end: int


@dataclass(frozen=True)
class UnmappedEntry:
    task_id: str
    evidence_index: int
    url: str
    title: str
    fact: str
    reason: str  # "NO_DOCUMENT" | "NO_MATCH"


@dataclass(frozen=True)
class GoldResult:
    gold_map: dict[str, list[list[str]]]  # task_id -> one group of chunk ids per entry
    unmapped: list[UnmappedEntry]
    entries: int  # evidence entries considered (null queries have none)
    exact: int
    fuzzy: int
    straddling: int  # entries placed via a straddling occurrence

    @property
    def mapped(self) -> int:
        return self.exact + self.fuzzy

    @property
    def coverage(self) -> float:
        return self.mapped / self.entries if self.entries else 1.0


def _occurrences(text: str, fact: str) -> list[Span]:
    spans: list[Span] = []
    start = text.find(fact)
    while start != -1:
        spans.append((start, start + len(fact)))
        start = text.find(fact, start + 1)
    return spans


def _fuzzy_occurrence(text: str, fact: str, threshold: int) -> Span | None:
    alignment = fuzz.partial_ratio_alignment(fact, text, score_cutoff=threshold)
    if alignment is None:
        return None
    return alignment.dest_start, alignment.dest_end


def chunks_for_span(chunks: Sequence[ChunkRef], span: Span) -> tuple[list[str], bool]:
    """Chunk ids fully containing ``span``, else those overlapping it; and whether it
    straddled (no single chunk contained it)."""
    start, end = span
    containing = [c.chunk_id for c in chunks if c.start <= start and end <= c.end]
    if containing:
        return containing, False
    return [c.chunk_id for c in chunks if c.start < end and start < c.end], True


def _dedupe(ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def build_gold_map(
    queries: Sequence[RawQuery],
    docs: Sequence[RawDoc],
    chunks_by_doc: Mapping[str, Sequence[ChunkRef]],
    fuzzy_threshold: int,
) -> GoldResult:
    by_url = {d.url: d for d in docs}
    by_title = {d.title: d for d in docs}
    gold: dict[str, list[list[str]]] = {}
    unmapped: list[UnmappedEntry] = []
    entries = exact = fuzzy = straddling = 0
    for query in queries:
        groups: list[list[str]] = []
        for index, ev in enumerate(query.evidence):
            entries += 1
            doc = by_url.get(ev.url) or by_title.get(ev.title)
            if doc is None:
                unmapped.append(
                    UnmappedEntry(query.task_id, index, ev.url, ev.title, ev.fact, "NO_DOCUMENT")
                )
                continue
            spans = _occurrences(doc.text, ev.fact)
            is_exact = bool(spans)
            if not spans and (found := _fuzzy_occurrence(doc.text, ev.fact, fuzzy_threshold)):
                spans = [found]
            if not spans:
                unmapped.append(
                    UnmappedEntry(query.task_id, index, ev.url, ev.title, ev.fact, "NO_MATCH")
                )
                continue
            ids: list[str] = []
            straddled = False
            for span in spans:
                found_ids, straddles = chunks_for_span(chunks_by_doc[doc.doc_id], span)
                ids.extend(found_ids)
                straddled = straddled or straddles
            groups.append(_dedupe(ids))
            exact += is_exact
            fuzzy += not is_exact
            straddling += straddled
        gold[query.task_id] = groups
    return GoldResult(gold, unmapped, entries, exact, fuzzy, straddling)

"""data/gold.py: placing gold facts in chunks (DATA.md §4)."""

from __future__ import annotations

from data.gold import ChunkRef, build_gold_map, chunks_for_span
from data.multihop import RawDoc, RawEvidence, RawQuery

TEXT = "Alpha one two. Beta three four. Gamma five six. Delta seven eight."
# sentences: Alpha [0,14)  Beta [15,31)  Gamma [32,47)  Delta [48,66)
CHUNKS = [ChunkRef("c0", 0, 31), ChunkRef("c1", 15, 48), ChunkRef("c2", 32, 66)]


def doc(text: str = TEXT, url: str = "u1", title: str = "T1") -> RawDoc:
    return RawDoc("d1", url, title, "src", "2024-01-01", None, None, text)


def query(
    *facts: str, url: str = "u1", title: str = "T1", task_id: str = "multihop:0000"
) -> RawQuery:
    ev = [RawEvidence(url, title, f) for f in facts]
    return RawQuery(task_id, "q?", "a", "inference_query", ev)


def gold_for(*facts: str, threshold: int = 90, **kw: str) -> tuple[list[list[str]], object]:
    result = build_gold_map([query(*facts, **kw)], [doc()], {"d1": CHUNKS}, threshold)
    return result.gold_map["multihop:0000"], result


def test_fact_inside_one_chunk_maps_to_it() -> None:
    groups, result = gold_for("Alpha one two.")
    assert groups == [["c0"]]
    assert result.exact == 1  # type: ignore[attr-defined]


def test_fact_in_an_overlap_maps_to_every_chunk_containing_it() -> None:
    groups, _ = gold_for("Beta three four.")
    assert groups == [["c0", "c1"]]


def test_straddling_fact_maps_to_the_chunks_it_overlaps() -> None:
    assert chunks_for_span(CHUNKS, (40, 55)) == (["c2"], False)  # fully inside c2 (32-66)
    straddle = [ChunkRef("a", 0, 20), ChunkRef("b", 21, 40)]
    assert chunks_for_span(straddle, (15, 25)) == (["a", "b"], True)


def test_repeated_fact_maps_to_chunks_of_every_occurrence() -> None:
    text = "Same fact here. Other words. Same fact here."
    chunks = [ChunkRef("x", 0, 15), ChunkRef("y", 16, 44)]
    result = build_gold_map([query("Same fact here.")], [doc(text)], {"d1": chunks}, 90)
    assert result.gold_map["multihop:0000"] == [["x", "y"]]


def test_one_group_per_evidence_entry() -> None:
    groups, result = gold_for("Alpha one two.", "Delta seven eight.")
    assert groups == [["c0"], ["c2"]]
    assert result.entries == 2  # type: ignore[attr-defined]


def test_fuzzy_fallback_above_threshold() -> None:
    groups, result = gold_for("Gamma fiv six.")  # one character missing
    assert groups == [["c1", "c2"]]
    assert (result.exact, result.fuzzy) == (0, 1)  # type: ignore[attr-defined]


def test_no_match_below_threshold_is_unmapped() -> None:
    groups, result = gold_for("Completely unrelated sentence about zebras.")
    assert groups == []
    assert [u.reason for u in result.unmapped] == ["NO_MATCH"]  # type: ignore[attr-defined]
    assert result.coverage == 0.0  # type: ignore[attr-defined]


def test_document_found_by_title_when_url_differs() -> None:
    groups, _ = gold_for("Alpha one two.", url="other-url", title="T1")
    assert groups == [["c0"]]


def test_missing_document_is_unmapped() -> None:
    groups, result = gold_for("Alpha one two.", url="nope", title="nope")
    assert groups == []
    unmapped = result.unmapped[0]  # type: ignore[attr-defined]
    assert (unmapped.reason, unmapped.evidence_index) == ("NO_DOCUMENT", 0)


def test_null_query_has_no_groups_and_full_coverage() -> None:
    result = build_gold_map([query()], [doc()], {"d1": CHUNKS}, 90)
    assert result.gold_map == {"multihop:0000": []}
    assert (result.entries, result.coverage) == (0, 1.0)

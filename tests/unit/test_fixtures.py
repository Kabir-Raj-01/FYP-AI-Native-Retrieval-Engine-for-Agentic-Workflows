"""tests/fixtures load and are internally consistent (TESTING §3)."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from engine.config import EngineConfig
from engine.state.models import RetrievalState
from tests.builders import make_evidence, make_gap
from tests.conftest import EngineFactory, StateFactory
from tests.fakes import CorpusChunk, FakeRetriever
from tests.fixture_data import MiniTask, load_injection_doc


def test_corpus_has_about_thirty_unique_chunks(mini_corpus: list[CorpusChunk]) -> None:
    assert 25 <= len(mini_corpus) <= 35
    ids = [c.chunk_id for c in mini_corpus]
    assert len(set(ids)) == len(ids)


def test_chunk_ids_follow_the_data_md_rule(mini_corpus: list[CorpusChunk]) -> None:
    for chunk in mini_corpus:
        start = chunk.metadata["char_start"]
        expected = hashlib.sha1(f"{chunk.doc_id}:{start}".encode()).hexdigest()[:16]
        assert chunk.chunk_id == expected
        assert chunk.metadata["char_end"] == int(str(start)) + len(chunk.text)
        assert {"title", "source", "published_at"} <= chunk.metadata.keys()


def test_five_tasks_with_the_required_mix(mini_tasks: list[MiniTask]) -> None:
    assert len(mini_tasks) == 5
    kinds = Counter(t.meta["kind"] for t in mini_tasks)
    assert kinds["exact_entity"] == 1
    assert kinds["paraphrase_heavy"] == 1
    assert kinds["null"] == 1
    assert len({t.task_id for t in mini_tasks}) == 5


def test_every_gold_chunk_exists_in_the_corpus(
    mini_tasks: list[MiniTask], mini_corpus: list[CorpusChunk]
) -> None:
    corpus_ids = {c.chunk_id for c in mini_corpus}
    for task in mini_tasks:
        if task.question_type == "null":
            assert task.gold_groups == []  # null questions have no supporting evidence
            continue
        assert len(task.gold_groups) == 2  # two-hop: two evidence groups
        for group in task.gold_groups:
            assert group
            assert set(group) <= corpus_ids


def test_gold_hops_come_from_different_documents(
    mini_tasks: list[MiniTask], mini_corpus: list[CorpusChunk]
) -> None:
    doc_of = {c.chunk_id: c.doc_id for c in mini_corpus}
    for task in mini_tasks:
        docs = {doc_of[cid] for group in task.gold_groups for cid in group}
        assert len(docs) == len(task.gold_groups)


def test_exact_entity_question_is_found_by_the_fake_retriever(
    mini_tasks: list[MiniTask], mini_corpus: list[CorpusChunk]
) -> None:
    task = next(t for t in mini_tasks if t.meta["kind"] == "exact_entity")
    results = FakeRetriever(mini_corpus).search("KX-41 drone", "dense", k=3)  # type: ignore[arg-type]
    assert results[0].chunk_id == task.gold_groups[0][0]


def test_corpus_loads_as_candidates_through_the_fake(mini_corpus: list[CorpusChunk]) -> None:
    assert FakeRetriever(mini_corpus).search("Linden Bay", "hybrid", k=30)  # type: ignore[arg-type]


def test_injection_doc_contains_planner_directed_instructions() -> None:
    text = load_injection_doc()
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in text
    assert "</system>" in text  # markup the wrapper must escape (SECURITY §3)


def test_fixture_files_are_utf8_with_lf_endings() -> None:
    fixtures = Path(__file__).parents[1] / "fixtures"
    for path in (p for p in fixtures.rglob("*") if p.is_file()):
        raw = path.read_bytes()
        raw.decode("utf-8")
        assert b"\r\n" not in raw, path.name


# --------------------------------------------------------------------- conftest fixtures
def test_cfg_fixture_uses_fake_llm_roles(cfg: EngineConfig) -> None:
    for role in ("planner", "verifier", "generator"):
        assert getattr(cfg.llm, role).provider == "fake"
        assert cfg.require_llm_model(role) == "fake-model"  # type: ignore[arg-type]


def test_cfg_fixture_is_a_fresh_copy_per_test(cfg: EngineConfig) -> None:
    cfg.retrieval.default_k = -1  # must not leak into other tests
    assert cfg.retrieval.default_k == -1


def test_cfg_fixture_was_not_mutated_by_the_previous_test(cfg: EngineConfig) -> None:
    assert cfg.retrieval.default_k > 0


def test_state_factory_builds_states_with_config_budget(
    state_factory: StateFactory, cfg: EngineConfig
) -> None:
    state = state_factory(
        gaps=[make_gap("g0")], evidence=[make_evidence("c1")], current_gap_id="g0"
    )
    assert isinstance(state, RetrievalState)
    assert state.budget.max_steps == cfg.budget_defaults.max_steps
    assert state.pending_query == "query for g0"
    assert [e.evidence_id for e in state.evidence] == ["c1"]


def test_tmp_state_db_is_an_unused_path(tmp_state_db: Path) -> None:
    assert tmp_state_db.suffix == ".sqlite3"
    assert not tmp_state_db.exists()
    assert tmp_state_db.parent.exists()


def test_engine_with_fakes_builds_a_controlled_engine(engine_with_fakes: EngineFactory) -> None:
    fake = engine_with_fakes()
    state = fake.engine.create_task("Linden Bay tram expansion", task_id="t-1")
    assert fake.engine.state("t-1") == state
    assert fake.parts.retriever is fake.retriever

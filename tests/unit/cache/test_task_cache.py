"""engine/cache/task_cache.py: CACHE_SPEC §7 tests, including the FR5 leakage test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from structlog.testing import capture_logs

from engine.cache.interfaces import Cache
from engine.cache.task_cache import TaskCache
from engine.config import EngineConfig
from engine.errors import TaskNotFound
from engine.state import sqlite as state_sqlite
from engine.state.models import CacheEntry, NeedType, RetrievalMode
from engine.text import normalize
from tests.builders import make_cache_entry, make_candidate
from tests.fakes import FakeEmbedder
from tests.unit.state.test_sqlite import add_task

DIM = 8
ACQ = "Who acquired the battery startup Voltrion?"
PARAPHRASE = "Which company bought Voltrion, the battery start-up?"
UNRELATED = "When did the tram expansion open?"


def unit(*xs: float) -> list[float]:
    v = np.zeros(DIM)
    v[: len(xs)] = xs
    return list(v / np.linalg.norm(v))


@pytest.fixture
def embedder() -> FakeEmbedder:
    # cos(ACQ, PARAPHRASE) = 0.96 (above the 0.92 threshold); cos(ACQ, UNRELATED) = 0.
    return FakeEmbedder(
        dimension=DIM,
        vectors={
            ACQ: unit(1.0),
            PARAPHRASE: unit(0.96, 0.28),
            UNRELATED: unit(0.0, 1.0),
        },
    )


@pytest.fixture
def cache(cfg: EngineConfig, tmp_state_db: Path, embedder: FakeEmbedder) -> TaskCache:
    cfg.paths.state_db = tmp_state_db
    conn = state_sqlite.connect(tmp_state_db)
    for task in ("t-1", "t-2"):
        add_task(conn, task)
    return TaskCache(cfg, embedder)


def stored(
    cache: TaskCache,
    embedder: FakeEmbedder,
    *,
    task: str = "t-1",
    query: str = ACQ,
    mode: RetrievalMode = RetrievalMode.DENSE,
    entry_id: str = "e1",
    n: int = 3,
    step: int = 1,
) -> CacheEntry:
    entry = make_cache_entry(
        task_id=task,
        entry_id=entry_id,
        normalized_query=normalize(query),
        query_vector=[float(x) for x in embedder.embed_queries([query])[0]],
        mode=mode,
        candidates=[make_candidate(f"c{i}", rank=i) for i in range(n)],
        created_at_step=step,
    )
    cache.put(entry)
    return entry


def test_task_cache_implements_the_interface(cache: TaskCache) -> None:
    assert isinstance(cache, Cache)


def test_exact_normalised_match_is_a_hit_with_similarity_one(
    cache: TaskCache, embedder: FakeEmbedder
) -> None:
    entry = stored(cache, embedder)
    embedder.calls.clear()
    hit = cache.lookup("t-1", "  who ACQUIRED the battery startup voltrion ", NeedType.SEMANTIC)
    assert hit is not None
    assert (hit.match, hit.similarity, hit.entry.entry_id) == ("exact", 1.0, entry.entry_id)
    assert embedder.calls == []  # an exact hit needs no embedding


def test_paraphrase_above_threshold_is_a_semantic_hit(
    cache: TaskCache, embedder: FakeEmbedder
) -> None:
    stored(cache, embedder)
    hit = cache.lookup("t-1", PARAPHRASE, NeedType.SEMANTIC)
    assert hit is not None
    assert hit.match == "semantic"
    assert hit.similarity == pytest.approx(0.96, abs=1e-6)


def test_below_threshold_is_a_miss(cache: TaskCache, embedder: FakeEmbedder) -> None:
    stored(cache, embedder)
    assert cache.lookup("t-1", UNRELATED, NeedType.SEMANTIC) is None


def test_threshold_is_inclusive_and_configurable(
    cache: TaskCache, embedder: FakeEmbedder, cfg: EngineConfig
) -> None:
    stored(cache, embedder)
    first = cache.lookup("t-1", PARAPHRASE, NeedType.SEMANTIC)
    assert first is not None
    cfg.cache.similarity_threshold = first.similarity  # exactly the computed similarity
    assert cache.lookup("t-1", PARAPHRASE, NeedType.SEMANTIC) is not None  # >= is a hit
    cfg.cache.similarity_threshold = min(1.0, first.similarity + 1e-6)
    assert cache.lookup("t-1", PARAPHRASE, NeedType.SEMANTIC) is None


def test_same_query_in_another_task_never_hits(cache: TaskCache, embedder: FakeEmbedder) -> None:
    # FR5 / TESTING §4.6 leakage test: identical query, different task_id -> miss.
    stored(cache, embedder, task="t-1")
    assert cache.lookup("t-2", ACQ, NeedType.SEMANTIC) is None
    assert cache.lookup("t-2", PARAPHRASE, NeedType.SEMANTIC) is None
    assert cache.lookup("t-1", ACQ, NeedType.SEMANTIC) is not None


@pytest.mark.parametrize(
    ("mode", "need", "serves"),
    [
        (RetrievalMode.DENSE, NeedType.SEMANTIC, True),
        (RetrievalMode.DENSE, NeedType.MIXED, True),
        (RetrievalMode.DENSE, NeedType.LEXICAL, False),
        (RetrievalMode.SPARSE, NeedType.LEXICAL, True),
        (RetrievalMode.SPARSE, NeedType.SEMANTIC, False),
        (RetrievalMode.SPARSE, NeedType.MIXED, False),
        (RetrievalMode.HYBRID, NeedType.SEMANTIC, True),
        (RetrievalMode.HYBRID, NeedType.MIXED, True),
        (RetrievalMode.HYBRID, NeedType.LEXICAL, True),
    ],
)
def test_mode_compatibility(
    cache: TaskCache, embedder: FakeEmbedder, mode: RetrievalMode, need: NeedType, serves: bool
) -> None:
    stored(cache, embedder, mode=mode)
    assert (cache.lookup("t-1", ACQ, need) is not None) is serves


def test_prefers_deeper_then_newer_entry(cache: TaskCache, embedder: FakeEmbedder) -> None:
    stored(cache, embedder, mode=RetrievalMode.DENSE, entry_id="shallow", n=2, step=1)
    stored(cache, embedder, mode=RetrievalMode.HYBRID, entry_id="deep", n=5, step=1)
    hit = cache.lookup("t-1", ACQ, NeedType.SEMANTIC)
    assert hit is not None and hit.entry.entry_id == "deep"


def test_replacement_and_eviction_through_put(
    cache: TaskCache, embedder: FakeEmbedder, cfg: EngineConfig
) -> None:
    cfg.cache.max_entries_per_task = 2
    stored(cache, embedder, entry_id="a", n=2, step=1)
    stored(cache, embedder, entry_id="b", n=4, step=2)  # same query + mode, deeper: replaces a
    stored(cache, embedder, query=PARAPHRASE, entry_id="c", step=3)
    stored(cache, embedder, query=UNRELATED, entry_id="d", step=4)  # 3 entries > cap 2
    conn = state_sqlite.connect(cfg.paths.state_db)
    assert [e.entry_id for e in state_sqlite.read_cache_entries(conn, "t-1")] == ["c", "d"]


def test_ablation_off_always_misses_without_reading_the_database(
    cache: TaskCache, embedder: FakeEmbedder, cfg: EngineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored(cache, embedder)
    cfg.ablation.cache = False

    def no_db(*args: object) -> None:
        raise AssertionError("the database must not be read with ablation.cache off")

    monkeypatch.setattr("engine.cache.task_cache.read_cache_entries", no_db)
    assert cache.lookup("t-1", ACQ, NeedType.SEMANTIC) is None


def test_hit_and_miss_events(cache: TaskCache, embedder: FakeEmbedder) -> None:
    stored(cache, embedder)
    with capture_logs() as logs:
        cache.lookup("t-1", ACQ, NeedType.SEMANTIC)
        cache.lookup("t-1", UNRELATED, NeedType.SEMANTIC)
    events = [(e["event"], e.get("match")) for e in logs]
    assert events == [("cache.hit", "exact"), ("cache.miss", None)]
    assert logs[0]["entry_id"] == "e1"


def test_put_for_an_unknown_task_is_task_not_found(
    cache: TaskCache, embedder: FakeEmbedder
) -> None:
    with pytest.raises(TaskNotFound) as info:
        stored(cache, embedder, task="no-such-task")
    assert info.value.task_id == "no-such-task"


def test_empty_cache_is_a_miss_without_embedding(cache: TaskCache, embedder: FakeEmbedder) -> None:
    assert cache.lookup("t-1", ACQ, NeedType.SEMANTIC) is None
    assert embedder.calls == []


def test_each_thread_gets_its_own_connection(cache: TaskCache, embedder: FakeEmbedder) -> None:
    from concurrent.futures import ThreadPoolExecutor

    stored(cache, embedder)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hits = list(pool.map(lambda _: cache.lookup("t-1", ACQ, NeedType.SEMANTIC), range(20)))
    assert all(h is not None and h.match == "exact" for h in hits)


def test_exact_hit_reads_only_matching_rows(
    cache: TaskCache, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored(cache, embedder, entry_id="a")
    stored(cache, embedder, query=UNRELATED, entry_id="b", step=2)
    calls: list[str | None] = []
    real = state_sqlite.read_cache_entries

    def spy(conn: object, task_id: str, normalized_query: str | None = None) -> object:
        calls.append(normalized_query)
        return real(conn, task_id, normalized_query)  # type: ignore[arg-type]

    monkeypatch.setattr("engine.cache.task_cache.read_cache_entries", spy)
    assert cache.lookup("t-1", ACQ, NeedType.SEMANTIC) is not None
    assert calls == [normalize(ACQ)]  # the full-task read happens only on an exact miss
    calls.clear()
    cache.lookup("t-1", PARAPHRASE, NeedType.SEMANTIC)
    assert calls == [normalize(PARAPHRASE), None]

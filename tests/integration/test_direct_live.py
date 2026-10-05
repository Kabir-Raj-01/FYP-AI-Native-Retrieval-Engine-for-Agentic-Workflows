"""FR3 / NFR-P1 against the real multihop_v1 collection (qdrant.exe + pinned models).

Skips unless ``qdrant.collection`` exists and holds every chunk of chunks.jsonl (run
``python -m data.ingest --dataset multihop --index`` first). NFR-P1's target (median dense
latency at k = 20 ≤ 200 ms, excluding reranking) is provisional until the Week 14 profile
(SRS TBD-4), so the median is printed and only a coarse ceiling is asserted.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from data.ingest import CHUNKS_FILE, processed_dir
from engine.api.app import create_app
from engine.config import load_config
from engine.core import Engine
from engine.state.models import RetrievalMode

pytestmark = pytest.mark.integration

QUERIES = [
    "Which company acquired a chip startup?",
    "What did the Federal Reserve decide about interest rates?",
    "Who won the football match on Sunday?",
    "Which streaming service raised its subscription price?",
    "What new features did the smartphone launch include?",
    "Which country reported higher inflation figures?",
    "What did the court rule in the antitrust case?",
    "Which electric car maker announced layoffs?",
]
NFR_P1_CEILING_MS = 2000.0  # coarse sanity bound, not the NFR target


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    cfg = load_config()
    chunks = processed_dir(cfg, "multihop") / CHUNKS_FILE
    if not chunks.exists():
        pytest.skip("run python -m data.ingest --dataset multihop first")
    with chunks.open(encoding="utf-8") as f:
        expected = sum(1 for line in f if line.strip())
    try:
        client = QdrantClient(url=cfg.qdrant.url, timeout=cfg.qdrant.timeout_s)
        count = client.count(cfg.qdrant.collection, exact=True).count
    except Exception as exc:
        pytest.skip(f"{cfg.qdrant.collection} not available: {exc}")
    if count != expected:
        pytest.skip(f"{cfg.qdrant.collection} holds {count}/{expected} points (index incomplete)")
    yield Engine.from_config(cfg)


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_direct_returns_k_results_for_every_mode(engine: Engine, mode: RetrievalMode) -> None:
    result = engine.retrieve_direct(QUERIES[0], mode, k=engine.cfg.retrieval.default_k)
    assert len(result.candidates) == engine.cfg.retrieval.default_k
    assert [c.rank for c in result.candidates] == list(range(len(result.candidates)))
    assert all(c.vector is not None for c in result.candidates)


def test_http_endpoint_end_to_end(engine: Engine) -> None:
    with TestClient(create_app(engine)) as client:
        health = client.get("/v1/health").json()
        assert (health["status"], health["qdrant"]) == ("ok", True)
        body = client.post(
            "/v1/retrieve/direct", json={"query": QUERIES[1], "mode": "hybrid", "k": 10}
        ).json()
        assert len(body["candidates"]) == 10
        assert all(c["vector"] is None for c in body["candidates"])


def test_nfr_p1_dense_latency_median(engine: Engine, capsys: pytest.CaptureFixture[str]) -> None:
    engine.retrieve_direct("warm-up query", RetrievalMode.DENSE, k=20)
    latencies = []
    for _ in range(3):
        for query in QUERIES:
            # distinct text per call, so the query-embedding LRU does not hide model time
            text = f"{query} ({len(latencies)})"
            latencies.append(engine.retrieve_direct(text, RetrievalMode.DENSE, k=20).latency_ms)
    median = statistics.median(latencies)
    with capsys.disabled():
        print(
            f"\nNFR-P1 dense k=20: median {median:.1f} ms, p90 "
            f"{sorted(latencies)[int(len(latencies) * 0.9)]:.1f} ms over {len(latencies)} "
            "queries (target <= 200 ms)"
        )
    assert median < NFR_P1_CEILING_MS

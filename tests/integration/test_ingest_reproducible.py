"""FR1 acceptance (PRD): ingestion is reproducible on the real MultiHop-RAG files.

Step 1 run twice from a clean state (fresh data folders, the pinned BGE tokenizer) writes
byte-identical chunks, gold map, unmapped list and split; they equal the local outputs and
the committed ``split.json`` / ``gold_map.json``, and the Qdrant collection holds one
point per chunk. Skips without the raw files (``python -m data.download``); the point-count
check skips without a running Qdrant.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from data.download import raw_dir
from data.ingest import (
    GOLD_MAP_FILE,
    MANIFEST_FILE,
    SPLIT_FILE,
    HFTokenizer,
    ingest,
    processed_dir,
)
from engine.config import load_config

pytestmark = pytest.mark.integration


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ingest_twice_is_identical_and_matches_the_index(tmp_path: Path) -> None:
    cfg = load_config()
    raw = raw_dir(cfg, "multihop")
    if not (raw / "corpus.json").exists():
        pytest.skip("run python -m data.download first")
    tokenizer = HFTokenizer(cfg.chunking.tokenizer, cfg.chunking.tokenizer_revision)
    runs = []
    for i in (1, 2):
        run_cfg = cfg.model_copy(deep=True)
        run_cfg.paths.data_dir = tmp_path / f"run{i}"
        shutil.copytree(raw, raw_dir(run_cfg, "multihop"))
        runs.append(
            ingest(
                run_cfg, "multihop", tokenizer, tokenizer_name=tokenizer.name, log=lambda _: None
            )
        )
    first, second = (r.manifest["outputs_sha256"] for r in runs)
    assert first == second  # chunks (ids), gold map, unmapped list, split
    assert runs[0].manifest["counts"] == runs[1].manifest["counts"]

    local = processed_dir(cfg, "multihop")
    for name in (SPLIT_FILE, GOLD_MAP_FILE):  # committed to git (DATA §5)
        assert _sha256(local / name) == first[name]
    if (local / MANIFEST_FILE).exists():  # outputs the index was built from
        manifest = json.loads((local / MANIFEST_FILE).read_text("utf-8"))
        assert manifest["outputs_sha256"] == first

    try:
        client = QdrantClient(url=cfg.qdrant.url, timeout=cfg.qdrant.timeout_s)
        points = client.count(cfg.qdrant.collection, exact=True).count
    except Exception as exc:
        pytest.skip(f"{cfg.qdrant.collection} not available for the point count: {exc}")
    assert points == runs[0].chunks

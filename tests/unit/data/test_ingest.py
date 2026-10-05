"""data/multihop.py, data/ingest.py and scripts/check_gold_mapping.py on synthetic raw files."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from data import ingest as ingest_module
from data.download import DownloadError, raw_dir
from data.ids import chunk_id_for, doc_id_for
from data.ingest import ingest, processed_dir
from data.multihop import load_corpus, load_queries, task_id_for
from engine.config import DataSource, EngineConfig
from engine.errors import ConfigError
from scripts import check_gold_mapping
from tests.unit.data.conftest import WhitespaceTokenizer

BODY_1 = " ".join(f"Sentence {i} of the first article about Kestrel drones." for i in range(30))
BODY_2 = "The Elsmoor\tcollective grows apricots.  It won a prize in 2025. " * 10
CORPUS = [
    {
        "title": "Drones",
        "author": "None",
        "source": "Ledger",
        "published_at": "2025-01-01",
        "category": "tech",
        "url": "https://x.example/drones",
        "body": BODY_1,
    },
    {
        "title": "Farms",
        "author": "A. Writer",
        "source": "Courier",
        "published_at": "2025-02-01",
        "category": "food",
        "url": "https://x.example/farms",
        "body": BODY_2,
    },
]
QUERIES = [
    {
        "query": "Q1?",
        "answer": "a",
        "question_type": "inference_query",
        "evidence_list": [
            {
                "fact": "Sentence 3 of the first article about Kestrel drones.",
                "title": "Drones",
                "url": "https://x.example/drones",
                "source": "Ledger",
                "published_at": "2025-01-01",
                "author": "None",
                "category": "tech",
            },
            {
                "fact": "The Elsmoor collective grows apricots.",
                "title": "Farms",
                "url": "https://x.example/farms",
                "source": "Courier",
                "published_at": "2025-02-01",
                "author": "A. Writer",
                "category": "food",
            },
        ],
    },
    {
        "query": "Q2?",
        "answer": "b",
        "question_type": "comparison_query",
        "evidence_list": [
            {
                "fact": "Not in any article at all, really.",
                "title": "Drones",
                "url": "https://x.example/drones",
                "source": "Ledger",
                "published_at": "2025-01-01",
                "author": "None",
                "category": "tech",
            },
        ],
    },
] + [
    {
        "query": f"N{i}?",
        "answer": "Insufficient information.",
        "question_type": "null_query",
        "evidence_list": [],
    }
    for i in range(8)
]
NOW = datetime(2026, 10, 5, tzinfo=UTC)


@pytest.fixture
def ing_cfg(cfg: EngineConfig, tmp_path: Path) -> EngineConfig:
    cfg.paths.data_dir = tmp_path / "data"
    raw = raw_dir(cfg, "multihop")
    raw.mkdir(parents=True)
    files = {"corpus.json": CORPUS, "MultiHopRAG.json": QUERIES}
    hashes = {}
    for name, obj in files.items():
        data = json.dumps(obj).encode("utf-8")
        (raw / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    cfg.data.sources = {"multihop": DataSource(hf_repo="o/r", revision="a" * 40, files=hashes)}
    cfg.chunking.chunk_size_tokens = 40
    cfg.chunking.chunk_overlap_tokens = 10
    cfg.data.dev_subset_size = 3
    return cfg


def run(cfg: EngineConfig) -> ingest_module.IngestResult:
    return ingest(
        cfg,
        "multihop",
        WhitespaceTokenizer(),
        tokenizer_name="whitespace",
        now=lambda: NOW,
        git=lambda: {"git_commit": "abc", "git_dirty": False},
        log=lambda m: None,
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- loader
def test_loader_normalises_and_assigns_ids(ing_cfg: EngineConfig) -> None:
    docs = load_corpus(raw_dir(ing_cfg, "multihop"))
    assert docs[0].doc_id == doc_id_for("https://x.example/drones")
    assert docs[0].author is None  # the dataset writes the string "None"
    assert "\t" not in docs[1].text and "  " not in docs[1].text
    queries = load_queries(raw_dir(ing_cfg, "multihop"))
    assert [q.task_id for q in queries[:2]] == ["multihop:0000", "multihop:0001"]
    assert task_id_for(2555) == "multihop:2555"


# --------------------------------------------------------------------------- ingest
def test_ingest_writes_all_outputs(ing_cfg: EngineConfig) -> None:
    result = run(ing_cfg)
    out = processed_dir(ing_cfg, "multihop")
    assert sorted(p.name for p in out.iterdir()) == [
        "chunks.jsonl",
        "gold_map.json",
        "gold_unmapped.json",
        "ingest_manifest.json",
        "split.json",
    ]
    rows = [json.loads(line) for line in (out / "chunks.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == result.chunks
    for row in rows:
        meta = row["metadata"]
        assert row["chunk_id"] == chunk_id_for(row["doc_id"], meta["char_start"])
        assert meta["n_tokens"] <= ing_cfg.chunking.chunk_size_tokens
        assert meta["dataset"] == "multihop"
        assert len(row["text"]) == meta["char_end"] - meta["char_start"]


def test_gold_map_and_unmapped_entries(ing_cfg: EngineConfig) -> None:
    run(ing_cfg)
    out = processed_dir(ing_cfg, "multihop")
    gold = read_json(out / "gold_map.json")
    chunk_ids = {
        json.loads(line)["chunk_id"]
        for line in (out / "chunks.jsonl").read_text("utf-8").splitlines()
    }
    assert len(gold["multihop:0000"]) == 2
    assert all(set(g) <= chunk_ids for g in gold["multihop:0000"])
    assert gold["multihop:0001"] == []  # its only fact matches nothing
    assert gold["multihop:0002"] == []  # null query
    unmapped = read_json(out / "gold_unmapped.json")
    assert [(u["task_id"], u["reason"]) for u in unmapped] == [("multihop:0001", "NO_MATCH")]


def test_split_file_partitions_tasks(ing_cfg: EngineConfig) -> None:
    run(ing_cfg)
    split = read_json(processed_dir(ing_cfg, "multihop") / "split.json")
    all_ids = {task_id_for(i) for i in range(len(QUERIES))}
    assert set(split["dev"]) | set(split["test"]) == all_ids
    assert not set(split["dev"]) & set(split["test"])
    assert len(split["dev_subset"]) == 3
    assert split["seed"] == ing_cfg.data.split_seed


def test_manifest_records_provenance_and_hashes(ing_cfg: EngineConfig) -> None:
    result = run(ing_cfg)
    out = processed_dir(ing_cfg, "multihop")
    manifest = read_json(out / "ingest_manifest.json")
    assert manifest == result.manifest
    assert manifest["git_commit"] == "abc"
    assert manifest["created_at"] == NOW.isoformat()
    assert manifest["counts"]["documents"] == 2
    assert manifest["gold"]["coverage"] == pytest.approx(2 / 3, abs=1e-6)
    for name, digest in manifest["outputs_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_ingest_is_deterministic(ing_cfg: EngineConfig) -> None:
    run(ing_cfg)
    out = processed_dir(ing_cfg, "multihop")
    first = {p.name: p.read_bytes() for p in out.iterdir() if p.name != "ingest_manifest.json"}
    run(ing_cfg)
    second = {p.name: p.read_bytes() for p in out.iterdir() if p.name != "ingest_manifest.json"}
    assert first == second


def test_outputs_use_lf_and_no_bom(ing_cfg: EngineConfig) -> None:
    run(ing_cfg)
    for path in processed_dir(ing_cfg, "multihop").iterdir():
        data = path.read_bytes()
        assert b"\r\n" not in data and not data.startswith(b"\xef\xbb\xbf"), path.name


def test_raw_checksum_mismatch_stops_ingestion(ing_cfg: EngineConfig) -> None:
    (raw_dir(ing_cfg, "multihop") / "corpus.json").write_bytes(b"[]")
    with pytest.raises(DownloadError, match="differs"):
        run(ing_cfg)


def test_missing_raw_file_stops_ingestion(ing_cfg: EngineConfig) -> None:
    (raw_dir(ing_cfg, "multihop") / "MultiHopRAG.json").unlink()
    with pytest.raises(DownloadError, match="missing"):
        run(ing_cfg)


def test_unsupported_dataset_rejected(ing_cfg: EngineConfig) -> None:
    with pytest.raises(ConfigError, match="supports"):
        ingest(ing_cfg, "hotpotqa", WhitespaceTokenizer(), tokenizer_name="w", log=lambda m: None)


def test_oversized_chunk_is_an_error(
    ing_cfg: EngineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_module, "chunk_spans", lambda text, *a: [(0, len(text))])
    with pytest.raises(ValueError, match="tokens"):
        run(ing_cfg)


def test_git_state_reports_commit_and_handles_missing_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ingest_module.git_state()  # runs inside this repository
    assert state["git_commit"] is None or len(state["git_commit"]) == 40

    def no_git(*args: object, **kwargs: object) -> None:
        raise OSError("git not found")

    monkeypatch.setattr(ingest_module.subprocess, "run", no_git)
    assert ingest_module.git_state() == {"git_commit": None, "git_dirty": None}


def test_main_runs_ingest_with_the_configured_tokenizer(
    monkeypatch: pytest.MonkeyPatch, ing_cfg: EngineConfig
) -> None:
    calls: list[tuple[str, str]] = []

    class FakeHF(WhitespaceTokenizer):
        def __init__(self, name: str) -> None:
            self.name = name

    def fake_ingest(cfg: EngineConfig, dataset: str, tok: object, *, tokenizer_name: str) -> None:
        calls.append((dataset, tokenizer_name))

    monkeypatch.setattr(ingest_module, "load_config", lambda path: ing_cfg)
    monkeypatch.setattr(ingest_module, "HFTokenizer", FakeHF)
    monkeypatch.setattr(ingest_module, "ingest", fake_ingest)
    assert ingest_module.main(["--dataset", "multihop"]) == 0
    assert calls == [("multihop", ing_cfg.chunking.tokenizer)]


def test_main_returns_1_on_errors(
    monkeypatch: pytest.MonkeyPatch, ing_cfg: EngineConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    (raw_dir(ing_cfg, "multihop") / "corpus.json").unlink()
    monkeypatch.setattr(ingest_module, "load_config", lambda path: ing_cfg)
    monkeypatch.setattr(ingest_module, "HFTokenizer", lambda name: WhitespaceTokenizer())
    monkeypatch.setattr(WhitespaceTokenizer, "name", "ws", raising=False)
    assert ingest_module.main(["--dataset", "multihop"]) == 1
    assert "missing" in capsys.readouterr().err


# ---------------------------------------------------------------- check_gold_mapping
def test_check_gold_mapping_fails_below_threshold(ing_cfg: EngineConfig) -> None:
    run(ing_cfg)  # coverage 2/3 < 0.95
    lines: list[str] = []
    assert check_gold_mapping.check(ing_cfg, "multihop", out=lines.append) is False
    assert lines[0].startswith("multihop: 2/3 gold evidence entries mapped")
    assert lines[-1].startswith("FAIL")


def test_check_gold_mapping_passes_at_threshold(ing_cfg: EngineConfig) -> None:
    run(ing_cfg)
    ing_cfg.data.gold_min_coverage = 0.6
    lines: list[str] = []
    assert check_gold_mapping.check(ing_cfg, "multihop", out=lines.append) is True
    assert lines[-1].startswith("PASS")


def test_check_gold_mapping_missing_outputs(
    ing_cfg: EngineConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_gold_mapping, "load_config", lambda path: ing_cfg)
    assert check_gold_mapping.main(["multihop"]) == 1
    assert "run: python -m data.ingest" in capsys.readouterr().err

"""eval/run.py and eval/datasets/multihop.py end-to-end with fakes and temp folders."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engine.config import EngineConfig, ExperimentConfig
from engine.core import Engine
from engine.errors import ConfigError
from eval import run as eval_run
from eval.datasets.base import Task
from eval.datasets.multihop import load_tasks
from tests.fakes import CorpusChunk, FakeEmbedder, FakeRetriever

NOW = datetime(2026, 10, 5, 13, 30, 0, tzinfo=UTC)
CLEAN = {"git_commit": "0123456789abcdef", "git_dirty": False}


def experiment(**overrides: object) -> ExperimentConfig:
    fields: dict[str, object] = {
        "name": "dense_oneshot_dev",
        "system": "dense_oneshot",
        "dataset": "multihop",
        "split": "dev",
        "subset": True,
        "budget_mode": "equal",
        "retrieval_budgets": [1],
        "notes": "E0",
    }
    fields.update(overrides)
    return ExperimentConfig.model_validate(fields)


@pytest.fixture
def run_cfg(cfg: EngineConfig, tmp_path: Path) -> EngineConfig:
    cfg.paths.runs_dir = tmp_path / "runs"
    cfg.paths.data_dir = tmp_path / "data"
    cfg.experiment = experiment()
    cfg.eval.max_workers = 2
    return cfg


@pytest.fixture
def engine(run_cfg: EngineConfig, mini_corpus: list[CorpusChunk]) -> Engine:
    return Engine(run_cfg, retriever=FakeRetriever(mini_corpus, embedder=FakeEmbedder()))


def tasks(mini_corpus: list[CorpusChunk]) -> list[Task]:
    by_doc = {c.doc_id: c.chunk_id for c in mini_corpus}
    return [
        Task(
            task_id="multihop:0000",
            dataset="multihop",
            question="KX-41 drone Talvane lab",
            answer="x",
            question_type="inference_query",
            gold_groups=[[by_doc["mini-d01"]], [by_doc["mini-d02"]]],
            split="dev",
        ),
        Task(
            task_id="multihop:0001",
            dataset="multihop",
            question="aquarium opening visitors",
            answer="Insufficient information.",
            question_type="null_query",
            gold_groups=[],
            split="dev",
        ),
    ]


def run(cfg: EngineConfig, engine: Engine, mini: list[CorpusChunk], **kw: object) -> Path:
    return eval_run.run_experiment(
        cfg, engine=engine, tasks=tasks(mini), now=lambda: NOW, git=lambda: CLEAN, **kw
    )  # type: ignore[arg-type]


def test_run_folder_layout_and_metrics(
    run_cfg: EngineConfig, engine: Engine, mini_corpus: list[CorpusChunk]
) -> None:
    run_dir = run(run_cfg, engine, mini_corpus)
    assert run_dir.name == "20261005-133000_dense_oneshot_dev_0123456"
    names = {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()}
    assert {
        "manifest.json",
        "config.resolved.yaml",
        "events.jsonl",
        "b1/per_task.csv",
        "b1/metrics.json",
        "b1/traces/multihop%3A0000.jsonl",
        "b1/bundles/multihop%3A0001.json",
    } <= names
    metrics = json.loads((run_dir / "b1" / "metrics.json").read_text("utf-8"))
    assert (metrics["tasks"], metrics["tasks_with_gold"]) == (2, 1)
    assert 0.0 <= metrics["quality"]["recall@10"] <= 1.0
    with (run_dir / "b1" / "per_task.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["task_id"] for r in rows] == ["multihop:0000", "multihop:0001"]
    assert rows[1]["has_gold"] == "0"
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text("utf-8").splitlines()
    ]
    assert {"run.started", "retrieval.call", "run.finished"} <= {e["event"] for e in events}
    # every line joins to the run; retrieval calls also to their task's trace step
    assert {e["run_id"] for e in events} == {run_dir.name}
    calls = [e for e in events if e["event"] == "retrieval.call"]
    assert sorted((e["task_id"], e["step"]) for e in calls) == [
        ("multihop:0000", 1),
        ("multihop:0001", 1),
    ]
    assert all(e["data"]["mode"] == "dense" for e in calls)


def test_manifest_is_complete_and_not_reportable_without_a_reranker(
    run_cfg: EngineConfig, engine: Engine, mini_corpus: list[CorpusChunk]
) -> None:
    manifest = json.loads((run(run_cfg, engine, mini_corpus) / "manifest.json").read_text("utf-8"))
    for key in (
        "run_id",
        "experiment",
        "system",
        "dataset",
        "split",
        "created_at",
        "git_commit",
        "git_dirty",
        "config_hash",
        "cli_overrides",
        "reportable",
        "python_version",
        "package_versions",
        "qdrant_server_version",
        "collection",
        "split_hash",
        "gold_map_hash",
        "prompt_hashes",
        "models",
        "seeds",
        "retrieval_budgets",
        "repeats",
        "cache_namespace",
        "n_tasks",
        "aborted",
        "totals",
    ):
        assert key in manifest, key  # OBSERVABILITY §5
    assert manifest["reportable"] is False
    assert any("no reranker" in r for r in manifest["reportable_reasons"])
    assert manifest["aborted"] == {"aborted": False, "reason": None}
    assert manifest["n_tasks"] == 2


def test_overrides_and_dirty_tree_are_recorded(
    run_cfg: EngineConfig, engine: Engine, mini_corpus: list[CorpusChunk]
) -> None:
    run_dir = eval_run.run_experiment(
        run_cfg,
        engine=engine,
        tasks=tasks(mini_corpus),
        now=lambda: NOW,
        overrides=["retrieval.default_k=5"],
        git=lambda: {"git_commit": "abc", "git_dirty": True},
    )
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    assert manifest["cli_overrides"] == ["retrieval.default_k=5"]
    assert {"--set overrides used", "git tree dirty or unknown"} <= set(
        manifest["reportable_reasons"]
    )


def test_max_tasks_limits_the_run(
    run_cfg: EngineConfig, engine: Engine, mini_corpus: list[CorpusChunk]
) -> None:
    manifest = json.loads(
        (run(run_cfg, engine, mini_corpus, max_tasks=1) / "manifest.json").read_text("utf-8")
    )
    assert manifest["n_tasks"] == 1
    assert "--max-tasks used" in manifest["reportable_reasons"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"split": "test", "subset": False}, "frozen-config tag"),
        ({"system": "controller_on"}, "not implemented yet"),
    ],
)
def test_refused_experiments(
    run_cfg: EngineConfig,
    engine: Engine,
    mini_corpus: list[CorpusChunk],
    change: dict[str, object],
    message: str,
) -> None:
    run_cfg.experiment = experiment(**change)
    with pytest.raises(ConfigError, match=message):
        run(run_cfg, engine, mini_corpus)


def test_base_config_is_not_an_experiment(
    run_cfg: EngineConfig, engine: Engine, mini_corpus: list[CorpusChunk]
) -> None:
    run_cfg.experiment = None
    with pytest.raises(ConfigError, match="not an experiment"):
        run(run_cfg, engine, mini_corpus)


def test_backend_failure_aborts_and_still_writes_the_manifest(
    run_cfg: EngineConfig, mini_corpus: list[CorpusChunk]
) -> None:
    from engine.errors import BackendError

    failing = Engine(run_cfg, retriever=FakeRetriever(mini_corpus, fail_on_call=1))
    with pytest.raises(BackendError):
        run(run_cfg, failing, mini_corpus)
    (run_dir,) = list((run_cfg.paths.runs_dir).iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    assert manifest["aborted"]["aborted"] is True
    assert manifest["reportable"] is False


# ----------------------------------------------------------------------- task loader
def test_load_tasks_uses_split_subset_and_gold(run_cfg: EngineConfig) -> None:
    raw = run_cfg.paths.data_dir / "raw" / "multihop"
    processed = run_cfg.paths.data_dir / "processed" / "multihop"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    queries = [
        {
            "query": f"Q{i}?",
            "answer": f"A{i}",
            "question_type": "inference_query",
            "evidence_list": [],
        }
        for i in range(4)
    ]
    (raw / "MultiHopRAG.json").write_text(json.dumps(queries), encoding="utf-8")
    ids = [f"multihop:{i:04d}" for i in range(4)]
    (processed / "split.json").write_text(
        json.dumps({"dev": ids[:3], "test": ids[3:], "dev_subset": ids[:2]}), encoding="utf-8"
    )
    (processed / "gold_map.json").write_text(
        json.dumps({i: [["c"]] for i in ids}), encoding="utf-8"
    )
    assert [t.task_id for t in load_tasks(run_cfg, "dev", subset=True)] == ids[:2]
    dev = load_tasks(run_cfg, "dev", subset=False)
    assert [t.task_id for t in dev] == ids[:3]
    assert (dev[0].question, dev[0].gold_groups, dev[0].split) == ("Q0?", [["c"]], "dev")
    with pytest.raises(ValueError, match="dev split only"):
        load_tasks(run_cfg, "test", subset=True)

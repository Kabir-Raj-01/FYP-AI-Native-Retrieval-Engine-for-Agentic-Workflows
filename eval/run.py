"""Experiment runner, v0 (EVAL_PROTOCOL §1-§2; OBSERVABILITY §4-§5).

    python -m eval.run --config configs\\experiments\\dense_oneshot_dev.yaml [--set key=value]...

Writes ``<paths.runs_dir>/<run_id>/`` with ``run_id = YYYYMMDD-HHMMSS_<experiment>_<git7>``:
``manifest.json``, ``config.resolved.yaml``, ``events.jsonl`` and one ``b<B>/`` folder per
retrieval budget (``traces/``, ``bundles/``, ``per_task.csv``, ``metrics.json``).

v0 limits (recorded in the manifest): only ``dense_oneshot`` is implemented; the test split
is refused (EVAL_PROTOCOL §1.2 requires the frozen-config tag); answers are not generated
(ADR-004); without a reranker (ADR-005) runs are not reportable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from baselines import dense_oneshot
from baselines.base import BaselineContext
from data.ingest import GOLD_MAP_FILE, SPLIT_FILE, git_state, processed_dir
from engine import factory
from engine.config import EngineConfig, config_hash, load_config
from engine.core import Engine
from engine.errors import ConfigError, EngineError
from engine.monitor.logging import configure_logging, get_logger
from engine.state.models import EvidenceBundle, RunContext
from eval.datasets.base import Task
from eval.datasets.multihop import load_tasks
from eval.metrics import TaskMetrics, aggregate, task_metrics

__all__ = ["main", "run_experiment"]

SYSTEMS: dict[str, Callable[[Task, BaselineContext], EvidenceBundle]] = {
    dense_oneshot.SYSTEM: dense_oneshot.run_task,
}
# Distributions whose versions go into the manifest (OBSERVABILITY §5).
_PACKAGES = (
    "pydantic",
    "qdrant-client",
    "sentence-transformers",
    "fastembed",
    "torch",
    "transformers",
    "numpy",
    "langgraph",
    "fastapi",
)


RunTask = Callable[[Task, BaselineContext], EvidenceBundle]


def _bound(run_task: RunTask, ctx: BaselineContext) -> Callable[[Task], EvidenceBundle]:
    """``run_task`` with its context bound (a closure per sub-run, not per loop variable)."""

    def run(task: Task) -> EvidenceBundle:
        return run_task(task, ctx)

    return run


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def _qdrant_version(cfg: EngineConfig) -> str | None:
    try:
        version = httpx.get(cfg.qdrant.url, timeout=cfg.qdrant.timeout_s).json().get("version")
    except (httpx.HTTPError, ValueError):
        return None
    return str(version) if version is not None else None


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))  # LF, no BOM on Windows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        columns.extend(c for c in row if c not in columns)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    cfg: EngineConfig,
    *,
    engine: Engine,
    overrides: Sequence[str] = (),
    max_tasks: int | None = None,
    tasks: Sequence[Task] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    git: Callable[[], dict[str, Any]] = git_state,
    clock: Callable[[], float] = time.perf_counter,
) -> Path:
    """Run ``cfg.experiment`` and return the run folder.

    Raises:
        ConfigError: no experiment section, test split, or an unimplemented system.
    """
    exp = cfg.experiment
    if exp is None:
        raise ConfigError("not an experiment config (no `experiment` section)")
    if exp.split == "test":
        raise ConfigError("the test split needs the frozen-config tag (EVAL_PROTOCOL §1.2)")
    if exp.system not in SYSTEMS:
        raise ConfigError(f"system {exp.system!r} is not implemented yet (v0: dense_oneshot)")
    run_task = SYSTEMS[exp.system]
    started = clock()
    created = now()
    repo = git()
    commit = repo.get("git_commit")
    run_id = f"{created:%Y%m%d-%H%M%S}_{exp.name}_{(commit or 'nogit')[:7]}"
    run_dir = cfg.paths.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_text(
        run_dir / "config.resolved.yaml",
        yaml.safe_dump(cfg.model_dump(mode="json", by_alias=True), sort_keys=False),
    )
    configure_logging(cfg.logging, events_file=run_dir / "events.jsonl")
    log = get_logger("eval.run").bind(run_id=run_id)

    selected = list(tasks) if tasks is not None else load_tasks(cfg, exp.split, exp.subset)
    if max_tasks is not None:
        selected = selected[:max_tasks]
    budgets = (
        exp.retrieval_budgets
        if exp.budget_mode == "equal"
        else [cfg.eval.natural_max_retrieval_calls]
    )
    reasons: list[str] = []
    if repo.get("git_dirty") is not False:
        reasons.append("git tree dirty or unknown")
    if overrides:
        reasons.append("--set overrides used")
    if max_tasks is not None:
        reasons.append("--max-tasks used")
    if not engine.has_reranker:
        reasons.append(f"no reranker (ADR-005): {exp.system} ran without rerank")

    aborted: dict[str, Any] = {"aborted": False, "reason": None}
    wall_time_s = 0.0
    ctx = RunContext(run_id=run_id, system=exp.system)
    log.info("run.started", experiment=exp.name, tasks=len(selected), budgets=budgets)
    try:
        for budget in budgets:
            sub = run_dir / f"b{budget}"
            traces = factory.build_trace_writer(cfg, ctx, sub)
            bctx = BaselineContext(
                cfg=cfg, engine=engine, traces=traces, retrieval_budget=budget, now=now
            )
            with ThreadPoolExecutor(max_workers=cfg.eval.max_workers) as pool:
                bundles = list(pool.map(_bound(run_task, bctx), selected))
            rows: list[TaskMetrics] = [
                task_metrics(
                    t.task_id,
                    t.question_type,
                    t.gold_groups,
                    b,
                    traces.read_steps(t.task_id),
                    cfg.eval.recall_ks,
                )
                for t, b in zip(selected, bundles, strict=True)
            ]
            _write_csv(sub / "per_task.csv", [r.row() for r in rows])
            _write_text(sub / "metrics.json", json.dumps(aggregate(rows), indent=1) + "\n")
            log.info("run.subrun_done", budget=budget, tasks=len(rows))
    except (EngineError, OSError) as exc:
        aborted = {"aborted": True, "reason": f"{type(exc).__name__}: {exc}"}
        log.error(
            "error",
            type=type(exc).__name__,
            message=str(exc),
            component="eval.run",
            retryable=False,
        )
        raise
    finally:
        wall_time_s = round(clock() - started, 2)
        folder = processed_dir(cfg, exp.dataset)
        manifest = {
            "run_id": run_id,
            "experiment": exp.name,
            "system": exp.system,
            "dataset": exp.dataset,
            "split": exp.split,
            "created_at": created.isoformat(),
            "git_commit": commit,
            "git_dirty": repo.get("git_dirty"),
            "config_hash": config_hash(cfg),
            "cli_overrides": list(overrides),
            "reportable": not reasons and not aborted["aborted"],
            "reportable_reasons": reasons,
            "python_version": platform.python_version(),
            "package_versions": _package_versions(),
            "qdrant_server_version": _qdrant_version(cfg),
            "collection": cfg.qdrant.collection,
            "split_hash": _sha256(folder / SPLIT_FILE),
            "gold_map_hash": _sha256(folder / GOLD_MAP_FILE),
            "prompt_hashes": {},  # no prompts in v0 (planner / generator not built)
            "models": {
                "embedding": f"{cfg.embedding.model_name}@{cfg.embedding.revision}",
                "sparse": f"{cfg.sparse.model_name}@{cfg.sparse.revision}",
                "reranker": cfg.rerank.cross_encoder_model if engine.has_reranker else None,
                "planner": cfg.llm.planner.model,
                "generator": cfg.llm.generator.model,
            },
            "seeds": {"project": cfg.project.seed, "split": cfg.data.split_seed},
            "retrieval_budgets": budgets,
            "repeats": cfg.eval.repeats,
            "cache_namespace": cfg.llm.cache_namespace,
            "n_tasks": len(selected),
            "aborted": aborted,
            "totals": {"llm_calls": 0, "tokens": 0, "cost_usd": 0.0, "wall_time_s": wall_time_s},
            "notes": exp.notes,
        }
        _write_text(run_dir / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    log.info("run.finished", wall_time_s=wall_time_s)
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run", description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="debug override; makes the run non-reportable",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="debug: only the first N tasks; makes the run non-reportable",
    )
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, overrides=args.overrides)
        run_dir = run_experiment(
            cfg, engine=Engine.from_config(cfg), overrides=args.overrides, max_tasks=args.max_tasks
        )
    except (ConfigError, EngineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"run folder: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

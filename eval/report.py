"""Run reports and system comparisons (EVAL_PROTOCOL §5, §7).

    python -m eval.report --run <run_id> [--run <run_id> ...]

One run: ``<run>/report/report.md`` (metrics per budget, answers when ``eval.generator``
ran, stop-reason and action-sequence distributions) and figures (``quality_vs_calls.png``,
``stop_reasons.png``). Several runs: additionally ``<first run>/report/comparison.md`` —
each later run against the first on the tasks both have, per shared budget: paired
bootstrap CI of the mean difference (Recall@10, CER, partial coverage) and exact McNemar
(CER, answer correctness).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engine.config import EngineConfig, load_config
from eval.stats import mcnemar_exact, paired_bootstrap

__all__ = ["compare_runs", "main", "report_run"]

_COLUMNS = (
    "recall@5",
    "recall@10",
    "hit@10",
    "ndcg@10",
    "mrr",
    "cer@10",
    "cer",
    "partial_coverage",
    "trajectory_coverage",
    "premature_stop",
)
_PAIRED = ("recall@10", "cer@10", "cer", "partial_coverage")


def _fmt(v: Any) -> str:
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {row["task_id"]: row for row in csv.DictReader(f)}


def _budgets(run: Path) -> list[Path]:
    return sorted(
        (p for p in run.iterdir() if p.is_dir() and p.name[:1] == "b" and p.name[1:].isdigit()),
        key=lambda p: int(p.name[1:]),
    )


def _trajectories(sub: Path) -> tuple[Counter[str], Counter[str]]:
    stops: Counter[str] = Counter()
    sequences: Counter[str] = Counter()
    for f in sorted((sub / "traces").glob("*.jsonl")):
        steps = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]
        if steps:
            stops[str(steps[-1].get("stop_reason"))] += 1
            sequences[" > ".join(str(s["action"]) for s in steps)] += 1
    return stops, sequences


def report_run(run: Path) -> Path:
    manifest = _read_json(run / "manifest.json") or {}
    out = run / "report"
    out.mkdir(exist_ok=True)
    lines = [
        f"# Report — {run.name}",
        "",
        f"- system: `{manifest.get('system')}` · split: {manifest.get('split')} · "
        f"tasks: {manifest.get('n_tasks')} · reportable: {manifest.get('reportable')}"
        f" {manifest.get('reportable_reasons') or ''}",
        f"- git: {manifest.get('git_commit')} (dirty: {manifest.get('git_dirty')}) · "
        f"models: {manifest.get('models')}",
        f"- wall time: {manifest.get('totals', {}).get('wall_time_s')} s · LLM calls: "
        f"{manifest.get('totals', {}).get('llm_calls')}",
        "",
        "## Quality (tasks with gold) and efficiency",
        "",
        "| B | " + " | ".join(_COLUMNS) + " | calls | steps | latency med ms | acc | EM |",
        "|---|" + "---|" * (len(_COLUMNS) + 5),
    ]
    points: list[tuple[int, float, float]] = []
    stop_tables: dict[str, Counter[str]] = {}
    seq_lines: list[str] = []
    for sub in _budgets(run):
        m = _read_json(sub / "metrics.json")
        if m is None:
            continue
        q, e = m["quality"], m["efficiency"]
        a = _read_json(sub / "answers.json") or {}
        lines.append(
            f"| {sub.name[1:]} | "
            + " | ".join(_fmt(q.get(c)) for c in _COLUMNS)
            + f" | {_fmt(e['retrieval_calls_mean'])} | {_fmt(e['steps_mean'])}"
            + f" | {_fmt(e['latency_ms_median'])} | {_fmt(a.get('accuracy'))}"
            + f" | {_fmt(a.get('exact_match'))} |"
        )
        points.append((int(sub.name[1:]), e["retrieval_calls_mean"] or 0.0, q.get("cer") or 0.0))
        stops, seqs = _trajectories(sub)
        stop_tables[sub.name] = stops
        seq_lines += (
            [f"### {sub.name}", ""] + [f"- {n} x `{s}`" for s, n in seqs.most_common(8)] + [""]
        )
    lines += ["", "## Quality by question type (largest budget)", ""]
    subs = _budgets(run)
    last = _read_json(subs[-1] / "metrics.json") if subs else None
    if last:
        lines += ["| type | tasks | recall@10 | cer |", "|---|---|---|---|"]
        for t, v in last["quality_by_type"].items():
            lines.append(f"| {t} | {v['tasks']} | {_fmt(v['recall@10'])} | {_fmt(v['cer'])} |")
    lines += ["", "## Stop reasons", "", "| B | " + "stop reason → tasks |", "|---|---|"]
    for b, c in stop_tables.items():
        lines.append(f"| {b[1:]} | " + ", ".join(f"{k}: {v}" for k, v in c.most_common()) + " |")
    lines += ["", "## Most common action sequences", "", *seq_lines]
    (out / "report.md").write_bytes("\n".join(lines).encode("utf-8"))
    _figures(out, points, stop_tables)
    return out / "report.md"


def _figures(
    out: Path, points: list[tuple[int, float, float]], stops: dict[str, Counter[str]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if points:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.plot([p[1] for p in points], [p[2] for p in points], marker="o")
        for b, x, y in points:
            ax.annotate(f"B={b}", (x, y), textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("retrieval calls / task")
        ax.set_ylabel("CER")
        fig.tight_layout()
        fig.savefig(out / "quality_vs_calls.png", dpi=120)
        plt.close(fig)
    reasons = sorted({r for c in stops.values() for r in c})
    if reasons:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        width = 0.8 / max(1, len(stops))
        for i, (b, c) in enumerate(stops.items()):
            ax.bar(
                [j + i * width for j in range(len(reasons))],
                [c.get(r, 0) for r in reasons],
                width,
                label=b,
            )
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(reasons))])
        ax.set_xticklabels(reasons, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("tasks")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "stop_reasons.png", dpi=120)
        plt.close(fig)


def compare_runs(runs: Sequence[Path], cfg: EngineConfig) -> Path:
    base = runs[0]
    lines = [
        f"# Comparison against `{base.name}`",
        "",
        "d = other - base, mean over paired tasks with gold; 95 % paired bootstrap CI "
        f"({cfg.eval.bootstrap_resamples} resamples, seed {cfg.project.seed}). McNemar exact "
        "p on CER (and on answer correctness when both runs have answers).",
        "",
    ]
    for other in runs[1:]:
        lines += [f"## `{other.name}`", ""]
        lines += ["| B | metric | n | base | other | d | 95 % CI | McNemar p |", "|" + "---|" * 8]
        shared = {s.name for s in _budgets(base)} & {s.name for s in _budgets(other)}
        for b in sorted(shared, key=lambda n: int(n[1:])):
            pa, pb = _csv(base / b / "per_task.csv"), _csv(other / b / "per_task.csv")
            ids = sorted(t for t in pa.keys() & pb.keys() if pa[t]["has_gold"] == "1")
            if not ids:
                continue
            for metric in _PAIRED:
                if metric not in pa[ids[0]] or metric not in pb[ids[0]]:
                    continue  # a run made before the metric existed
                xa = [float(pa[t][metric]) for t in ids]
                xb = [float(pb[t][metric]) for t in ids]
                d, lo, hi = paired_bootstrap(xa, xb, cfg.eval.bootstrap_resamples, cfg.project.seed)
                p = (
                    f"{mcnemar_exact([x == 1 for x in xa], [x == 1 for x in xb])[2]:.4f}"
                    if metric.startswith("cer")
                    else ""
                )
                lines.append(
                    f"| {b[1:]} | {metric} | {len(ids)} | {sum(xa) / len(xa):.3f} | "
                    f"{sum(xb) / len(xb):.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p} |"
                )
            aa, ab = _csv(base / b / "answers.csv"), _csv(other / b / "answers.csv")
            both = sorted(aa.keys() & ab.keys())
            if both:
                xa = [aa[t]["correct"] == "1" for t in both]
                xb = [ab[t]["correct"] == "1" for t in both]
                p_val = mcnemar_exact(xa, xb)[2]
                lines.append(
                    f"| {b[1:]} | accuracy | {len(both)} | {sum(xa) / len(xa):.3f} | "
                    f"{sum(xb) / len(xb):.3f} | {(sum(xb) - sum(xa)) / len(both):+.3f} | - | "
                    f"{p_val:.4f} |"
                )
        lines.append("")
    out = base / "report" / "comparison.md"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes("\n".join(lines).encode("utf-8"))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.report", description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="run id; repeatable")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    runs = [cfg.paths.runs_dir / r for r in args.run]
    missing = [str(r) for r in runs if not r.is_dir()]
    if missing:
        print(f"error: no such run folder: {', '.join(missing)}", file=sys.stderr)
        return 1
    for run in runs:
        print(report_run(run))
    if len(runs) > 1:
        print(compare_runs(runs, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""eval/report.py on a synthetic run folder (EVAL_PROTOCOL §7)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from engine.config import EngineConfig
from eval import report

METRICS = {
    "tasks": 3,
    "tasks_with_gold": 2,
    "quality": {"recall@5": 0.5, "recall@10": 0.75, "cer": 0.5},
    "quality_by_type": {"inference_query": {"tasks": 2, "recall@10": 0.75, "cer": 0.5}},
    "efficiency": {"retrieval_calls_mean": 2.0, "steps_mean": 3.0, "latency_ms_median": 9.0},
}


def make_run(root: Path, name: str, cer: list[int], correct: list[int]) -> Path:
    run = root / name
    sub = run / "b2"
    (sub / "traces").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"system": "s", "n_tasks": 3}), "utf-8")
    (sub / "metrics.json").write_text(json.dumps(METRICS), "utf-8")
    with (sub / "per_task.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "has_gold", "recall@10", "cer", "partial_coverage"])
        for i, c in enumerate(cer):
            w.writerow([f"t{i}", 1, c, c, c])
        w.writerow(["null", 0, "", "", ""])
    with (sub / "answers.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "correct"])
        w.writerows([f"t{i}", c] for i, c in enumerate(correct))
    steps = [
        {"action": "DENSE", "stop_reason": None},
        {"action": "STOP", "stop_reason": "ALL_GAPS_RESOLVED"},
    ]
    (sub / "traces" / "t0.jsonl").write_text(
        "\n".join(json.dumps(s) for s in steps) + "\n", "utf-8"
    )
    return run


def test_report_run_writes_tables_and_figures(tmp_path: Path) -> None:
    run = make_run(tmp_path, "a", [1, 0], [1, 0])
    text = report.report_run(run).read_text(encoding="utf-8")
    assert "| 2 | 0.500 | 0.750 |" in text
    assert "ALL_GAPS_RESOLVED: 1" in text
    assert "1 x `DENSE > STOP`" in text
    assert "| inference_query | 2 | 0.750 | 0.500 |" in text
    assert (run / "report" / "quality_vs_calls.png").exists()
    assert (run / "report" / "stop_reasons.png").exists()


def test_compare_runs_pairs_tasks_with_gold(tmp_path: Path, cfg: EngineConfig) -> None:
    a = make_run(tmp_path, "a", [0, 0, 0, 1], [0, 0, 1, 1])
    b = make_run(tmp_path, "b", [1, 1, 1, 1], [1, 1, 1, 1])
    cfg.eval.bootstrap_resamples = 200
    text = report.compare_runs([a, b], cfg).read_text(encoding="utf-8")
    assert "| 2 | cer | 4 | 0.250 | 1.000 | +0.750 |" in text
    assert "| 2 | accuracy | 4 | 0.500 | 1.000 | +0.500 | - | 0.5000 |" in text


def test_main_rejects_unknown_runs(tmp_path: Path, capsys: object) -> None:
    assert report.main(["--run", "no-such-run-folder"]) == 1

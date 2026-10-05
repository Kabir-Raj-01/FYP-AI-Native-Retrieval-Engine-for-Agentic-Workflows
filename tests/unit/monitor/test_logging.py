"""engine/monitor/logging.py: JSON event lines with ts and level (OBSERVABILITY §3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.config import EngineConfig
from engine.monitor.logging import bound_contextvars, configure_logging, get_logger


def test_json_event_line_has_the_observability_shape(
    cfg: EngineConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(cfg.logging)
    get_logger("test.component").info("retrieval.call", mode="dense", k=3)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    # OBSERVABILITY §3: ts, level, event, run_id, task_id, step, data
    assert list(line) == ["ts", "level", "event", "run_id", "task_id", "step", "data"]
    assert (line["event"], line["level"]) == ("retrieval.call", "info")
    assert (line["run_id"], line["task_id"], line["step"]) == (None, None, None)
    assert line["data"] == {"component": "test.component", "mode": "dense", "k": 3}
    assert line["ts"].endswith("Z")


def test_bound_ids_are_top_level_and_restored_on_exit(
    cfg: EngineConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(cfg.logging)
    log = get_logger("x")
    with bound_contextvars(run_id="r1", task_id="multihop:0001"):
        with bound_contextvars(step=2):
            log.info("inner")
        log.info("outer")
    log.info("after")
    lines = [json.loads(s) for s in capsys.readouterr().out.strip().splitlines()]
    assert [(e["run_id"], e["task_id"], e["step"]) for e in lines] == [
        ("r1", "multihop:0001", 2),
        ("r1", "multihop:0001", None),
        (None, None, None),
    ]
    assert all(e["data"] == {"component": "x"} for e in lines)


def test_logger_created_before_configuration_follows_it(cfg: EngineConfig, tmp_path: Path) -> None:
    early = get_logger("engine.core")  # e.g. the Engine, built before the run configures logs
    configure_logging(cfg.logging, events_file=tmp_path / "events.jsonl")
    early.info("retrieval.call", k=3)
    (line,) = (tmp_path / "events.jsonl").read_text("utf-8").splitlines()
    event = json.loads(line)
    assert (event["event"], event["data"]) == (
        "retrieval.call",
        {"component": "engine.core", "k": 3},
    )


def test_level_filters_lower_events(cfg: EngineConfig, capsys: pytest.CaptureFixture[str]) -> None:
    cfg.logging.level = "WARNING"
    configure_logging(cfg.logging)
    get_logger("x").info("hidden")
    get_logger("x").warning("shown")
    out = capsys.readouterr().out
    assert "hidden" not in out
    assert "shown" in out

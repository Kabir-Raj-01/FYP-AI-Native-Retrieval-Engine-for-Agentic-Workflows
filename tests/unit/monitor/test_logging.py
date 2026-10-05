"""engine/monitor/logging.py: JSON event lines with ts and level (OBSERVABILITY §3)."""

from __future__ import annotations

import json

import pytest

from engine.config import EngineConfig
from engine.monitor.logging import configure_logging, get_logger


def test_json_events_carry_ts_level_component_and_fields(
    cfg: EngineConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(cfg.logging)
    get_logger("test.component").info("retrieval.call", mode="dense", k=3)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["event"] == "retrieval.call"
    assert (line["level"], line["component"], line["mode"], line["k"]) == (
        "info",
        "test.component",
        "dense",
        3,
    )
    assert line["ts"].endswith("Z")


def test_level_filters_lower_events(cfg: EngineConfig, capsys: pytest.CaptureFixture[str]) -> None:
    cfg.logging.level = "WARNING"
    configure_logging(cfg.logging)
    get_logger("x").info("hidden")
    get_logger("x").warning("shown")
    out = capsys.readouterr().out
    assert "hidden" not in out
    assert "shown" in out

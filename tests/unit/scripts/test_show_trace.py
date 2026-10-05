"""scripts/show_trace.py on traces written by the controlled loop (OBSERVABILITY §6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.config import EngineConfig
from engine.state.models import Action
from scripts import show_trace
from tests.conftest import EngineFactory
from tests.fakes import FakeController

GOAL = "Linden Bay tram expansion council harbour"


def test_formats_steps_plan_events_and_the_bundle(engine_with_fakes: EngineFactory) -> None:
    fake = engine_with_fakes(
        controller=FakeController([Action.DENSE, Action.TRANSFORM, Action.STOP])
    )
    fake.engine.create_task(GOAL, task_id="t-1")
    fake.engine.run("t-1")
    text = show_trace.format_trajectory(
        fake.parts.traces.read_steps("t-1"), fake.parts.traces.read_bundle("t-1")
    )
    assert "step 1  DENSE" in text and "step 2  TRANSFORM" in text and "step 3  STOP" in text
    assert f"query   {GOAL!r}  [dense, k=20]" in text
    assert "plan    init attempt 1: accepted" in text
    assert "plan    rewrite attempt 1: accepted" in text
    assert "STOP    LOW_MARGINAL_UTILITY" in text
    assert "bundle  controller_on  stop=LOW_MARGINAL_UTILITY" in text


def test_main_reads_a_run_folder(
    cfg: EngineConfig,
    engine_with_fakes: EngineFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = engine_with_fakes(controller=FakeController([Action.DENSE, Action.STOP]))
    fake.engine.create_task(GOAL, task_id="multihop:0001")
    fake.engine.run("multihop:0001")
    cfg.paths.runs_dir = tmp_path  # engine_with_fakes writes traces to tmp_path/run
    monkeypatch.setattr(show_trace, "load_config", lambda: cfg)
    assert show_trace.main(["run", "multihop:0001"]) == 0
    assert "step 2  STOP" in capsys.readouterr().out
    assert show_trace.main(["run", "multihop:9999"]) == 1
    assert "no trace" in capsys.readouterr().err
    assert show_trace.main(["missing-run", "x"]) == 1


def test_trace_root_picks_the_budget_sub_run(tmp_path: Path) -> None:
    for b in (2, 10):
        (tmp_path / "r" / f"b{b}" / "traces").mkdir(parents=True)
    assert show_trace.trace_root(tmp_path, "r", None).name == "b2"
    assert show_trace.trace_root(tmp_path, "r", 10).name == "b10"
    assert show_trace.trace_root(tmp_path, "server", None) == tmp_path / "server"

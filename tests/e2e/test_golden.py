"""Golden trajectories (TESTING §4.4): the action sequence, stop reason and accepted evidence
per mini task match a stored snapshot, with the real Controller, TrajectoryMonitor,
EvidenceManager and state-aware Reranker on fakes (planner, scorer, retriever, embedder).

    pytest -m e2e                       # check
    $env:UPDATE_GOLDEN = "1"; pytest -m e2e   # rewrite the snapshot (explain why in the PR)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from engine.config import EngineConfig
from engine.controller.policy import RuleUtilityController
from engine.evidence.manager import ThresholdEvidenceManager
from engine.monitor.trajectory import RuleTrajectoryMonitor
from engine.rerank.state_aware import StateAwareReranker
from tests.conftest import EngineFactory
from tests.fakes import FakeEmbedder
from tests.fixture_data import MiniTask
from tests.unit.graph.test_loop_real import overlap_scorer

pytestmark = pytest.mark.e2e

SNAPSHOT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "mini_tasks.json"


def trajectories(
    cfg: EngineConfig, engine_with_fakes: EngineFactory, tasks: list[MiniTask]
) -> dict[str, object]:
    cfg.evidence.relevance_threshold = 0.3  # the overlap scorer's scale
    embedder = FakeEmbedder()
    fake = engine_with_fakes(
        controller=RuleUtilityController(
            cfg, embedder, cfg.controller.default_latency_ms.model_dump()
        ),
        evidence=ThresholdEvidenceManager(cfg, embedder),
        monitor=RuleTrajectoryMonitor(cfg),
        reranker=StateAwareReranker(cfg, embedder, overlap_scorer),
        embedder=embedder,
    )
    out: dict[str, object] = {}
    for task in tasks:
        fake.engine.create_task(task.question, task_id=task.task_id)
        bundle = fake.engine.run(task.task_id)
        steps = fake.parts.traces.read_steps(task.task_id)
        out[task.task_id] = {
            "actions": [s.action.value for s in steps],
            "rules": [s.rule for s in steps],
            "stop_reason": bundle.stop_reason.value,
            "evidence": [e.evidence_id for e in bundle.evidence],
        }
    return out


def test_golden_trajectories(
    cfg: EngineConfig, engine_with_fakes: EngineFactory, mini_tasks: list[MiniTask]
) -> None:
    got = trajectories(cfg, engine_with_fakes, mini_tasks)
    if os.environ.get("UPDATE_GOLDEN") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(got, indent=1, sort_keys=True) + "\n"
        SNAPSHOT.write_bytes(text.encode("utf-8"))  # LF, also on Windows
        pytest.skip(f"snapshot rewritten: {SNAPSHOT}")
    assert SNAPSHOT.exists(), "no snapshot yet: run once with UPDATE_GOLDEN=1"
    assert got == json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_controlled_path_logs_every_decision_event(
    cfg: EngineConfig, engine_with_fakes: EngineFactory, mini_tasks: list[MiniTask]
) -> None:
    """OBSERVABILITY §3 / CLAUDE.md rule 13 with the real controller, monitor, evidence
    manager and reranker: every kind of decision on these tasks reaches the event log."""
    with capture_logs() as logs:
        trajectories(cfg, engine_with_fakes, mini_tasks)
    events = {e["event"] for e in logs}
    assert events >= {
        "task.created",
        "gap.created",
        "cache.miss",
        "cache.hit",
        "cache.put",
        "controller.decision",
        "retrieval.call",
        "rerank.done",
        "evidence.accepted",
        "evidence.rejected",
        "gap.exhausted",
        "monitor.verdict",
        "task.stopped",
    }

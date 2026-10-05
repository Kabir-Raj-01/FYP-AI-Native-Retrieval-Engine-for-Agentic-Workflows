"""engine/monitor/trace.py: JSONL TraceWriter (OBSERVABILITY §1-§2, §4; ADR-009 §6)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pytest

from engine.monitor.interfaces import TraceWriter
from engine.monitor.trace import JsonlTraceWriter, task_file_stem
from engine.state.models import (
    Action,
    EvidenceBundle,
    RunContext,
    StopReason,
    TraceStep,
)
from tests.builders import NOW, make_budget, make_evidence, make_gap, make_scores

CTX = RunContext(run_id="20261005-120000_dense_oneshot_dev_abc1234", system="dense_oneshot")
TASK = "multihop:0042"


def step_fields(n: int, task: str = TASK, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "task_id": task,
        "step": n,
        "action": Action.DENSE,
        "rule": "utility_argmax",
        "eligible_actions": [Action.DENSE, Action.SPARSE_HYBRID],
        "utility_scores": {Action.DENSE: make_scores()},
        "query": "q",
        "mode": "dense",
        "k": 20,
        "cache_hit": False,
        "candidate_ids": ["c1", "c2"],
        "new_candidate_ids": ["c1", "c2"],
        "accepted_ids": ["c1"],
        "rejected": {"c2": "LOW_RELEVANCE"},
        "evicted_ids": [],
        "resolved_gap_ids": [],
        "evidence_gain": 0.4,
        "plan_events": [],
        "stagnation_count": 0,
        "latency_ms": {"total": 100.0, "execute": 80.0},
        "llm_calls": 0,
        "tokens": 0,
        "budget_after": make_budget(steps_used=n, retrieval_calls_used=n, reranker_calls_used=n),
        "timestamp": NOW,
    }
    fields.update(overrides)
    return fields


@pytest.fixture
def writer(tmp_path: Path) -> JsonlTraceWriter:
    return JsonlTraceWriter(tmp_path, CTX)


def bundle(system: str = CTX.system, task: str = TASK) -> EvidenceBundle:
    steps = [TraceStep(run_id=CTX.run_id, system=system, **step_fields(1, task))]
    return EvidenceBundle(
        task_id=task,
        system=system,
        goal="g",
        evidence=[make_evidence("c1")],
        gaps=[make_gap("g0")],
        stop_reason=StopReason.ONE_SHOT_COMPLETE,
        trace_summary=TraceWriter.summarize(steps),
    )


def test_new_step_stamps_the_run_context(writer: JsonlTraceWriter) -> None:
    step = writer.new_step(**step_fields(1))
    assert (step.run_id, step.system) == (CTX.run_id, CTX.system)
    assert writer.context == CTX


def test_steps_round_trip_in_order(writer: JsonlTraceWriter) -> None:
    written = [writer.new_step(**step_fields(n)) for n in (1, 2, 3)]
    for s in written:
        writer.write_step(s)
    assert writer.read_steps(TASK) == written


def test_file_is_one_json_object_per_line_with_lf_and_safe_name(
    writer: JsonlTraceWriter, tmp_path: Path
) -> None:
    writer.write_step(writer.new_step(**step_fields(1)))
    writer.write_step(writer.new_step(**step_fields(2)))
    path = tmp_path / "traces" / "multihop%3A0042.jsonl"
    data = path.read_bytes()
    assert b"\r\n" not in data and not data.startswith(b"\xef\xbb\xbf")
    lines = data.decode("utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]
    assert unquote(task_file_stem(TASK)) == TASK


def test_unknown_task_has_no_steps_and_no_bundle(writer: JsonlTraceWriter) -> None:
    assert writer.read_steps("nobody") == []
    assert writer.read_bundle("nobody") is None


@pytest.mark.parametrize("bad", [0, 2, 3])
def test_first_step_must_be_one(writer: JsonlTraceWriter, bad: int) -> None:
    with pytest.raises(ValueError, match="expected 1"):
        writer.write_step(writer.new_step(**step_fields(bad)))


def test_duplicate_and_skipped_steps_are_refused(writer: JsonlTraceWriter) -> None:
    writer.write_step(writer.new_step(**step_fields(1)))
    with pytest.raises(ValueError, match="expected 2"):
        writer.write_step(writer.new_step(**step_fields(1)))  # duplicate (e.g. a retry)
    with pytest.raises(ValueError, match="expected 2"):
        writer.write_step(writer.new_step(**step_fields(3)))  # gap
    assert len(writer.read_steps(TASK)) == 1  # refused steps were not written


def test_a_new_writer_continues_an_existing_trace(tmp_path: Path) -> None:
    first = JsonlTraceWriter(tmp_path, CTX)
    first.write_step(first.new_step(**step_fields(1)))
    second = JsonlTraceWriter(tmp_path, CTX)  # e.g. after a server restart
    with pytest.raises(ValueError, match="expected 2"):
        second.write_step(second.new_step(**step_fields(1)))
    second.write_step(second.new_step(**step_fields(2)))
    assert [s.step for s in second.read_steps(TASK)] == [1, 2]


@pytest.mark.parametrize("field", ["run_id", "system"])
def test_steps_of_another_run_are_refused(writer: JsonlTraceWriter, field: str) -> None:
    other = {"run_id": CTX.run_id, "system": CTX.system, field: "something-else"}
    with pytest.raises(ValueError, match="given to the writer"):
        writer.write_step(TraceStep(**other, **step_fields(1)))


def test_tasks_are_written_independently_and_in_parallel(writer: JsonlTraceWriter) -> None:
    tasks = [f"multihop:{i:04d}" for i in range(16)]

    def run_task(task: str) -> None:
        for n in range(1, 6):
            writer.write_step(writer.new_step(**step_fields(n, task)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run_task, tasks))
    for task in tasks:
        assert [s.step for s in writer.read_steps(task)] == [1, 2, 3, 4, 5]


def test_bundle_round_trip_without_vectors(writer: JsonlTraceWriter, tmp_path: Path) -> None:
    b = bundle()
    writer.write_bundle(b)
    stored = writer.read_bundle(TASK)
    assert stored is not None
    assert stored.model_dump(exclude={"evidence"}) == b.model_dump(exclude={"evidence"})
    assert stored.evidence[0].vector is None  # vectors are never written (OBSERVABILITY §2)
    raw = json.loads((tmp_path / "bundles" / "multihop%3A0042.json").read_text("utf-8"))
    assert raw["evidence"][0]["vector"] is None
    writer.write_bundle(b)  # rewriting is allowed (idempotent)
    assert not list((tmp_path / "bundles").glob("*.tmp"))


def test_bundle_of_another_system_is_refused(writer: JsonlTraceWriter) -> None:
    with pytest.raises(ValueError, match="bundle of system"):
        writer.write_bundle(bundle(system="controller_on"))


def test_unknown_schema_version_is_rejected(writer: JsonlTraceWriter, tmp_path: Path) -> None:
    writer.write_step(writer.new_step(**step_fields(1)))
    path = tmp_path / "traces" / "multihop%3A0042.jsonl"
    record = json.loads(path.read_text("utf-8"))
    record["schema_version"] = 2
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version 2"):
        writer.read_steps(TASK)


# ----------------------------------------------------------------------- summarize
def test_summary_counts_come_from_the_last_budget_and_the_steps() -> None:
    steps = [
        TraceStep(run_id="r", system="s", **step_fields(1, latency_ms={"total": 120.0})),
        TraceStep(
            run_id="r",
            system="s",
            **step_fields(
                2,
                action=Action.REUSE,
                cache_hit=True,
                latency_ms={"total": 5.0},
                budget_after=make_budget(
                    steps_used=2,
                    retrieval_calls_used=1,
                    llm_calls_used=1,
                    tokens_used=640,
                    reranker_calls_used=2,
                ),
            ),
        ),
        TraceStep(
            run_id="r",
            system="s",
            **step_fields(
                3,
                action=Action.STOP,
                latency_ms={"total": 1.5},
                budget_after=make_budget(
                    steps_used=3,
                    retrieval_calls_used=1,
                    llm_calls_used=1,
                    tokens_used=640,
                    reranker_calls_used=2,
                ),
            ),
        ),
    ]
    summary = TraceWriter.summarize(steps)
    assert (
        summary.steps,
        summary.retrieval_calls,
        summary.llm_calls,
        summary.tokens,
        summary.reranker_calls,
    ) == (3, 1, 1, 640, 2)
    assert summary.cache_hits == 1
    assert summary.total_latency_ms == pytest.approx(126.5)
    assert summary.actions == [Action.DENSE, Action.REUSE, Action.STOP]


def test_summary_of_no_steps_is_an_error() -> None:
    with pytest.raises(ValueError, match="no trace steps"):
        TraceWriter.summarize([])

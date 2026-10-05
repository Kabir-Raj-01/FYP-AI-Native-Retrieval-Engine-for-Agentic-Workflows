"""JSONL TraceWriter (OBSERVABILITY §1-§2, §4).

Layout under ``root`` (an eval sub-run folder ``eval/runs/<run_id>/b<B>/``, or the server's
trace folder):
- ``traces/<task>.jsonl``  one TraceStep per line, in step order
- ``bundles/<task>.json``  the terminal EvidenceBundle
``<task>`` is the task id percent-encoded (``multihop:0001`` -> ``multihop%3A0001``):
Windows forbids ``:`` in file names, and the encoding is reversible.

Files are UTF-8 with LF line endings; records are default dumps (no vectors). Each task file
has its own lock, so eval worker threads write different tasks in parallel; every appended
line is flushed. Reading rejects an unknown ``schema_version``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from urllib.parse import quote

from engine.monitor.interfaces import TraceWriter
from engine.state.models import EvidenceBundle, RunContext, TraceStep

__all__ = ["JsonlTraceWriter", "task_file_stem"]

TRACE_SCHEMA_VERSION = 1  # TraceStep.schema_version this code reads and writes


def task_file_stem(task_id: str) -> str:
    """File-system-safe, reversible name for ``task_id`` (``urllib.parse.unquote`` inverts it)."""
    return quote(task_id, safe="-_.")


class JsonlTraceWriter(TraceWriter):
    def __init__(self, root: Path, context: RunContext) -> None:
        self._root = root
        self._context = context
        self._last_step: dict[str, int] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @property
    def context(self) -> RunContext:
        return self._context

    def _trace_path(self, task_id: str) -> Path:
        return self._root / "traces" / f"{task_file_stem(task_id)}.jsonl"

    def _bundle_path(self, task_id: str) -> Path:
        return self._root / "bundles" / f"{task_file_stem(task_id)}.json"

    def _lock(self, task_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(task_id, threading.Lock())

    def write_step(self, step: TraceStep) -> None:
        if (step.run_id, step.system) != (self._context.run_id, self._context.system):
            raise ValueError(
                f"step of run {step.run_id!r}/{step.system!r} given to the writer of "
                f"{self._context.run_id!r}/{self._context.system!r}"
            )
        with self._lock(step.task_id):
            last = self._last_step.get(step.task_id)
            if last is None:  # first write for this task here: continue an existing file
                existing = self.read_steps(step.task_id)
                last = existing[-1].step if existing else 0
            if step.step != last + 1:
                raise ValueError(
                    f"task {step.task_id}: step {step.step} written after step {last} "
                    f"(expected {last + 1})"
                )
            path = self._trace_path(step.task_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(step.model_dump_json() + "\n")
                f.flush()
            self._last_step[step.task_id] = step.step

    def write_bundle(self, bundle: EvidenceBundle) -> None:
        if bundle.system != self._context.system:
            raise ValueError(
                f"bundle of system {bundle.system!r} given to the writer of "
                f"{self._context.system!r}"
            )
        path = self._bundle_path(bundle.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with self._lock(bundle.task_id):
            tmp.write_bytes((bundle.model_dump_json(indent=1) + "\n").encode("utf-8"))
            os.replace(tmp, path)  # atomic on the same volume: readers never see half a file

    def read_steps(self, task_id: str) -> list[TraceStep]:
        path = self._trace_path(task_id)
        if not path.exists():
            return []
        steps: list[TraceStep] = []
        with path.open(encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                step = TraceStep.model_validate_json(line)
                if step.schema_version != TRACE_SCHEMA_VERSION:
                    raise ValueError(
                        f"{path.name} line {number}: TraceStep schema_version "
                        f"{step.schema_version}, expected {TRACE_SCHEMA_VERSION} (ADR-009 §6)"
                    )
                steps.append(step)
        return steps

    def read_bundle(self, task_id: str) -> EvidenceBundle | None:
        path = self._bundle_path(task_id)
        if not path.exists():
            return None
        return EvidenceBundle.model_validate_json(path.read_text(encoding="utf-8"))

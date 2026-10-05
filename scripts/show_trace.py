"""Pretty-print one task's trajectory (OBSERVABILITY §6).

    python scripts\\show_trace.py <run_id> <task_id> [--budget B]

Reads ``<paths.runs_dir>/<run_id>/b<B>/traces/<task_id>.jsonl`` and the bundle next to it
(eval runs; ``--budget`` defaults to the only sub-run, or the first), or
``<paths.runs_dir>/server/`` for ``run_id`` ``server`` (the API server). Shows per step the
action and rule, the query (mode, k, cache hit), candidates (new), accepted / rejected
evidence, resolved gaps, evidence gain, plan events and LLM use, then the stop reason and
the bundle.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from engine import factory
from engine.config import load_config
from engine.state.models import EvidenceBundle, RunContext, TraceStep

__all__ = ["format_trajectory", "main", "trace_root"]


def trace_root(runs_dir: Path, run_id: str, budget: int | None) -> Path:
    run = runs_dir / run_id
    if run_id == "server" or (run / "traces").is_dir():
        return run
    subs = sorted(run.glob("b*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0)
    if budget is not None:
        return run / f"b{budget}"
    if not subs:
        raise FileNotFoundError(f"no traces under {run}")
    return subs[0]


def format_trajectory(steps: Sequence[TraceStep], bundle: EvidenceBundle | None) -> str:
    lines: list[str] = []
    for s in steps:
        head = f"step {s.step}  {s.action.value:<13} rule={s.rule}"
        if s.current_gap_id:
            head += f"  gap={s.current_gap_id}"
        lines.append(head)
        if s.query is not None:
            mode = s.mode.value if s.mode is not None else "-"
            hit = "  cache-hit" if s.cache_hit else ""
            lines.append(f"    query   {s.query!r}  [{mode}, k={s.k}]{hit}")
            lines.append(
                f"    cands   {len(s.candidate_ids)} ({len(s.new_candidate_ids)} new)"
                f"  accepted {len(s.accepted_ids)}  rejected {len(s.rejected)}"
                f"  gain {s.evidence_gain:.3f}"
            )
            if s.accepted_ids:
                lines.append(f"    + {', '.join(s.accepted_ids)}")
            reasons: dict[str, int] = {}
            for code in s.rejected.values():
                reasons[code] = reasons.get(code, 0) + 1
            if reasons:
                lines.append("    - " + ", ".join(f"{c} x{n}" for c, n in sorted(reasons.items())))
        if s.resolved_gap_ids:
            lines.append(f"    resolved {', '.join(s.resolved_gap_ids)}")
        for e in s.plan_events:
            reason = f" ({e.reason})" if e.reason else ""
            lines.append(
                f"    plan    {e.mode.value} attempt {e.attempt}: {e.status.value}{reason}"
            )
        if s.llm_calls:
            lines.append(f"    llm     {s.llm_calls} call(s), {s.tokens} tokens")
        if s.stop_reason is not None:
            lines.append(f"    STOP    {s.stop_reason.value}")
    if bundle is not None:
        t = bundle.trace_summary
        lines.append("")
        lines.append(
            f"bundle  {bundle.system}  stop={bundle.stop_reason.value}  steps={t.steps}"
            f"  retrieval={t.retrieval_calls}  cache_hits={t.cache_hits}  llm={t.llm_calls}"
        )
        for e in bundle.evidence:
            lines.append(f"    {e.final_score:.3f}  {e.evidence_id}  {e.summary[:80]!r}")
        for g in bundle.gaps:
            lines.append(f"    gap {g.gap_id} {g.status.value:<10} {g.description[:70]!r}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts\\show_trace.py", description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("task_id")
    parser.add_argument("--budget", type=int, default=None, help="sub-run b<B> of an eval run")
    args = parser.parse_args(argv)
    cfg = load_config()
    try:
        root = trace_root(cfg.paths.runs_dir, args.run_id, args.budget)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    reader = factory.build_trace_writer(cfg, RunContext(run_id=args.run_id, system="reader"), root)
    steps = reader.read_steps(args.task_id)
    if not steps:
        print(f"no trace for {args.task_id} under {root}", file=sys.stderr)
        return 1
    print(format_trajectory(steps, reader.read_bundle(args.task_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One live LLM call to check a role's provider and model (SETUP §2).

    python -m engine.llm.smoke --role planner [--config configs\\base.yaml]

``planner``: the real planner (factory-built, with validation) runs INIT on a fixed goal
and prints the outcome — ACCEPTED, or REJECTED attempts and the FALLBACK. ``generator`` /
``verifier``: one short text call. The response cache is off, so the provider is hit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from engine import factory
from engine.config import load_config
from engine.errors import EngineError
from engine.llm.interfaces import Message
from engine.state.models import Budget, PlanMode, PlanStatus, RetrievalState

GOAL = "Which company founded by Elon Musk launched a rocket in 2024, and where was it built?"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m engine.llm.smoke", description=__doc__)
    parser.add_argument("--role", choices=["planner", "generator", "verifier"], default="planner")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, overrides=["llm.cache_enabled=false"])
        model = cfg.require_llm_model(args.role)
        llm = factory.build_llm_client(cfg)
        if args.role == "planner":
            now = datetime.now(UTC)
            d = cfg.budget_defaults
            state = RetrievalState(
                task_id="smoke",
                goal=GOAL,
                budget=Budget(
                    max_steps=d.max_steps,
                    max_retrieval_calls=d.max_retrieval_calls,
                    max_llm_calls=d.max_llm_calls,
                    max_tokens=d.max_tokens,
                ),
                created_at=now,
                updated_at=now,
            )
            outcome = factory.build_planner(cfg, llm).plan(state, PlanMode.INIT)
            print(outcome.result.model_dump_json(indent=1))
            events = [(e.status.value, e.reason) for e in outcome.events]
            print(f"events: {events}")
            ok = outcome.events[-1].status == PlanStatus.ACCEPTED
        else:
            reply = llm.text(args.role, [Message(role="user", content="Reply with: OK")])
            print(reply.value)
            ok = bool(reply.value.strip())
    except EngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{'ok' if ok else 'FAILED'}: role={args.role} model={model}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

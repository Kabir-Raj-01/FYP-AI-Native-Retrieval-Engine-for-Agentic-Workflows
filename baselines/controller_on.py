"""controller_on: the full engine via ``Engine.run``, in-process (BASELINES_SPEC §2).

The task's ``max_retrieval_calls`` is the sub-run's budget B (EVAL_PROTOCOL §1, equal or
natural); the other caps are ``budget_defaults``. ``ctx.engine`` is the sub-run's Engine
(``Engine.for_run``: its own TaskStore and traces). Eval runs are not HTTP requests, so
``api.run_timeout_s`` does not apply to them.
"""

from __future__ import annotations

import math

from baselines.base import BaselineContext
from engine.state.models import EvidenceBundle
from eval.datasets.base import Task

__all__ = ["SYSTEM", "run_task"]

SYSTEM = "controller_on"


def run_task(task: Task, ctx: BaselineContext) -> EvidenceBundle:
    budget = ctx.engine.default_budget().model_copy(
        update={"max_retrieval_calls": ctx.retrieval_budget}
    )
    ctx.engine.create_task(task.question, budget, task.task_id)
    return ctx.engine.run(task.task_id, timeout_s=math.inf)

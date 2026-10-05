"""hybrid_oneshot baseline (BASELINES_SPEC §2): dense_oneshot with ``mode=hybrid``.

Traced as action SPARSE_HYBRID; same interim no-reranker behaviour as dense_oneshot.
"""

from __future__ import annotations

from baselines.base import BaselineContext
from baselines.dense_oneshot import run_oneshot
from engine.state.models import Action, EvidenceBundle, RetrievalMode
from eval.datasets.base import Task

__all__ = ["SYSTEM", "run_task"]

SYSTEM = "hybrid_oneshot"


def run_task(task: Task, ctx: BaselineContext) -> EvidenceBundle:
    return run_oneshot(task, ctx, RetrievalMode.HYBRID, Action.SPARSE_HYBRID)

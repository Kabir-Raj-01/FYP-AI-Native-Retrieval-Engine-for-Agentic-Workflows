"""baselines/hybrid_oneshot.py with fakes (BASELINES_SPEC §2)."""

from __future__ import annotations

from pathlib import Path

from baselines import hybrid_oneshot
from engine.config import EngineConfig
from engine.state.models import Action, RetrievalMode
from tests.fakes import CorpusChunk
from tests.unit.baselines.test_dense_oneshot import make_ctx, task


def test_one_hybrid_call_traced_as_sparse_hybrid(
    cfg: EngineConfig, tmp_path: Path, mini_corpus: list[CorpusChunk]
) -> None:
    ctx, retriever = make_ctx(cfg, tmp_path, mini_corpus)
    bundle = hybrid_oneshot.run_task(task(), ctx)
    assert [(r.mode, r.k) for r in retriever.requests] == [
        (RetrievalMode.HYBRID, cfg.retrieval.default_k)
    ]
    (step,) = ctx.traces.read_steps("multihop:0001")
    assert (step.action, step.mode, step.eligible_actions) == (
        Action.SPARSE_HYBRID,
        RetrievalMode.HYBRID,
        [Action.SPARSE_HYBRID],
    )
    assert bundle.evidence
    assert all(e.source_action == Action.SPARSE_HYBRID for e in bundle.evidence)
    assert bundle.trace_summary.actions == [Action.SPARSE_HYBRID]

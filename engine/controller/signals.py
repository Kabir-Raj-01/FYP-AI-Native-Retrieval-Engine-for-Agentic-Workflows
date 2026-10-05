"""Utility signals of the controller (ACTION_POLICY §4), one pure function each.

All values are in [0, 1]. Identifiers are ``gain``, ``cost``, ``redundancy_signal`` and
``latency_signal`` (ADR-009 "Also"); they become ``ActionScores(G, C, R, L, U)`` only in
``engine/controller/policy.py``. Vectors are L2-normalised; cosines are clipped to [0, 1].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from engine.config import EngineConfig
from engine.state.models import Action, CacheHit, LastResult, NeedType, RetrievalState

__all__ = [
    "ambiguity",
    "cost",
    "ema",
    "gain_rerank_expand",
    "gain_reuse",
    "gain_search",
    "gain_transform",
    "latency_signal",
    "max_cos",
    "redundancy_rerank_expand",
]


def max_cos(q: Sequence[float], others: Sequence[Sequence[float]]) -> float:
    """max over ``others`` of cos(q, o), clipped to [0, 1]; 0 if there are none."""
    if not others:
        return 0.0
    v = np.asarray(q, dtype=np.float64)
    m = np.asarray(others, dtype=np.float64)
    v = v / (np.linalg.norm(v) or 1.0)
    m = m / np.where((n := np.linalg.norm(m, axis=1, keepdims=True)) == 0.0, 1.0, n)
    return float(np.clip(m @ v, 0.0, 1.0).max())


def ema(state: RetrievalState, action: Action, cfg: EngineConfig) -> float:
    return state.action_gain_ema.get(action, cfg.controller.prior_action_gain)


def gain_search(
    action: Action,
    state: RetrievalState,
    cfg: EngineConfig,
    gap_alignment: float,
) -> float:
    """DENSE / SPARSE_HYBRID: s_gap·gap_align + s_ema·ema(a) + s_type·type_match(a)."""
    w = cfg.controller.gain_subweights
    row = getattr(cfg.controller.type_match, action.value)
    type_match: float = getattr(row, NeedType(state.current_need_type).value)
    return (
        w.gap_alignment * gap_alignment
        + w.recent_gain * ema(state, action, cfg)
        + w.type_match * type_match
    )


def gain_reuse(hit: CacheHit, state: RetrievalState) -> float:
    """similarity * (share of the hit's candidates not yet in E_t)."""
    n = len(hit.entry.candidates)
    if n == 0:
        return 0.0
    have = {e.chunk_id for e in state.evidence}
    n_new = sum(1 for c in hit.entry.candidates if c.chunk_id not in have)
    return hit.similarity * n_new / n


def ambiguity(last: LastResult, cfg: EngineConfig) -> float:
    """Fraction of ``last.top_relevances`` inside the ambiguous band (inclusive)."""
    if not last.top_relevances:
        return 0.0
    band = cfg.controller.ambiguous_band
    inside = sum(1 for r in last.top_relevances if band.low <= r <= band.high)
    return inside / len(last.top_relevances)


def gain_rerank_expand(state: RetrievalState, cfg: EngineConfig) -> float:
    last = state.last_result
    amb = ambiguity(last, cfg) if last is not None else 0.0
    return 0.5 * amb + 0.5 * ema(state, Action.RERANK_EXPAND, cfg)


def gain_transform(state: RetrievalState, cfg: EngineConfig, blocked_only: bool) -> float:
    """min(1, stagnation / patience); 1.0 if eligible only because retrieval is blocked."""
    if blocked_only:
        return 1.0
    return min(1.0, state.stagnation_count / cfg.monitor.stagnation_patience)


def cost(action: Action, state: RetrievalState, cfg: EngineConfig) -> float:
    """(cost_units[a] / max cost_units) * (0.5 + 0.5*pressure); pressure 0 when the
    ``budget_awareness`` ablation is off."""
    units = cfg.controller.cost_units.model_dump()
    top = max(units.values())
    pressure = state.budget.pressure() if cfg.ablation.budget_awareness else 0.0
    share = units[action.value] / top if top else 0.0
    return share * (0.5 + 0.5 * pressure)


def redundancy_rerank_expand(state: RetrievalState) -> float:
    """Fraction of ``last_result.candidate_ids`` already in E_t."""
    last = state.last_result
    if last is None or not last.candidate_ids:
        return 0.0
    have = {e.chunk_id for e in state.evidence}
    return sum(1 for c in last.candidate_ids if c in have) / len(last.candidate_ids)


def latency_signal(action: Action, latency_ms: Mapping[str, float]) -> float:
    top = max(latency_ms.values())
    return latency_ms[action.value] / top if top else 0.0

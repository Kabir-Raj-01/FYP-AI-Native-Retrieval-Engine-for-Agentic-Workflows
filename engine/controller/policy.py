"""Rule + utility controller (ACTION_POLICY §2-§5, §7; ADR-009 §3, §12).

Deterministic: hard rules (budget, preconditions) first, then utility
``U = alpha*gain - beta*cost - gamma*redundancy_signal - delta*latency_signal``, then
the fixed tie-break REUSE > SPARSE_HYBRID > DENSE > RERANK_EXPAND > TRANSFORM. Never calls an LLM.

Interpretations of the spec (recorded for review, ACTION_POLICY owner):
- "every retrieval action is blocked only by Issued(·)" (TRANSFORM precondition) = both
  DENSE and SPARSE_HYBRID have already issued the pending query.
- "failed queries" of the current gap (REWRITE vs SPLIT, PLANNER_SPEC §2) = the gap's
  planner queries (``query_history``) that were issued (``retrieval_history``) — state keeps
  no per-query gain, and an issued query of a still-OPEN gap did not resolve it.
- The fixed policy's "STOP(STAGNATION) for this gap" is a STOP decision with
  ``rule="fixed_policy"``; the monitor exhausts the gap and advances instead of stopping
  the task (``engine/monitor/trajectory.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from engine.config import EngineConfig
from engine.controller import signals
from engine.controller.interfaces import Controller
from engine.embeddings.interfaces import Embedder
from engine.errors import ConfigError
from engine.state.models import (
    Action,
    ActionDecision,
    ActionScores,
    CacheHit,
    GapStatus,
    NeedType,
    PlanMode,
    RetrievalState,
    StopReason,
)
from engine.text import normalize

__all__ = ["FIXED_POLICY_RULE", "MIN_UTILITY_RULE", "RuleUtilityController", "load_latency_ms"]

FIXED_POLICY_RULE = "fixed_policy"
MIN_UTILITY_RULE = "min_utility"
_TIE_ORDER = (
    Action.REUSE,
    Action.SPARSE_HYBRID,
    Action.DENSE,
    Action.RERANK_EXPAND,
    Action.TRANSFORM,
)
_SEARCH = (Action.DENSE, Action.SPARSE_HYBRID)
_NEEDS_RETRIEVAL_BUDGET = frozenset(
    {Action.DENSE, Action.SPARSE_HYBRID, Action.RERANK_EXPAND, Action.TRANSFORM}
)


def load_latency_ms(cfg: EngineConfig) -> dict[str, float]:
    """``latency_ms`` of ``controller.latency_profile_path`` if the file exists, else
    ``controller.default_latency_ms`` (ACTION_POLICY §4)."""
    defaults = {k: float(v) for k, v in cfg.controller.default_latency_ms.model_dump().items()}
    path = Path(cfg.controller.latency_profile_path)
    if not path.exists():
        return defaults
    profile = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    measured = profile.get("latency_ms", {})
    if set(measured) != set(defaults):
        raise ConfigError(f"{path}: latency_ms must have exactly the keys {sorted(defaults)}")
    return {k: float(v) for k, v in measured.items()}


class RuleUtilityController(Controller):
    def __init__(
        self,
        cfg: EngineConfig,
        embedder: Embedder,
        latency_ms: Mapping[str, float] | None = None,
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._latency_ms = dict(latency_ms) if latency_ms is not None else load_latency_ms(cfg)

    # ------------------------------------------------------------------ selection (§2)
    def select_action(self, state: RetrievalState, cache_hit: CacheHit | None) -> ActionDecision:
        b = state.budget
        if b.steps_used >= b.max_steps:  # defensive only (ADR-009 §12)
            return self._stop(StopReason.MAX_STEPS, "budget_guard", [])
        eligible, llm_only, blocked_only = self._eligible(state, cache_hit)
        if b.retrieval_calls_used >= b.max_retrieval_calls:
            eligible = [a for a in eligible if a not in _NEEDS_RETRIEVAL_BUDGET]
            if not eligible:
                return self._stop(StopReason.MAX_RETRIEVAL_CALLS, "budget_guard", [])
        if not eligible:
            reason = StopReason.MAX_LLM_CALLS if llm_only else StopReason.STAGNATION
            return self._stop(reason, "no_eligible_action", [])
        if not self._cfg.ablation.adaptive_actions:
            return self._fixed(state, eligible)
        scores = self._scores(state, cache_hit, eligible, blocked_only)
        best = min(eligible, key=lambda a: (-scores[a].U, _TIE_ORDER.index(a)))
        if self._cfg.controller.min_utility > scores[best].U and self._gap_searched(state):
            return ActionDecision(
                action=Action.STOP,
                eligible=eligible,
                scores=scores,
                stop_reason=StopReason.LOW_MARGINAL_UTILITY,
                rule=MIN_UTILITY_RULE,
            )
        return ActionDecision(
            action=best,
            eligible=eligible,
            scores=scores,
            plan_mode=self._plan_mode(state) if best == Action.TRANSFORM else None,
            rule="utility_argmax",
        )

    @staticmethod
    def _gap_searched(state: RetrievalState) -> bool:
        """Step 8's guard, per gap: the current gap already produced candidates (one of its
        queries is in ``retrieval_history``); without a gap, any retrieval call."""
        gap = state.current_gap
        if gap is None:
            return state.budget.retrieval_calls_used >= 1
        issued = {normalize(r.query) for r in state.retrieval_history}
        queries = {normalize(q.query) for q in state.query_history if q.gap_id == gap.gap_id}
        return bool((queries | {normalize(gap.query)}) & issued)

    @staticmethod
    def _stop(reason: StopReason, rule: str, eligible: list[Action]) -> ActionDecision:
        return ActionDecision(
            action=Action.STOP, eligible=eligible, scores={}, stop_reason=reason, rule=rule
        )

    # --------------------------------------------------------------- preconditions (§3)
    def _eligible(
        self, state: RetrievalState, hit: CacheHit | None
    ) -> tuple[list[Action], bool, bool]:
        """(eligible actions in tie order, TRANSFORM excluded only by the LLM budget,
        TRANSFORM eligible only because both searches already issued the query)."""
        cfg, b = self._cfg, state.budget
        pq = normalize(state.pending_query)
        issued = {(r.action, normalize(r.query)) for r in state.retrieval_history}

        def was(a: Action) -> bool:
            return (a, pq) in issued

        need = NeedType(state.current_need_type)
        preferred = {
            NeedType.SEMANTIC: {Action.DENSE},
            NeedType.LEXICAL: {Action.SPARSE_HYBRID},
        }.get(need, {Action.DENSE, Action.SPARSE_HYBRID})
        left = b.retrieval_calls_used < b.max_retrieval_calls
        stagnated = state.stagnation_count >= 1
        eligible: set[Action] = set()
        # REUSE at most once per pending query, like the searches' not-Issued rule: reusing
        # the same entry again for the same query returns the same candidates (decided
        # 2026-10-05 after the golden test showed REUSE looping to MAX_STEPS) — and never for
        # a query this task already searched: its hit is that search's own result, already
        # scored against E_t (2026-10-08, live runs: SEARCH > REUSE of it > STOP per gap)
        searched = any(was(a) for a in _SEARCH)
        if cfg.ablation.cache and hit is not None and not was(Action.REUSE) and not searched:
            eligible.add(Action.REUSE)
        for a in _SEARCH:
            switch = a in preferred or stagnated or any(was(p) for p in preferred)
            if left and pq and not was(a) and switch:
                eligible.add(a)
        last = state.last_result
        if (
            left
            and last is not None
            and normalize(last.query) == pq
            and last.depth < cfg.retrieval.max_k
            and signals.ambiguity(last, cfg) > 0.0
        ):
            eligible.add(Action.RERANK_EXPAND)
        blocked_only = bool(pq) and all(was(a) for a in _SEARCH)
        transform_ok = (
            cfg.ablation.planner
            and state.current_gap is not None
            and state.transforms_in_streak < cfg.monitor.max_transforms_per_streak
            and (stagnated or blocked_only)
        )
        llm_left = b.llm_calls_used < b.max_llm_calls
        if transform_ok and llm_left:
            eligible.add(Action.TRANSFORM)
        ordered = [a for a in _TIE_ORDER if a in eligible]
        return ordered, transform_ok and not llm_left, blocked_only and not stagnated

    # ---------------------------------------------------------------------- utility (§4)
    def _scores(
        self,
        state: RetrievalState,
        hit: CacheHit | None,
        eligible: list[Action],
        blocked_only: bool,
    ) -> dict[Action, ActionScores]:
        cfg = self._cfg
        w = cfg.controller.weights
        open_gaps = [g.description for g in state.gaps if g.status == GapStatus.OPEN]
        previous = {a: [r.query for r in state.retrieval_history if r.action == a] for a in _SEARCH}
        texts = [
            state.pending_query,
            *open_gaps,
            *previous[Action.DENSE],
            *previous[Action.SPARSE_HYBRID],
        ]
        vectors = self._embedder.embed_queries(texts).tolist()
        q = vectors[0]
        gaps_v = vectors[1 : 1 + len(open_gaps)]
        rest = vectors[1 + len(open_gaps) :]
        prev_v = {
            Action.DENSE: rest[: len(previous[Action.DENSE])],
            Action.SPARSE_HYBRID: rest[len(previous[Action.DENSE]) :],
        }
        gap_alignment = signals.max_cos(q, gaps_v)
        use_redundancy = cfg.ablation.redundancy_signal
        scores: dict[Action, ActionScores] = {}
        for a in eligible:
            if a in _SEARCH:
                gain = signals.gain_search(a, state, cfg, gap_alignment)
                redundancy = signals.max_cos(q, prev_v[a]) if use_redundancy else 0.0
            elif a == Action.REUSE and hit is not None:
                gain, redundancy = signals.gain_reuse(hit, state), 0.0
            elif a == Action.RERANK_EXPAND:
                gain = signals.gain_rerank_expand(state, cfg)
                redundancy = signals.redundancy_rerank_expand(state) if use_redundancy else 0.0
            else:  # TRANSFORM
                gain, redundancy = signals.gain_transform(state, cfg, blocked_only), 0.0
            cost = signals.cost(a, state, cfg)
            latency = signals.latency_signal(a, self._latency_ms)
            utility = w.alpha * gain - w.beta * cost - w.gamma * redundancy - w.delta * latency
            scores[a] = ActionScores(G=gain, C=cost, R=redundancy, L=latency, U=utility)
        return scores

    # ------------------------------------------------------------- TRANSFORM mode (§2.9)
    def _plan_mode(self, state: RetrievalState) -> PlanMode:
        gap = state.current_gap
        if gap is None:
            return PlanMode.REWRITE
        issued = {normalize(r.query) for r in state.retrieval_history}
        proposed = {normalize(q.query) for q in state.query_history if q.gap_id == gap.gap_id}
        failed = len(proposed & issued)
        live = sum(1 for g in state.gaps if g.status in (GapStatus.OPEN, GapStatus.RESOLVED))
        room = live + 1 <= self._cfg.planner.max_gaps  # SPLIT: -1 parent, +≥2 children
        return PlanMode.SPLIT if failed >= 2 and room else PlanMode.REWRITE

    # ------------------------------------------------------------------ fixed policy (§7)
    def _fixed(self, state: RetrievalState, eligible: list[Action]) -> ActionDecision:
        b = state.budget
        if b.retrieval_calls_used >= b.max_retrieval_calls:  # no REUSE in the fixed policy
            return self._stop(StopReason.MAX_RETRIEVAL_CALLS, "budget_guard", [])
        pq = normalize(state.pending_query)
        issued = {(r.action, normalize(r.query)) for r in state.retrieval_history}
        semantic = state.current_need_type == NeedType.SEMANTIC
        first, second = (
            (Action.DENSE, Action.SPARSE_HYBRID)
            if semantic
            else (Action.SPARSE_HYBRID, Action.DENSE)
        )
        for a in (first, second):
            if (a, pq) not in issued and a in eligible:
                return ActionDecision(
                    action=a, eligible=eligible, scores={}, rule=FIXED_POLICY_RULE
                )
        return ActionDecision(
            action=Action.STOP,
            eligible=eligible,
            scores={},
            stop_reason=StopReason.STAGNATION,
            rule=FIXED_POLICY_RULE,
        )

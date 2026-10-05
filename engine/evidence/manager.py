"""EvidenceManager (EVIDENCE_MANAGER_SPEC §2-§5; ADR-009 §4).

Per step: acceptance checks in ``final_score`` order (DUPLICATE, LOW_RELEVANCE, LOW_NOVELTY,
STEP_CAP, CAP_REACHED), eviction of the lowest-``final_score`` unpinned item when E_t is
full, gap resolution by embedding coverage, and the step's evidence gain. Pure with respect
to the state: the ``score_and_update`` node applies the returned EvidenceUpdate.

Not built: the optional LLM verifier (``llm.verifier.enabled``). Its ``{supports: bool}``
output needs a shared model (STATE_SPEC change → ADR) and ADR-004 has no verifier model yet,
so enabling it raises ConfigError.
"""

from __future__ import annotations

import numpy as np

from engine.config import EngineConfig
from engine.embeddings.interfaces import Embedder
from engine.errors import ConfigError
from engine.evidence.interfaces import EvidenceManager
from engine.monitor.logging import get_logger
from engine.state.models import (
    Action,
    EvidenceGap,
    EvidenceItem,
    EvidenceUpdate,
    GapStatus,
    ItemChange,
    RetrievalState,
    ScoredCandidate,
)
from engine.text import summary_of

__all__ = ["ThresholdEvidenceManager"]


class ThresholdEvidenceManager(EvidenceManager):
    def __init__(self, cfg: EngineConfig, embedder: Embedder) -> None:
        if cfg.llm.verifier.enabled:
            raise ConfigError(
                "llm.verifier.enabled: the evidence verifier is not built yet (needs a "
                "verifier output model via ADR and a verifier model in ADR-004)"
            )
        self._cfg = cfg
        self._embedder = embedder
        self._log = get_logger("engine.evidence")

    def update(
        self, state: RetrievalState, scored: list[ScoredCandidate], step: int
    ) -> EvidenceUpdate:
        ev = self._cfg.evidence
        decision = state.last_decision
        source = decision.action if decision is not None else Action.DENSE
        query = state.retrieval_history[-1].query if state.retrieval_history else ""
        existing = list(state.evidence)
        evicted: list[str] = []
        accepted: list[EvidenceItem] = []
        rejected: dict[str, str] = {}
        for s in sorted(scored, key=lambda x: -x.final_score):  # stable
            cid = s.candidate.chunk_id
            reason = self._reject(s, existing, accepted)
            if reason is None:
                current = [*existing, *accepted]
                if len(current) >= ev.max_items:
                    victims = [i for i in current if not i.pinned and i.final_score < s.final_score]
                    if not victims:
                        reason = "CAP_REACHED"
                    else:
                        victim = min(victims, key=lambda i: i.final_score)
                        if victim in accepted:
                            accepted.remove(victim)
                        else:
                            existing.remove(victim)
                            evicted.append(victim.evidence_id)
                        self._log.info("evidence.evicted", chunk_id=victim.evidence_id)
            if reason is not None:
                rejected[cid] = reason
                self._log.info("evidence.rejected", chunk_id=cid, reason=reason)
                continue
            accepted.append(self._item(s, source, query, step))
            self._log.info(
                "evidence.accepted",
                chunk_id=cid,
                scores={"relevance": s.relevance, "novelty": s.novelty, "final": s.final_score},
            )

        open_at_start = [g for g in state.gaps if g.status == GapStatus.OPEN]
        quality = float(np.mean([e.relevance * e.novelty for e in accepted])) if accepted else 0.0
        if self._cfg.ablation.gap_tracking:
            resolutions = self._resolve_by_coverage([*existing, *accepted], open_at_start)
            w = ev.gain_weights
            gain = w.gap_resolution * len(resolutions) / max(1, len(open_at_start))
            gain += w.quality * quality
        else:
            gain = quality
            resolutions = self._resolve_by_sufficiency([*existing, *accepted], open_at_start, gain)

        accepted, changes = self._pin(accepted, existing, resolutions)
        for gap_id, evidence_id in resolutions.items():
            self._log.info("gap.resolved", gap_id=gap_id, by=evidence_id)
        return EvidenceUpdate(
            accepted=accepted,
            rejected=rejected,
            evicted=evicted,
            resolved_gap_ids=list(resolutions),
            gap_resolutions=resolutions,
            item_changes=changes,
            evidence_gain=min(1.0, max(0.0, gain)),
        )

    def _reject(
        self, s: ScoredCandidate, existing: list[EvidenceItem], accepted: list[EvidenceItem]
    ) -> str | None:
        ev = self._cfg.evidence
        cid = s.candidate.chunk_id
        if any(i.chunk_id == cid for i in (*existing, *accepted)):
            return "DUPLICATE"
        if s.relevance < ev.relevance_threshold:
            return "LOW_RELEVANCE"
        if s.novelty < ev.novelty_threshold:
            return "LOW_NOVELTY"
        if len(accepted) >= ev.max_accept_per_step:
            return "STEP_CAP"
        return None

    def _item(self, s: ScoredCandidate, source: Action, query: str, step: int) -> EvidenceItem:
        c = s.candidate
        return EvidenceItem(
            evidence_id=c.chunk_id,
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            text=c.text,
            summary=summary_of(c.text, self._cfg.planner.evidence_summary_chars),
            relevance=s.relevance,
            novelty=s.novelty,
            final_score=s.final_score,
            source_action=source,
            query=query,
            metadata=dict(c.metadata),
            added_at_step=step,
            vector=c.vector,
        )

    def _resolve_by_coverage(
        self, items: list[EvidenceItem], gaps: list[EvidenceGap]
    ) -> dict[str, str]:
        ev = self._cfg.evidence
        usable = [
            i for i in items if i.relevance >= ev.relevance_threshold and i.vector is not None
        ]
        if not usable or not gaps:
            return {}
        g = np.asarray(
            self._embedder.embed_queries([x.description for x in gaps]), dtype=np.float64
        )
        g = g / np.where((n := np.linalg.norm(g, axis=1, keepdims=True)) == 0, 1.0, n)
        v = np.asarray([i.vector for i in usable], dtype=np.float64)
        v = v / np.where((n := np.linalg.norm(v, axis=1, keepdims=True)) == 0, 1.0, n)
        cov = v @ g.T  # (items, gaps)
        out: dict[str, str] = {}
        for j, gap in enumerate(gaps):
            best = int(np.argmax(cov[:, j]))  # ties: the earliest item
            if cov[best, j] >= ev.gap_coverage_threshold:
                out[gap.gap_id] = usable[best].evidence_id
        return out

    def _resolve_by_sufficiency(
        self, items: list[EvidenceItem], gaps: list[EvidenceGap], gain: float
    ) -> dict[str, str]:
        """``ablation.gap_tracking`` false: the single gap (the goal) is RESOLVED once E_t
        has ≥ ``min_items_for_sufficiency`` relevant items and this step's gain is below
        ``monitor.stagnation_threshold`` (§5)."""
        ev = self._cfg.evidence
        relevant = [i for i in items if i.relevance >= ev.relevance_threshold]
        enough = len(relevant) >= ev.min_items_for_sufficiency
        if not gaps or not enough or gain >= self._cfg.monitor.stagnation_threshold:
            return {}
        best = max(relevant, key=lambda i: i.final_score)
        return {gaps[0].gap_id: best.evidence_id}

    @staticmethod
    def _pin(
        accepted: list[EvidenceItem], existing: list[EvidenceItem], resolutions: dict[str, str]
    ) -> tuple[list[EvidenceItem], dict[str, ItemChange]]:
        """Resolving items become pinned and list the gaps they support."""
        by_item: dict[str, list[str]] = {}
        for gap_id, evidence_id in resolutions.items():
            by_item.setdefault(evidence_id, []).append(gap_id)
        new = [
            i.model_copy(
                update={
                    "pinned": True,
                    "supports_gap_ids": [*i.supports_gap_ids, *by_item[i.evidence_id]],
                }
            )
            if i.evidence_id in by_item
            else i
            for i in accepted
        ]
        changes = {
            i.evidence_id: ItemChange(
                pinned=True, supports_gap_ids=[*i.supports_gap_ids, *by_item[i.evidence_id]]
            )
            for i in existing
            if i.evidence_id in by_item
        }
        return new, changes

"""engine/evidence/manager.py (EVIDENCE_MANAGER_SPEC §2-§5, §8)."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine.config import EngineConfig
from engine.errors import ConfigError
from engine.evidence.manager import ThresholdEvidenceManager
from engine.state.models import (
    Action,
    ActionDecision,
    RetrievalMode,
    RetrievalRecord,
    RetrievalState,
    ScoredCandidate,
)
from tests.builders import make_candidate, make_evidence, make_gap, make_state
from tests.fakes import FakeEmbedder

GAP_VEC = [1.0, 0.0, 0.0]
COVERS = [1.0, 0.0, 0.0]  # coverage 1.0 of the gap
MISSES = [0.0, 1.0, 0.0]  # coverage 0.0


def scored(
    cid: str,
    relevance: float = 0.9,
    novelty: float = 0.9,
    final: float | None = None,
    vector: list[float] | None = None,
) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=make_candidate(cid, text=f"Text of {cid}. More text.", vector=vector or MISSES),
        relevance=relevance,
        gap_coverage=0.0,
        novelty=novelty,
        redundancy=0.0,
        final_score=relevance if final is None else final,
    )


def state(**kw: Any) -> RetrievalState:
    fields: dict[str, Any] = {
        "gaps": [make_gap("g0", description="the gap")],
        "current_gap_id": "g0",
        "needs_initial_plan": False,
        "last_decision": ActionDecision(
            action=Action.DENSE, eligible=[Action.DENSE], scores={}, rule="r"
        ),
        "retrieval_history": [
            RetrievalRecord(
                step=1, action=Action.DENSE, mode=RetrievalMode.DENSE, query="the query", k=5,
                candidate_ids=[], new_candidate_ids=[], cache_hit=False, latency_ms=1.0,
            )
        ],
    }  # fmt: skip
    fields.update(kw)
    return make_state(**fields)


def manager(cfg: EngineConfig) -> ThresholdEvidenceManager:
    return ThresholdEvidenceManager(cfg, FakeEmbedder(dimension=3, vectors={"the gap": GAP_VEC}))


def test_acceptance_checks_and_reason_codes(cfg: EngineConfig) -> None:
    ev = cfg.evidence
    s = state(evidence=[make_evidence("old", vector=MISSES)])
    u = manager(cfg).update(
        s,
        [
            scored("old"),
            scored("low-rel", relevance=ev.relevance_threshold - 0.01),
            scored("low-nov", novelty=ev.novelty_threshold - 0.01),
            scored("ok"),
            scored("ok"),  # same chunk twice in one list
        ],
        2,
    )
    assert u.rejected == {
        "old": "DUPLICATE",
        "low-rel": "LOW_RELEVANCE",
        "low-nov": "LOW_NOVELTY",
    } | {"ok": "DUPLICATE"}
    (item,) = u.accepted
    assert (item.evidence_id, item.added_at_step, item.query, item.source_action) == (
        "ok",
        2,
        "the query",
        Action.DENSE,
    )
    assert len(item.summary) <= cfg.planner.evidence_summary_chars and item.vector == MISSES


def test_step_cap(cfg: EngineConfig) -> None:
    cfg.evidence.max_accept_per_step = 2
    u = manager(cfg).update(state(), [scored(f"c{i}", final=1 - i / 10) for i in range(4)], 1)
    assert [i.evidence_id for i in u.accepted] == ["c0", "c1"]
    assert u.rejected == {"c2": "STEP_CAP", "c3": "STEP_CAP"}


def test_full_evidence_evicts_the_lowest_unpinned_item(cfg: EngineConfig) -> None:
    cfg.evidence.max_items = 2
    s = state(
        evidence=[
            make_evidence("pinned", final_score=0.1, pinned=True),
            make_evidence("weak", final_score=0.2),
        ]
    )
    u = manager(cfg).update(s, [scored("new", final=0.5)], 3)
    assert u.evicted == ["weak"] and [i.evidence_id for i in u.accepted] == ["new"]


def test_full_evidence_without_an_evictable_item_rejects(cfg: EngineConfig) -> None:
    cfg.evidence.max_items = 2
    s = state(
        evidence=[
            make_evidence("p1", final_score=0.1, pinned=True),
            make_evidence("strong", final_score=0.95),
        ]
    )
    u = manager(cfg).update(s, [scored("new", final=0.5)], 3)
    assert u.rejected == {"new": "CAP_REACHED"} and u.evicted == [] and u.accepted == []


def test_gap_resolution_threshold_is_inclusive(cfg: EngineConfig) -> None:
    t = cfg.evidence.gap_coverage_threshold
    at = [t, (1 - t * t) ** 0.5, 0.0]  # cos with the gap == threshold exactly
    u = manager(cfg).update(state(), [scored("at", vector=at)], 1)
    assert u.gap_resolutions == {"g0": "at"} and u.resolved_gap_ids == ["g0"]
    assert u.accepted[0].pinned and u.accepted[0].supports_gap_ids == ["g0"]
    below = [t - 0.01, (1 - (t - 0.01) ** 2) ** 0.5, 0.0]
    assert manager(cfg).update(state(), [scored("below", vector=below)], 1).gap_resolutions == {}


def test_an_earlier_item_can_resolve_a_gap(cfg: EngineConfig) -> None:
    s = state(evidence=[make_evidence("old", vector=COVERS, relevance=0.9)])
    u = manager(cfg).update(s, [], 2)
    assert u.gap_resolutions == {"g0": "old"}
    assert u.item_changes["old"].pinned is True and u.item_changes["old"].supports_gap_ids == ["g0"]


def test_low_relevance_items_never_resolve(cfg: EngineConfig) -> None:
    s = state(
        evidence=[
            make_evidence("old", vector=COVERS, relevance=cfg.evidence.relevance_threshold - 0.1)
        ]
    )
    assert manager(cfg).update(s, [], 2).gap_resolutions == {}


def test_gain_formula_by_hand(cfg: EngineConfig) -> None:
    w = cfg.evidence.gain_weights
    s = state(gaps=[make_gap("g0", description="the gap"), make_gap("g1", description="other")])
    u = manager(cfg).update(
        s,
        [
            scored("a", relevance=0.8, novelty=0.5, vector=COVERS),
            scored("b", relevance=0.6, novelty=1.0),
        ],
        1,
    )
    resolved_part = 1 / 2  # g0 of 2 open gaps
    quality_part = (0.8 * 0.5 + 0.6 * 1.0) / 2
    assert u.evidence_gain == pytest.approx(
        w.gap_resolution * resolved_part + w.quality * quality_part
    )
    assert manager(cfg).update(state(), [], 1).evidence_gain == 0.0


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.floats(0, 1), st.floats(0, 1)), max_size=8))
def test_gain_is_always_in_unit_interval(pairs: list[tuple[float, float]]) -> None:
    from engine.config import load_config
    from tests.conftest import TEST_CONFIG_OVERRIDES

    cfg = load_config(environ={}, overrides=TEST_CONFIG_OVERRIDES)
    cands = [
        scored(f"c{i}", relevance=r, novelty=n, vector=COVERS if i % 2 else MISSES)
        for i, (r, n) in enumerate(pairs)
    ]
    u = manager(cfg).update(state(), cands, 1)
    assert 0.0 <= u.evidence_gain <= 1.0


def test_sufficiency_rule_without_gap_tracking(cfg: EngineConfig) -> None:
    cfg.ablation.gap_tracking = False
    cfg.evidence.min_items_for_sufficiency = 2
    s = state(
        evidence=[
            make_evidence("e1", relevance=0.9, final_score=0.9),
            make_evidence("e2", relevance=0.9, final_score=0.7),
        ]
    )
    u = manager(cfg).update(s, [], 3)  # no new information: gain 0 < stagnation_threshold
    assert u.gap_resolutions == {"g0": "e1"} and u.item_changes["e1"].pinned
    assert manager(cfg).update(state(evidence=s.evidence[:1]), [], 3).gap_resolutions == {}


def test_reuse_items_record_their_source(cfg: EngineConfig) -> None:
    s = state(
        last_decision=ActionDecision(
            action=Action.REUSE, eligible=[Action.REUSE], scores={}, rule="r"
        )
    )
    assert manager(cfg).update(s, [scored("a")], 2).accepted[0].source_action == Action.REUSE


def test_verifier_is_not_built(cfg: EngineConfig) -> None:
    cfg.llm.verifier.enabled = True
    with pytest.raises(ConfigError, match="verifier is not built"):
        manager(cfg)

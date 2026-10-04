"""Budget.consume / Budget.pressure (STATE_SPEC §3; ADR-009 §8)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.config import load_config
from engine.state.models import Budget
from tests.builders import make_budget


def test_consume_returns_a_new_budget_and_never_mutates() -> None:
    before = make_budget()
    after = before.consume(steps=1, retrieval_calls=1, llm_calls=2, tokens=300, reranker_calls=1)
    assert after is not before
    assert before == make_budget()
    assert (
        after.steps_used,
        after.retrieval_calls_used,
        after.llm_calls_used,
        after.tokens_used,
        after.reranker_calls_used,
    ) == (1, 1, 2, 300, 1)
    assert after.max_steps == before.max_steps


def test_consume_accumulates_and_defaults_to_zero_deltas() -> None:
    budget = make_budget().consume(tokens=100).consume(tokens=50).consume()
    assert budget.tokens_used == 150
    assert budget.steps_used == 0


def test_pressure_is_max_ratio_worked_example() -> None:
    # ACTION_POLICY §8: steps 2/8, retrieval 2/4, LLM 1/8 -> pressure 0.5
    budget = Budget(
        max_steps=8,
        max_retrieval_calls=4,
        max_llm_calls=8,
        max_tokens=20000,
        steps_used=2,
        retrieval_calls_used=2,
        llm_calls_used=1,
        tokens_used=100,
    )
    assert budget.pressure() == pytest.approx(0.5)


def test_pressure_is_zero_for_a_fresh_budget() -> None:
    assert make_budget().pressure() == 0.0


def test_pressure_skips_a_zero_cap() -> None:
    budget = make_budget(max_llm_calls=0, steps_used=2)
    assert budget.pressure() == pytest.approx(2 / 8)


def test_pressure_is_capped_at_one_when_tokens_overshoot() -> None:
    budget = make_budget(tokens_used=25000)
    assert budget.pressure() == 1.0


def test_reranker_calls_do_not_affect_pressure() -> None:
    assert make_budget(reranker_calls_used=1000).pressure() == 0.0


@pytest.mark.parametrize(
    ("field", "lowest_valid"),
    [("max_steps", 1), ("max_retrieval_calls", 1), ("max_llm_calls", 0), ("max_tokens", 1)],
)
def test_budget_cap_bounds(field: str, lowest_valid: int) -> None:
    make_budget(**{field: lowest_valid})
    with pytest.raises(ValidationError):
        make_budget(**{field: lowest_valid - 1})


def test_zero_llm_budget_from_config_loads_and_pressure_does_not_raise() -> None:
    # TESTING §4.10 / ADR-009 §8
    cfg = load_config(environ={}, overrides=["budget_defaults.max_llm_calls=0"])
    defaults = cfg.budget_defaults
    budget = Budget(
        max_steps=defaults.max_steps,
        max_retrieval_calls=defaults.max_retrieval_calls,
        max_llm_calls=defaults.max_llm_calls,
        max_tokens=defaults.max_tokens,
    )
    assert budget.max_llm_calls == 0
    assert budget.pressure() == 0.0

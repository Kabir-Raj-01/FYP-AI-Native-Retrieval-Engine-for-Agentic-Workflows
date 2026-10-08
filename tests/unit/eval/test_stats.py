"""eval/stats.py on cases with known answers (EVAL_PROTOCOL §5)."""

from __future__ import annotations

import pytest

from eval.stats import mcnemar_exact, median_ci, paired_bootstrap


def test_paired_bootstrap_constant_difference_has_a_point_ci() -> None:
    d, lo, hi = paired_bootstrap([0.0, 0.5, 1.0], [0.2, 0.7, 1.2], 500, seed=1)
    assert d == pytest.approx(0.2)
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(0.2)


def test_paired_bootstrap_ci_contains_the_mean_and_is_seeded() -> None:
    a = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    b = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    first = paired_bootstrap(a, b, 2000, seed=42)
    assert first == paired_bootstrap(a, b, 2000, seed=42)
    d, lo, hi = first
    assert d == pytest.approx(0.375)
    assert lo <= d <= hi


@pytest.mark.parametrize(("a", "b"), [([], []), ([1.0], [1.0, 2.0])])
def test_paired_bootstrap_rejects_bad_samples(a: list[float], b: list[float]) -> None:
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap(a, b, 10, seed=0)


def test_mcnemar_exact_counts_discordant_pairs() -> None:
    a = [True, True, False, False, True]
    b = [True, False, True, True, True]
    a_only, b_only, p = mcnemar_exact(a, b)
    assert (a_only, b_only) == (1, 2)
    assert p == pytest.approx(1.0)  # binomial(1 of 3, 0.5), two-sided


def test_mcnemar_exact_significant_and_degenerate() -> None:
    assert mcnemar_exact([False] * 12, [True] * 12)[2] == pytest.approx(2 * 0.5**12)
    assert mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)


def test_median_ci() -> None:
    m, lo, hi = median_ci([5.0, 1.0, 3.0], 500, seed=0)
    assert m == 3.0
    assert 1.0 <= lo <= m <= hi <= 5.0
    with pytest.raises(ValueError):
        median_ci([], 10, seed=0)

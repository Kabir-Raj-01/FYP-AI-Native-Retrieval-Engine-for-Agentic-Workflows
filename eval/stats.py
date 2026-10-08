"""Statistics for system comparisons (EVAL_PROTOCOL §5).

- ``paired_bootstrap``: mean of per-task differences (b - a) with a percentile 95 % CI.
- ``mcnemar_exact``: two-sided exact McNemar test for paired binary outcomes.
- ``median_ci``: median with a bootstrap 95 % CI (latency).
Resamples = ``eval.bootstrap_resamples``, seed = ``project.seed`` (passed in by the caller).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import binomtest

__all__ = ["mcnemar_exact", "median_ci", "paired_bootstrap"]


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], resamples: int, seed: int
) -> tuple[float, float, float]:
    """``(mean(b - a), ci_low, ci_high)``."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and of equal length")
    diff = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """``(a_only, b_only, p_value)``: discordant counts and the exact two-sided p-value."""
    if len(a) != len(b):
        raise ValueError("paired samples must be of equal length")
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return a_only, b_only, p


def median_ci(values: Sequence[float], resamples: int, seed: int) -> tuple[float, float, float]:
    """``(median, ci_low, ci_high)``."""
    if not values:
        raise ValueError("no values")
    x = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(resamples, len(x)))
    low, high = np.percentile(np.median(x[idx], axis=1), [2.5, 97.5])
    return float(np.median(x)), float(low), float(high)

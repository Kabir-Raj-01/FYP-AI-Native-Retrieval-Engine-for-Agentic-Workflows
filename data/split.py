"""Stratified dev/test split and dev subset (DATA.md §5; ADR-003).

Within each question type the task ids are sorted, shuffled with a generator seeded by
``(data.split_seed, question type)`` and the first ``round(n * data.dev_fraction)`` go to
dev. The dev subset (``data.dev_subset_size``) is drawn from dev with exact proportional
allocation per type (largest remainder; ties by type name), seeded the same way. Seeding
per type makes each type's assignment independent of the others.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["Split", "stratified_split"]


@dataclass(frozen=True)
class Split:
    dev: list[str]
    test: list[str]
    dev_subset: list[str]


def _by_type(types: Mapping[str, str], ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id in sorted(ids):
        groups[types[task_id]].append(task_id)
    return dict(sorted(groups.items()))


def _shuffled(ids: list[str], seed: int, *salt: str) -> list[str]:
    out = list(ids)
    random.Random(":".join([str(seed), *salt])).shuffle(out)
    return out


def _half_up(x: float) -> int:
    return math.floor(x + 0.5)


def stratified_split(
    types: Mapping[str, str], dev_fraction: float, subset_size: int, seed: int
) -> Split:
    """Split ``types`` (task_id -> question type) into dev/test plus a dev subset.

    Raises:
        ValueError: ``subset_size`` is larger than the dev split.
    """
    dev: list[str] = []
    test: list[str] = []
    for qtype, ids in _by_type(types, list(types)).items():
        order = _shuffled(ids, seed, qtype)
        cut = _half_up(len(order) * dev_fraction)
        dev.extend(order[:cut])
        test.extend(order[cut:])

    if subset_size > len(dev):
        raise ValueError(f"dev subset size {subset_size} exceeds the dev split ({len(dev)})")
    if subset_size == 0:
        return Split(sorted(dev), sorted(test), [])
    dev_groups = _by_type(types, dev)
    quotas = {t: subset_size * len(ids) / len(dev) for t, ids in dev_groups.items()}
    alloc = {t: math.floor(q) for t, q in quotas.items()}
    remaining = subset_size - sum(alloc.values())
    for t in sorted(quotas, key=lambda t: (-(quotas[t] - alloc[t]), t))[:remaining]:
        alloc[t] += 1
    subset: list[str] = []
    for t, ids in dev_groups.items():
        subset.extend(_shuffled(ids, seed, t, "subset")[: alloc[t]])
    return Split(sorted(dev), sorted(test), sorted(subset))

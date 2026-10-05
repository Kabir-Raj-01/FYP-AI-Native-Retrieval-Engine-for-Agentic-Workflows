"""data/split.py: stratified, deterministic split and dev subset (DATA.md §5)."""

from __future__ import annotations

from collections import Counter

import pytest

from data.split import stratified_split

TYPES = {f"t:{i:04d}": qt for i, qt in enumerate(["a"] * 50 + ["b"] * 30 + ["c"] * 21)}


def test_dev_and_test_partition_all_tasks() -> None:
    split = stratified_split(TYPES, 0.7, 10, seed=42)
    assert set(split.dev) | set(split.test) == set(TYPES)
    assert not set(split.dev) & set(split.test)
    assert split.dev == sorted(split.dev)


def test_each_type_is_split_by_the_dev_fraction() -> None:
    split = stratified_split(TYPES, 0.7, 10, seed=42)
    per_type = Counter(TYPES[t] for t in split.dev)
    assert per_type == {"a": 35, "b": 21, "c": 15}  # round-half-up of 0.7 * n (14.7 -> 15)


def test_split_is_deterministic_and_seed_dependent() -> None:
    assert stratified_split(TYPES, 0.7, 10, seed=42) == stratified_split(TYPES, 0.7, 10, seed=42)
    assert stratified_split(TYPES, 0.7, 10, seed=42) != stratified_split(TYPES, 0.7, 10, seed=7)


def test_types_are_assigned_independently() -> None:
    fewer = {k: v for k, v in TYPES.items() if v != "c"}
    full = stratified_split(TYPES, 0.7, 10, seed=42)
    reduced = stratified_split(fewer, 0.7, 10, seed=42)
    a_dev = {t for t in full.dev if TYPES[t] == "a"}
    assert a_dev == {t for t in reduced.dev if TYPES[t] == "a"}


def test_dev_subset_has_exact_size_and_proportional_mix() -> None:
    split = stratified_split(TYPES, 0.7, 20, seed=42)
    assert len(split.dev_subset) == 20
    assert set(split.dev_subset) <= set(split.dev)
    # dev has a 35/21/15 mix of 71 -> quotas 9.86 / 5.92 / 4.23 -> 10 / 6 / 4
    assert Counter(TYPES[t] for t in split.dev_subset) == {"a": 10, "b": 6, "c": 4}


def test_zero_subset_and_oversized_subset() -> None:
    assert stratified_split(TYPES, 0.7, 0, seed=1).dev_subset == []
    with pytest.raises(ValueError, match="exceeds the dev split"):
        stratified_split(TYPES, 0.7, 1000, seed=1)

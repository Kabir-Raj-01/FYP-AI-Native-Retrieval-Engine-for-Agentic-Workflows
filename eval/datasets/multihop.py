"""MultiHop-RAG tasks: raw queries + committed split.json + gold_map.json (DATA.md §2-§5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.download import raw_dir
from data.ingest import GOLD_MAP_FILE, SPLIT_FILE, processed_dir
from data.multihop import DATASET, load_queries
from engine.config import EngineConfig
from eval.datasets.base import Task

__all__ = ["load_tasks"]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_tasks(cfg: EngineConfig, split: str, subset: bool) -> list[Task]:
    """Tasks of ``split`` ("dev" | "test"); ``subset`` selects the dev subset
    (``data.dev_subset_size``, DATA §5), dev only. Sorted by task_id.

    Raises:
        ValueError: unknown split, or ``subset`` with the test split.
    """
    if split not in ("dev", "test"):
        raise ValueError(f"unknown split {split!r}")
    if subset and split != "dev":
        raise ValueError("experiment.subset applies to the dev split only")
    folder = processed_dir(cfg, DATASET)
    split_doc = _read_json(folder / SPLIT_FILE)
    gold = _read_json(folder / GOLD_MAP_FILE)
    wanted = set(split_doc["dev_subset"] if subset else split_doc[split])
    tasks = [
        Task(
            task_id=q.task_id,
            dataset=DATASET,
            question=q.question,
            answer=q.answer,
            question_type=q.question_type,
            gold_groups=gold[q.task_id],
            split=split,
        )
        for q in load_queries(raw_dir(cfg, DATASET))
        if q.task_id in wanted
    ]
    return sorted(tasks, key=lambda t: t.task_id)

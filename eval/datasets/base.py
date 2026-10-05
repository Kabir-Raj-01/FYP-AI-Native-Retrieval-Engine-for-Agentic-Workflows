"""Common task interface for every benchmark (DATA.md §1): ``eval/run.py`` has no dataset
branches (FR12)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["Task"]


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str  # "<dataset>:<original_id>"
    dataset: str  # "multihop" | "hotpotqa" | "curated"
    question: str  # becomes RetrievalState.goal
    answer: str | None  # gold answer (None for tasks without one)
    question_type: str | None  # e.g. inference / comparison / temporal / null
    gold_groups: list[list[str]]  # one inner list per evidence entry (§4); any member satisfies it
    split: str  # "dev" | "test"
    meta: dict[str, str | int | None] = {}

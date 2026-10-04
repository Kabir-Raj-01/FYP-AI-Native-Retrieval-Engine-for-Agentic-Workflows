"""Persistence (de)serialization helpers (STATE_SPEC §2/§6; ADR-009 "Also" and §4a).

Default dumps (API responses, traces, bundles) leave vector fields out (``None``).
TaskStore and cache writes use :func:`dump_with_vectors`, which keeps vectors and drops
every ``@computed_field`` key: with ``extra="forbid"`` a payload containing them would be
rejected by ``model_validate``, and they recompute themselves on load anyway.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from engine.state.models import WITH_VECTORS

__all__ = ["dump_with_vectors", "load_with_vectors"]

ModelT = TypeVar("ModelT", bound=BaseModel)


def _computed_field_exclusions(value: object) -> dict[Any, Any] | None:
    """Pydantic ``exclude`` spec naming every computed field found inside ``value``."""
    if isinstance(value, BaseModel):
        spec: dict[Any, Any] = dict.fromkeys(type(value).model_computed_fields, True)
        for name in type(value).model_fields:
            nested = _computed_field_exclusions(getattr(value, name))
            if nested:
                spec[name] = nested
        return spec or None
    if isinstance(value, list | tuple):
        items = {i: _computed_field_exclusions(item) for i, item in enumerate(value)}
        return {i: s for i, s in items.items() if s} or None
    if isinstance(value, Mapping):
        entries = {k: _computed_field_exclusions(item) for k, item in value.items()}
        return {k: s for k, s in entries.items() if s} or None
    return None


def dump_with_vectors(model: BaseModel) -> dict[str, Any]:
    """JSON-compatible dict of ``model`` including vectors and excluding computed fields."""
    return model.model_dump(
        mode="json",
        context={WITH_VECTORS: True},
        exclude=_computed_field_exclusions(model),
    )


def load_with_vectors(cls: type[ModelT], data: Mapping[str, Any]) -> ModelT:
    """Validate a :func:`dump_with_vectors` payload back into ``cls``."""
    return cls.model_validate(data)

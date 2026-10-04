"""configs/base.yaml and docs/CONFIG_SCHEMA.md must never drift apart.

Parses the key tables of CONFIG_SCHEMA §2 and §3, expands the shorthand used there
(``{a,b}``, ``<role>``, ``<Action>``, ``x.y / z``) and compares:
- the set of keys with the set of base.yaml leaf keys (exactly);
- defaults, where the table states them as literals;
- the §3 experiment keys with ``ExperimentConfig``.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterator
from typing import Any

import pytest
import yaml

from engine.config import DEFAULT_CONFIG_PATH, REPO_ROOT, ExperimentConfig

SCHEMA_PATH = REPO_ROOT / "docs" / "CONFIG_SCHEMA.md"
ROLES = ["planner", "verifier", "generator"]
# `<Action>` in CONFIG_SCHEMA = the actions listed in the controller.cost_units row.
ACTIONS = ["REUSE", "DENSE", "SPARSE_HYBRID", "RERANK_EXPAND", "TRANSFORM"]

# Defaults the table describes in words rather than as a literal (key -> reason).
DESCRIBED_DEFAULTS = {
    "embedding.query_prefix": "described as 'BGE retrieval instruction'",
    "qdrant.id_namespace": "described as 'fixed UUID'",
    "eval.repeats": "described as '1 (3 for final test)'",
}
DESCRIBED_PREFIXES = ("controller.type_match.",)  # 'see base.yaml'

_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def _section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def _rows(section: str) -> Iterator[list[str]]:
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("\\|", "|").strip("`") for c in _CELL_SPLIT.split(line)]
        cells = cells[1:-1]
        if not cells or cells[0] in ("Key",) or set(cells[0]) <= set("-: "):
            continue
        yield cells


def _expand(key: str) -> list[str]:
    key = key.replace("<role>", "{" + ",".join(ROLES) + "}")
    key = key.replace("<Action>", "{" + ",".join(ACTIONS) + "}")
    if " / " in key:  # "qdrant.hnsw.m / ef_construct" -> two sibling keys
        first, *rest = (part.strip() for part in key.split(" / "))
        parent = first.rsplit(".", 1)[0]
        return [k for part in [first, *(f"{parent}.{r}" for r in rest)] for k in _expand(part)]
    parts = re.split(r"(\{[^}]*\})", key)
    options = [p[1:-1].split(",") if p.startswith("{") else [p] for p in parts]
    return ["".join(combo) for combo in itertools.product(*options)]


def _schema_rows() -> list[tuple[list[str], str]]:
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    section = _section(text, "## 2. Keys", "## 3.")
    out = []
    for cells in _rows(section):
        if "." not in cells[0]:
            continue
        default = cells[2] if len(cells) > 2 else ""
        out.append((_expand(cells[0]), default))
    return out


def _leaves(node: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict) and node:
        for key, value in node.items():
            yield from _leaves(value, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, node


def _base_leaves() -> dict[str, Any]:
    return dict(_leaves(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))))


def _parse_defaults(keys: list[str], cell: str) -> dict[str, Any] | None:
    """Map each key to its documented default, or None if the cell is not literal."""
    cell = cell.strip()
    if cell.endswith(" each"):
        value = yaml.safe_load(cell.removesuffix(" each"))
        return dict.fromkeys(keys, value)
    named = re.findall(r"([A-Z_]+) (-?[\d.]+)", cell)
    if named and len(named) == len(keys):  # "REUSE 0, DENSE 1, ..."
        values = {name: yaml.safe_load(v) for name, v in named}
        return {k: values[k.rsplit(".", 1)[1]] for k in keys}
    if len(keys) > 1 and "/" in cell and not cell.startswith(("http", "eval/", "var/")):
        parts = [yaml.safe_load(p.strip()) for p in cell.split("/")]
        if len(parts) == len(keys):
            return dict(zip(keys, parts, strict=True))
    try:
        value = yaml.safe_load(cell)
    except yaml.YAMLError:
        return None
    return dict.fromkeys(keys, value)


def test_schema_keys_equal_base_yaml_keys() -> None:
    documented = {k for keys, _ in _schema_rows() for k in keys}
    actual = set(_base_leaves())
    assert sorted(actual - documented) == [], "keys in base.yaml missing from CONFIG_SCHEMA"
    assert sorted(documented - actual) == [], "keys in CONFIG_SCHEMA missing from base.yaml"


def test_documented_literal_defaults_match_base_yaml() -> None:
    actual = _base_leaves()
    checked = 0
    for keys, cell in _schema_rows():
        literal_keys = [
            k for k in keys if k not in DESCRIBED_DEFAULTS and not k.startswith(DESCRIBED_PREFIXES)
        ]
        if not literal_keys:
            continue
        documented = _parse_defaults(keys, cell)
        assert documented is not None, f"cannot parse default {cell!r} for {keys}"
        for key in literal_keys:
            assert documented[key] == actual[key], f"{key}: schema {cell!r} vs base.yaml"
            checked += 1
    assert checked > 100  # guards against the parser silently matching nothing


@pytest.mark.parametrize("key", sorted(DESCRIBED_DEFAULTS))
def test_described_defaults_exist_in_base_yaml(key: str) -> None:
    assert key in _base_leaves()


def test_experiment_section_keys_match_model() -> None:
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    section = _section(text, "## 3. `experiment` section", "## 4.")
    documented = {cells[0].removeprefix("experiment.") for cells in _rows(section)}
    assert documented == set(ExperimentConfig.model_fields)

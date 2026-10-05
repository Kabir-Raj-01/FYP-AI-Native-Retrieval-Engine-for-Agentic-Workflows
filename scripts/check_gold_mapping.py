"""Report gold-evidence mapping coverage (DATA.md §4).

    python scripts\\check_gold_mapping.py multihop [--config PATH]

Reads ``gold_map.json``, ``gold_unmapped.json`` and ``split.json`` written by
``python -m data.ingest`` and prints coverage overall and per split. Exits 1 when coverage
is below ``data.gold_min_coverage`` (acceptance: >= 95 %) or the files are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.ingest import GOLD_MAP_FILE, GOLD_UNMAPPED_FILE, SPLIT_FILE, processed_dir
from engine.config import EngineConfig, load_config
from engine.errors import ConfigError


@dataclass(frozen=True)
class Coverage:
    mapped: int
    unmapped: int

    @property
    def entries(self) -> int:
        return self.mapped + self.unmapped

    @property
    def ratio(self) -> float:
        return self.mapped / self.entries if self.entries else 1.0


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def coverage(
    gold_map: dict[str, list[list[str]]],
    unmapped: list[dict[str, Any]],
    task_ids: set[str] | None = None,
) -> Coverage:
    """Coverage over ``task_ids`` (all tasks if None)."""

    def keep(task_id: str) -> bool:
        return task_ids is None or task_id in task_ids

    mapped = sum(len(groups) for t, groups in gold_map.items() if keep(t))
    missing = sum(1 for u in unmapped if keep(u["task_id"]))
    return Coverage(mapped, missing)


def check(cfg: EngineConfig, dataset: str, out: Any = print) -> bool:
    folder = processed_dir(cfg, dataset)
    paths = [folder / GOLD_MAP_FILE, folder / GOLD_UNMAPPED_FILE, folder / SPLIT_FILE]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing; run: python -m data.ingest --dataset {dataset}"
            )
    gold_map, unmapped, split = (_load(p) for p in paths)
    total = coverage(gold_map, unmapped)
    out(
        f"{dataset}: {total.mapped}/{total.entries} gold evidence entries mapped "
        f"({total.ratio:.2%}); unmapped {total.unmapped}"
    )
    for name in ("dev", "test", "dev_subset"):
        part = coverage(gold_map, unmapped, set(split[name]))
        out(f"  {name:<10} {part.mapped}/{part.entries} ({part.ratio:.2%})")
    threshold = cfg.data.gold_min_coverage
    ok = total.ratio >= threshold
    out(
        f"{'PASS' if ok else 'FAIL'}: coverage {'>=' if ok else '<'} {threshold:.0%} "
        "(data.gold_min_coverage)"
    )
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_gold_mapping", description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        return 0 if check(load_config(args.config), args.dataset) else 1
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

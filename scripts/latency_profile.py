"""Measure per-action latency for the controller's latency signal (ACTION_POLICY §4).

    python scripts\\latency_profile.py [--config PATH] [--queries N] [--out PATH]

Runs each action type on dev-subset questions (split.json — never the test split) against
the live index and writes the median wall-clock milliseconds per action to
``controller.latency_profile_path`` (default ``configs/latency_profile.yaml``):

- REUSE          ``Cache.lookup`` returning an exact hit (the cached candidates)
- DENSE          dense search at ``retrieval.default_k`` (query embedding + request)
- SPARSE_HYBRID  hybrid search at ``retrieval.default_k`` (the mode used for mixed needs)
- RERANK_EXPAND  dense search at ``min(default_k + expand_step_k, max_k)``
- TRANSFORM      not measurable until the planner and LLM client exist: the value of
                 ``controller.default_latency_ms.TRANSFORM`` is kept, marked
                 ``source: default``.

Reranking is excluded: it follows every candidate-producing action alike. Query embeddings
are not cached (``embedding.lru_size`` = 0 for the run), so every timing includes a cold
query embedding — an upper bound. The REUSE cache lives in a temporary database.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from engine import factory
from engine.cache.interfaces import Cache
from engine.config import EngineConfig, load_config
from engine.embeddings.interfaces import Embedder
from engine.errors import BackendError, ConfigError
from engine.monitor.logging import configure_logging
from engine.retrieval.interfaces import RetrieverAdapter
from engine.state.models import CacheEntry, NeedType, RetrievalMode
from engine.state.sqlite import connect
from engine.text import normalize

ACTIONS = ("REUSE", "DENSE", "SPARSE_HYBRID", "RERANK_EXPAND", "TRANSFORM")
MEASURED = ("REUSE", "DENSE", "SPARSE_HYBRID", "RERANK_EXPAND")
PROFILE_TASK_PREFIX = "latency-profile"
DEFAULT_QUERIES = 50  # profiling sample size (a script parameter, not an engine setting)


def profile_task_id(i: int) -> str:
    return f"{PROFILE_TASK_PREFIX}-{i}"


def _ms(clock: Callable[[], float], fn: Callable[[], object]) -> float:
    start = clock()
    fn()
    return (clock() - start) * 1000.0


def _p90(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]


def measure(
    cfg: EngineConfig,
    queries: Sequence[str],
    retriever: RetrieverAdapter,
    embedder: Embedder,
    cache: Cache,
    *,
    clock: Callable[[], float],
) -> dict[str, list[float]]:
    """Per-action latencies in ms. Each query gets its own task (``profile_task_id(i)``,
    which ``cache`` must accept), so a REUSE lookup sees a small per-task cache as in a
    real run rather than every profiling entry at once."""
    k = cfg.retrieval.default_k
    deep = min(k + cfg.retrieval.expand_step_k, cfg.retrieval.max_k)
    timings: dict[str, list[float]] = {a: [] for a in MEASURED}
    for i, query in enumerate(queries):
        timings["DENSE"].append(
            _ms(clock, partial(retriever.search, query, RetrievalMode.DENSE, k))
        )
        timings["SPARSE_HYBRID"].append(
            _ms(clock, partial(retriever.search, query, RetrievalMode.HYBRID, k))
        )
        timings["RERANK_EXPAND"].append(
            _ms(clock, partial(retriever.search, query, RetrievalMode.DENSE, deep))
        )
        cache.put(
            CacheEntry(
                task_id=profile_task_id(i),
                entry_id=f"e{i}",
                normalized_query=normalize(query),
                query_vector=[float(x) for x in embedder.embed_queries([query])[0]],
                need_type=NeedType.SEMANTIC,
                mode=RetrievalMode.DENSE,
                candidates=retriever.search(query, RetrievalMode.DENSE, k),
                accepted_ids=[],
                created_at_step=i + 1,
            )
        )
    for i, query in enumerate(queries):
        timings["REUSE"].append(
            _ms(clock, partial(cache.lookup, profile_task_id(i), query, NeedType.SEMANTIC))
        )
    return timings


def build_profile(
    cfg: EngineConfig, timings: dict[str, list[float]], meta: dict[str, Any]
) -> dict[str, Any]:
    defaults = cfg.controller.default_latency_ms.model_dump()
    latency = {
        a: round(statistics.median(timings[a]), 2) if timings.get(a) else float(defaults[a])
        for a in ACTIONS
    }
    return {
        **meta,
        "latency_ms": latency,
        "p90_ms": {a: round(_p90(timings[a]), 2) for a in ACTIONS if timings.get(a)},
        "source": {a: "measured" if timings.get(a) else "default" for a in ACTIONS},
    }


def write_profile(profile: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by scripts/latency_profile.py - do not edit by hand "
        "(ACTION_POLICY section 4).\n"
    )
    text = header + yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)
    path.write_bytes(text.encode("utf-8"))  # bytes: LF, no BOM on Windows


def dev_subset_questions(cfg: EngineConfig, n: int) -> list[str]:
    root = cfg.paths.data_dir
    with (root / "processed" / "multihop" / "split.json").open(encoding="utf-8") as f:
        subset = json.load(f)["dev_subset"]
    with (root / "raw" / "multihop" / "MultiHopRAG.json").open(encoding="utf-8") as f:
        raw = json.load(f)
    by_id = {f"multihop:{i:04d}": q["query"] for i, q in enumerate(raw)}
    return [by_id[t] for t in subset[:n]]


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    import time

    parser = argparse.ArgumentParser(prog="latency_profile", description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        run_cfg = cfg.model_copy(deep=True)
        run_cfg.embedding.lru_size = 0  # every timing includes a cold query embedding
        run_cfg.logging.level = "WARNING"  # no per-call events on the console
        configure_logging(run_cfg.logging)
        queries = dev_subset_questions(cfg, args.queries)
        embedder = factory.build_embedder(run_cfg)
        retriever = factory.build_retriever(
            run_cfg, embedder, factory.build_sparse_encoder(run_cfg)
        )
        if not retriever.health():
            raise BackendError(f"collection {cfg.qdrant.collection!r} not reachable")
        retriever.search("warm-up", RetrievalMode.DENSE, cfg.retrieval.default_k)
        # The cache keeps a per-thread SQLite connection; on Windows an open file cannot be
        # deleted, so cleanup errors of the throwaway folder are ignored.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            run_cfg.paths.state_db = Path(tmp) / "profile.sqlite3"
            conn = connect(run_cfg.paths.state_db)
            now = datetime.now(UTC).isoformat()
            with conn:
                conn.executemany(
                    "INSERT INTO tasks (task_id, version, status, state_json, created_at, "
                    "updated_at) VALUES (?, 0, 'active', '{}', ?, ?)",
                    [(profile_task_id(i), now, now) for i in range(len(queries))],
                )
            conn.close()
            cache = factory.build_cache(run_cfg, embedder)
            timings = measure(run_cfg, queries, retriever, embedder, cache, clock=time.perf_counter)
        profile = build_profile(
            cfg,
            timings,
            {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "git_commit": _git_commit(),
                "collection": cfg.qdrant.collection,
                "queries": len(queries),
                "query_source": "dev_subset (split.json)",
            },
        )
        out = args.out if args.out is not None else cfg.controller.latency_profile_path
        write_profile(profile, out)
        print(
            yaml.safe_dump(
                {"latency_ms": profile["latency_ms"], "source": profile["source"]}, sort_keys=False
            ),
            end="",
        )
        print(f"wrote {out}")
    except (ConfigError, BackendError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

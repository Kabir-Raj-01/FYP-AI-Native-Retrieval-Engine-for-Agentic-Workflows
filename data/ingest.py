"""Ingestion pipeline (DATA.md §3).

    python -m data.ingest --dataset multihop [--config PATH] [--index]

Step 1 (always): chunks, gold map, split.

Writes to ``<paths.data_dir>/processed/<dataset>/``:
- ``chunks.jsonl``        one chunk per line: chunk_id, doc_id, text, metadata
- ``gold_map.json``       task_id -> one group of chunk ids per evidence entry (DATA §4)
- ``gold_unmapped.json``  evidence entries that could not be placed
- ``split.json``          dev / test / dev_subset task ids (DATA §5)
- ``ingest_manifest.json`` config hash, git commit, counts, coverage, output hashes

Step 2 (``--index``): embed, sparse-encode and upsert the chunks into ``qdrant.collection``
(DATA §3 steps 5-6; ``data/index.py``, resumable), then add an ``index`` section to the
manifest. Every step-1 output except the manifest is a deterministic function of the raw
files and config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data import multihop
from data.chunking import Span, Tokenizer, chunk_spans
from data.download import DownloadError, raw_dir, verify_raw_files
from data.gold import ChunkRef, GoldResult, build_gold_map
from data.ids import chunk_id_for
from data.split import Split, stratified_split
from engine import factory
from engine.config import EngineConfig, load_config
from engine.errors import BackendError, ConfigError

__all__ = ["HFTokenizer", "IngestResult", "ingest", "main", "processed_dir"]

SUPPORTED_DATASETS = (multihop.DATASET,)
CHUNKS_FILE = "chunks.jsonl"
GOLD_MAP_FILE = "gold_map.json"
GOLD_UNMAPPED_FILE = "gold_unmapped.json"
SPLIT_FILE = "split.json"
MANIFEST_FILE = "ingest_manifest.json"


class HFTokenizer:
    """``chunking.tokenizer`` loaded with transformers' fast tokenizer (offsets, no specials)."""

    def __init__(self, name: str, revision: str) -> None:
        from transformers import AutoTokenizer  # heavy import, only for real runs

        self.name = name
        self.revision = revision
        self._tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self._tok.model_max_length = sys.maxsize  # we count long texts; never truncate/warn

    def offsets(self, text: str) -> list[Span]:
        encoded = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(a), int(b)) for a, b in encoded["offset_mapping"]]


@dataclass(frozen=True)
class IngestResult:
    out_dir: Path
    documents: int
    chunks: int
    gold: GoldResult
    split: Split
    manifest: dict[str, Any]


def processed_dir(cfg: EngineConfig, dataset: str) -> Path:
    """``<paths.data_dir>/processed/<dataset>`` (DATA.md §3, §8)."""
    return cfg.paths.data_dir / "processed" / dataset


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_text(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    path.write_bytes(data)  # bytes: LF line endings and no BOM on Windows
    return _sha256_bytes(data)


def _json_text(obj: Any) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def config_hash(cfg: EngineConfig) -> str:
    canonical = json.dumps(cfg.model_dump(mode="json", by_alias=True), sort_keys=True)
    return _sha256_bytes(canonical.encode("utf-8"))


def git_state() -> dict[str, Any]:
    """Current commit and whether the tree is dirty (None if git is unavailable)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def ingest(
    cfg: EngineConfig,
    dataset: str,
    tokenizer: Tokenizer,
    *,
    tokenizer_name: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    git: Callable[[], dict[str, Any]] = git_state,
    log: Callable[[str], None] = print,
) -> IngestResult:
    """Run step 1 of the pipeline for ``dataset`` and write its outputs.

    Raises:
        ConfigError: unsupported dataset.
        DownloadError: raw files missing or failing their checksums.
        ValueError: a chunk exceeds ``chunking.chunk_size_tokens`` or the split is invalid.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ConfigError(f"ingestion supports {SUPPORTED_DATASETS}, not {dataset!r}")
    raw_hashes = verify_raw_files(cfg, dataset)
    raw = raw_dir(cfg, dataset)
    docs = multihop.load_corpus(raw)
    queries = multihop.load_queries(raw)
    log(f"loaded {len(docs)} documents, {len(queries)} queries")

    size, overlap = cfg.chunking.chunk_size_tokens, cfg.chunking.chunk_overlap_tokens
    rows: list[str] = []
    chunks_by_doc: dict[str, list[ChunkRef]] = {}
    token_counts: list[int] = []
    for doc in docs:
        refs: list[ChunkRef] = []
        for start, end in chunk_spans(doc.text, tokenizer, size, overlap):
            text = doc.text[start:end]
            n_tokens = len(tokenizer.offsets(text))
            if n_tokens > size:
                raise ValueError(f"chunk of {doc.doc_id} at {start} has {n_tokens} > {size} tokens")
            chunk_id = chunk_id_for(doc.doc_id, start)
            refs.append(ChunkRef(chunk_id, start, end))
            token_counts.append(n_tokens)
            row = {
                "chunk_id": chunk_id,
                "doc_id": doc.doc_id,
                "text": text,
                "metadata": {
                    "title": doc.title,
                    "source": doc.source,
                    "url": doc.url,
                    "published_at": doc.published_at,
                    "author": doc.author,
                    "category": doc.category,
                    "dataset": dataset,
                    "char_start": start,
                    "char_end": end,
                    "n_tokens": n_tokens,
                },
            }
            rows.append(json.dumps(row, ensure_ascii=False))
        chunks_by_doc[doc.doc_id] = refs
    ids = [r.chunk_id for refs in chunks_by_doc.values() for r in refs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chunk ids")
    log(f"chunked into {len(ids)} chunks (max {max(token_counts, default=0)} tokens)")

    gold = build_gold_map(queries, docs, chunks_by_doc, cfg.data.gold_fuzzy_threshold)
    log(
        f"gold map: {gold.mapped}/{gold.entries} evidence entries mapped "
        f"({gold.coverage:.2%}; exact {gold.exact}, fuzzy {gold.fuzzy}, "
        f"straddling {gold.straddling})"
    )

    types = {q.task_id: q.question_type for q in queries}
    split = stratified_split(
        types, cfg.data.dev_fraction, cfg.data.dev_subset_size, cfg.data.split_seed
    )
    split_doc = {
        "dataset": dataset,
        "seed": cfg.data.split_seed,
        "dev_fraction": cfg.data.dev_fraction,
        "dev_subset_size": cfg.data.dev_subset_size,
        "dev": split.dev,
        "test": split.test,
        "dev_subset": split.dev_subset,
    }

    out = processed_dir(cfg, dataset)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {
        CHUNKS_FILE: _write_text(out / CHUNKS_FILE, "\n".join(rows) + "\n"),
        GOLD_MAP_FILE: _write_text(out / GOLD_MAP_FILE, _json_text(gold.gold_map)),
        GOLD_UNMAPPED_FILE: _write_text(
            out / GOLD_UNMAPPED_FILE, _json_text([asdict(u) for u in gold.unmapped])
        ),
        SPLIT_FILE: _write_text(out / SPLIT_FILE, _json_text(split_doc)),
    }
    by_type = Counter(types.values())
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "created_at": now().isoformat(),
        **git(),
        "config_hash": config_hash(cfg),
        "raw_files_sha256": raw_hashes,
        "chunking": {
            "tokenizer": tokenizer_name,
            "tokenizer_revision": cfg.chunking.tokenizer_revision,
            "chunk_size_tokens": size,
            "chunk_overlap_tokens": overlap,
            "sentence_aware": cfg.chunking.sentence_aware,
        },
        "counts": {
            "documents": len(docs),
            "chunks": len(ids),
            "chunk_tokens_max": max(token_counts, default=0),
            "chunk_tokens_mean": round(sum(token_counts) / max(1, len(token_counts)), 1),
            "queries": len(queries),
            "queries_by_type": dict(sorted(by_type.items())),
        },
        "gold": {
            "entries": gold.entries,
            "mapped": gold.mapped,
            "exact": gold.exact,
            "fuzzy": gold.fuzzy,
            "straddling": gold.straddling,
            "unmapped": len(gold.unmapped),
            "coverage": round(gold.coverage, 6),
        },
        "split": {
            "dev": len(split.dev),
            "test": len(split.test),
            "dev_subset": len(split.dev_subset),
            "split_json_sha256": hashes[SPLIT_FILE],
            "dev_subset_sha256": _sha256_bytes("\n".join(split.dev_subset).encode("utf-8")),
        },
        "outputs_sha256": hashes,
    }
    _write_text(out / MANIFEST_FILE, _json_text(manifest))
    log(f"wrote {out}")
    return IngestResult(out, len(docs), len(ids), gold, split, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m data.ingest", description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--config", type=Path, default=None, help="config file (default: base)")
    parser.add_argument(
        "--index",
        action="store_true",
        help="also embed, sparse-encode and upsert into qdrant.collection",
    )
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        tokenizer = HFTokenizer(cfg.chunking.tokenizer, cfg.chunking.tokenizer_revision)
        result = ingest(cfg, args.dataset, tokenizer, tokenizer_name=tokenizer.name)
        if args.index:
            run_index(cfg, result)
    except (ConfigError, DownloadError, BackendError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def run_index(
    cfg: EngineConfig, result: IngestResult, *, log: Callable[[str], None] = print
) -> dict[str, Any]:
    """Step 2 with factory-built components; records it in the manifest and returns it."""
    from data.index import index_chunks  # step-2 code only when asked for

    embedder = factory.build_embedder(cfg)
    sparse = factory.build_sparse_encoder(cfg)
    writer = factory.build_index_writer(cfg)
    report = index_chunks(cfg, result.out_dir / CHUNKS_FILE, embedder, sparse, writer, log=log)
    section = {
        "collection": report.collection,
        "points_expected": report.chunks,
        "already_indexed": report.already_indexed,
        "upserted": report.upserted,
        "seconds": round(report.seconds, 1),
        "embedding_model": cfg.embedding.model_name,
        "embedding_revision": cfg.embedding.revision,
        "sparse_model": cfg.sparse.model_name,
        "sparse_revision": cfg.sparse.revision,
        "sparse_params": {"k1": cfg.sparse.k1, "b": cfg.sparse.b, "avg_len": cfg.sparse.avg_len},
        "qdrant_url": cfg.qdrant.url,
    }
    manifest = {**result.manifest, "index": section}
    _write_text(result.out_dir / MANIFEST_FILE, _json_text(manifest))
    log(
        f"indexed {report.upserted} new points into {report.collection} "
        f"({report.seconds / 60:.1f} min)"
    )
    return section


if __name__ == "__main__":
    raise SystemExit(main())

"""Download a benchmark's raw files and verify their checksums (DATA.md §2-§3; ADR-003).

    python -m data.download --dataset multihop [--config PATH] [--force]

Files listed in ``data.sources.<dataset>.files`` are fetched from the pinned Hugging Face
dataset revision into ``<paths.data_dir>/raw/<dataset>/``. Each file is streamed to a
``.part`` file, its SHA-256 compared with the configured value, and only then renamed
into place; a mismatch deletes the partial file and fails the run (ADR-003: record and
review new checksums before any ingestion). A file already present with the right
checksum is not downloaded again unless ``--force`` is given. Every run writes
``download_manifest.json`` next to the files (CLAUDE.md rule 13).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from engine.config import DataSource, EngineConfig, load_config
from engine.errors import ConfigError

__all__ = [
    "DownloadError",
    "FileRecord",
    "download_dataset",
    "main",
    "raw_dir",
    "verify_raw_files",
]

# Hugging Face "resolve" endpoint for one file of a dataset at a given commit.
HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
MANIFEST_NAME = "download_manifest.json"
_READ_CHUNK_BYTES = 1 << 20  # streaming buffer size; not a tunable


class DownloadError(Exception):
    """A file could not be fetched, or its content does not match the expected SHA-256."""


@dataclass(frozen=True)
class FileRecord:
    name: str
    url: str
    size: int
    sha256: str
    expected_sha256: str
    action: str  # "downloaded" | "already_present"


def raw_dir(cfg: EngineConfig, dataset: str) -> Path:
    """``<paths.data_dir>/raw/<dataset>`` (DATA.md §3, §8)."""
    return cfg.paths.data_dir / "raw" / dataset


def _source(cfg: EngineConfig, dataset: str) -> DataSource:
    try:
        return cfg.data.sources[dataset]
    except KeyError:
        known = ", ".join(sorted(cfg.data.sources)) or "none"
        raise ConfigError(f"no data.sources entry for {dataset!r} (configured: {known})") from None


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(client: httpx.Client, url: str, dest: Path) -> tuple[int, str]:
    """Stream ``url`` into ``dest``; return (bytes written, SHA-256 hex)."""
    digest = hashlib.sha256()
    size = 0
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            with dest.open("wb") as f:
                for chunk in response.iter_bytes(_READ_CHUNK_BYTES):
                    f.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
    except httpx.HTTPStatusError as exc:
        raise DownloadError(f"HTTP {exc.response.status_code} for {url}") from exc
    except httpx.HTTPError as exc:
        raise DownloadError(f"cannot download {url}: {exc}") from exc
    if declared is not None and int(declared) != size:
        raise DownloadError(f"truncated download of {url}: {size} of {declared} bytes")
    return size, digest.hexdigest()


def _download_file(
    client: httpx.Client,
    source: DataSource,
    name: str,
    expected: str,
    dest_dir: Path,
    *,
    force: bool,
    log: Callable[[str], None],
) -> FileRecord:
    url = HF_RESOLVE_URL.format(repo=source.hf_repo, revision=source.revision, filename=name)
    dest = dest_dir / name
    if dest.exists() and not force:
        actual = _sha256_of(dest)
        if actual == expected:
            log(f"{name}: already present, SHA-256 verified")
            return FileRecord(name, url, dest.stat().st_size, actual, expected, "already_present")
        log(f"{name}: present but SHA-256 differs, downloading again")

    part = dest.with_name(dest.name + ".part")
    try:
        size, actual = _fetch(client, url, part)
        if actual != expected:
            raise DownloadError(
                f"{name}: SHA-256 mismatch (expected {expected}, got {actual}); "
                "record and review the new checksum in ADR-003 before ingesting"
            )
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    log(f"{name}: downloaded {size:,} bytes, SHA-256 verified")
    return FileRecord(name, url, size, actual, expected, "downloaded")


def download_dataset(
    cfg: EngineConfig,
    dataset: str,
    *,
    client: httpx.Client | None = None,
    force: bool = False,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    log: Callable[[str], None] = print,
) -> Path:
    """Download and verify every file of ``dataset``; return the manifest's path.

    Raises:
        ConfigError: ``dataset`` has no ``data.sources`` entry.
        DownloadError: a file cannot be fetched or fails its checksum. Files verified
            before the failure are kept; no manifest is written for a failed run.
    """
    source = _source(cfg, dataset)
    dest_dir = raw_dir(cfg, dataset)
    dest_dir.mkdir(parents=True, exist_ok=True)
    accessed_at = now()

    own_client = client is None
    http = client or httpx.Client(timeout=cfg.data.download_timeout_s, follow_redirects=True)
    try:
        records = [
            _download_file(http, source, name, expected, dest_dir, force=force, log=log)
            for name, expected in source.files.items()
        ]
    finally:
        if own_client:
            http.close()

    manifest = {
        "dataset": dataset,
        "source": {"hf_repo": source.hf_repo, "revision": source.revision},
        "accessed_at": accessed_at.isoformat(),
        "files": [asdict(r) for r in records],
        "all_verified": all(r.sha256 == r.expected_sha256 for r in records),
    }
    path = dest_dir / MANIFEST_NAME
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    log(f"manifest: {path}")
    return path


def verify_raw_files(cfg: EngineConfig, dataset: str) -> dict[str, str]:
    """Check every configured raw file of ``dataset`` exists with its expected SHA-256.

    Returns ``{file name: sha256}``. Ingestion calls this first (DATA.md §3 step 1).

    Raises:
        ConfigError: ``dataset`` has no ``data.sources`` entry.
        DownloadError: a file is missing or its SHA-256 differs (run ``data.download``).
    """
    source = _source(cfg, dataset)
    found: dict[str, str] = {}
    for name, expected in source.files.items():
        path = raw_dir(cfg, dataset) / name
        if not path.is_file():
            raise DownloadError(
                f"{path} is missing; run: python -m data.download --dataset {dataset}"
            )
        actual = _sha256_of(path)
        if actual != expected:
            raise DownloadError(f"{path}: SHA-256 {actual} differs from the expected {expected}")
        found[name] = actual
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m data.download", description=__doc__)
    parser.add_argument("--dataset", required=True, help="a key of data.sources, e.g. multihop")
    parser.add_argument("--config", type=Path, default=None, help="config file (default: base)")
    parser.add_argument("--force", action="store_true", help="download even if files verify")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        download_dataset(cfg, args.dataset, force=args.force)
    except (ConfigError, DownloadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

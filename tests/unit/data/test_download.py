"""data/download.py with httpx.MockTransport (no network)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from data import download
from data.download import DownloadError, download_dataset, raw_dir
from engine.config import DataSource, EngineConfig
from engine.errors import ConfigError

CONTENT = {"a.json": b'{"a": 1}\n', "b.json": b"[1, 2, 3]\n"}
FIXED_NOW = datetime(2026, 10, 5, 9, 30, tzinfo=UTC)
REV = "f" * 40


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def dl_cfg(cfg: EngineConfig, tmp_path: Path) -> EngineConfig:
    cfg.paths.data_dir = tmp_path / "data"
    cfg.data.sources = {
        "mini": DataSource(
            hf_repo="org/mini", revision=REV, files={n: sha(b) for n, b in CONTENT.items()}
        )
    }
    return cfg


def serving(
    content: dict[str, bytes], calls: list[str] | None = None, status: int = 200
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        name = request.url.path.rsplit("/", 1)[1]
        if status != 200 or name not in content:
            return httpx.Response(status if status != 200 else 404)
        return httpx.Response(200, content=content[name])

    return httpx.Client(transport=httpx.MockTransport(handler))


def run(cfg: EngineConfig, client: httpx.Client, **kw: object) -> Path:
    messages: list[str] = []
    return download_dataset(
        cfg, "mini", client=client, now=lambda: FIXED_NOW, log=messages.append, **kw
    )


def test_downloads_verifies_and_writes_the_manifest(dl_cfg: EngineConfig) -> None:
    calls: list[str] = []
    manifest_path = run(dl_cfg, serving(CONTENT, calls))
    target = raw_dir(dl_cfg, "mini")
    assert {p.name for p in target.iterdir()} == {"a.json", "b.json", "download_manifest.json"}
    assert (target / "a.json").read_bytes() == CONTENT["a.json"]
    assert calls[0] == f"https://huggingface.co/datasets/org/mini/resolve/{REV}/a.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "mini"
    assert manifest["source"] == {"hf_repo": "org/mini", "revision": REV}
    assert manifest["accessed_at"] == FIXED_NOW.isoformat()
    assert manifest["all_verified"] is True
    a = next(f for f in manifest["files"] if f["name"] == "a.json")
    assert a["size"] == len(CONTENT["a.json"])
    assert a["sha256"] == a["expected_sha256"] == sha(CONTENT["a.json"])
    assert a["action"] == "downloaded"


def test_second_run_skips_verified_files(dl_cfg: EngineConfig) -> None:
    run(dl_cfg, serving(CONTENT))
    calls: list[str] = []
    manifest = json.loads(run(dl_cfg, serving(CONTENT, calls)).read_text(encoding="utf-8"))
    assert calls == []
    assert {f["action"] for f in manifest["files"]} == {"already_present"}


def test_force_downloads_again(dl_cfg: EngineConfig) -> None:
    run(dl_cfg, serving(CONTENT))
    calls: list[str] = []
    run(dl_cfg, serving(CONTENT, calls), force=True)
    assert len(calls) == len(CONTENT)


def test_corrupted_existing_file_is_downloaded_again(dl_cfg: EngineConfig) -> None:
    target = raw_dir(dl_cfg, "mini")
    target.mkdir(parents=True)
    (target / "a.json").write_bytes(b"corrupted")
    run(dl_cfg, serving(CONTENT))
    assert (target / "a.json").read_bytes() == CONTENT["a.json"]


def test_checksum_mismatch_fails_and_leaves_no_file(dl_cfg: EngineConfig) -> None:
    tampered = {**CONTENT, "b.json": b"tampered"}
    with pytest.raises(DownloadError, match="SHA-256 mismatch"):
        run(dl_cfg, serving(tampered))
    target = raw_dir(dl_cfg, "mini")
    assert not (target / "b.json").exists()
    assert not (target / "b.json.part").exists()
    assert not (target / "download_manifest.json").exists()
    assert (target / "a.json").exists()  # verified before the failure: kept


def test_http_error_raises_download_error(dl_cfg: EngineConfig) -> None:
    with pytest.raises(DownloadError, match="HTTP 503"):
        run(dl_cfg, serving(CONTENT, status=503))
    assert list(raw_dir(dl_cfg, "mini").iterdir()) == []


def test_missing_file_on_server_raises(dl_cfg: EngineConfig) -> None:
    with pytest.raises(DownloadError, match="HTTP 404"):
        run(dl_cfg, serving({"a.json": CONTENT["a.json"]}))


def test_transport_error_raises_download_error(dl_cfg: EngineConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(DownloadError, match="cannot download"):
        run(dl_cfg, httpx.Client(transport=httpx.MockTransport(handler)))


def test_truncated_download_is_detected(dl_cfg: EngineConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = CONTENT["a.json"]
        return httpx.Response(200, content=body, headers={"Content-Length": str(len(body) + 5)})

    with pytest.raises(DownloadError, match="truncated"):
        run(dl_cfg, httpx.Client(transport=httpx.MockTransport(handler)))


def test_unknown_dataset_raises_config_error(dl_cfg: EngineConfig) -> None:
    with pytest.raises(ConfigError, match=r"no data\.sources entry for 'nope'"):
        download_dataset(dl_cfg, "nope", client=serving(CONTENT), log=lambda m: None)


def test_main_returns_1_with_a_message_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], dl_cfg: EngineConfig
) -> None:
    monkeypatch.setattr(download, "load_config", lambda path: dl_cfg)
    assert download.main(["--dataset", "nope"]) == 1
    assert "no data.sources entry" in capsys.readouterr().err


def test_main_returns_0_on_success(monkeypatch: pytest.MonkeyPatch, dl_cfg: EngineConfig) -> None:
    calls: list[tuple[str, bool]] = []

    def fake(cfg: EngineConfig, dataset: str, *, force: bool) -> Path:
        calls.append((dataset, force))
        return Path("manifest.json")

    monkeypatch.setattr(download, "load_config", lambda path: dl_cfg)
    monkeypatch.setattr(download, "download_dataset", fake)
    assert download.main(["--dataset", "mini", "--force"]) == 0
    assert calls == [("mini", True)]

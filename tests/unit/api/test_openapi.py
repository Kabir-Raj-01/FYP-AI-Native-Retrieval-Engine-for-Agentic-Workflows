"""OpenAPI schema snapshot (TESTING §4.7): API_CONTRACT is frozen, so the generated schema
changes only with an ADR.

    $env:UPDATE_OPENAPI = "1"; pytest tests\\unit\\api\\test_openapi.py   # after the ADR
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from engine.api.app import create_app
from engine.config import EngineConfig
from engine.core import Engine
from tests.fakes import CorpusChunk, FakeRetriever

SNAPSHOT = Path(__file__).resolve().parents[2] / "fixtures" / "openapi.json"


def test_openapi_schema_matches_the_snapshot(
    cfg: EngineConfig, mini_corpus: list[CorpusChunk]
) -> None:
    schema = create_app(Engine(cfg, retriever=FakeRetriever(mini_corpus))).openapi()
    if os.environ.get("UPDATE_OPENAPI") == "1":
        text = json.dumps(schema, indent=1, sort_keys=True) + "\n"
        SNAPSHOT.write_bytes(text.encode("utf-8"))  # LF, also on Windows
        pytest.skip(f"snapshot rewritten: {SNAPSHOT}")
    assert SNAPSHOT.exists(), "no snapshot yet: run once with UPDATE_OPENAPI=1"
    assert schema == json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_every_contract_endpoint_is_served(
    cfg: EngineConfig, mini_corpus: list[CorpusChunk]
) -> None:
    paths = create_app(Engine(cfg, retriever=FakeRetriever(mini_corpus))).openapi()["paths"]
    assert {(p, m) for p, ms in paths.items() for m in ms} == {
        ("/v1/health", "get"),
        ("/v1/tasks", "post"),
        ("/v1/tasks/{task_id}/step", "post"),
        ("/v1/tasks/{task_id}/run", "post"),
        ("/v1/tasks/{task_id}/result", "get"),
        ("/v1/tasks/{task_id}/state", "get"),
        ("/v1/tasks/{task_id}/trace", "get"),
        ("/v1/retrieve/direct", "post"),
    }

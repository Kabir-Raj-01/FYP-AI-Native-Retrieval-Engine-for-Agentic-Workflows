"""LLM response cache (LLM_CLIENT_SPEC §7; ARCHITECTURE §7).

SQLite at ``paths.llm_cache_db``: ``llm_cache(key PK, response_json, usage_json,
created_at)``. One connection per thread (WAL). The key is computed by the client.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["LLMResponseCache"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
  key           TEXT PRIMARY KEY,
  response_json TEXT NOT NULL,
  usage_json    TEXT NOT NULL,
  created_at    TEXT NOT NULL
);
"""


class LLMResponseCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)
            self._local.conn = conn
        return conn

    def get(self, key: str) -> tuple[str, dict[str, Any]] | None:
        row = (
            self._conn()
            .execute("SELECT response_json, usage_json FROM llm_cache WHERE key = ?", (key,))
            .fetchone()
        )
        if row is None:
            return None
        return str(json.loads(row[0])["text"]), dict(json.loads(row[1]))

    def put(self, key: str, text: str, usage: dict[str, Any]) -> None:
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, response_json, usage_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, json.dumps({"text": text}), json.dumps(usage), datetime.now(UTC).isoformat()),
            )

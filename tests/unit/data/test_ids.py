"""data/ids.py follows DATA.md §3 and RETRIEVAL_SPEC §2."""

from __future__ import annotations

import hashlib
import re
from uuid import UUID, uuid5

from data.ids import chunk_id_for, doc_id_for, point_id_for

NAMESPACE = UUID("05855794-d843-5065-9ec6-07da88359bb4")  # qdrant.id_namespace in base.yaml


def test_doc_id_is_sha1_of_url_prefix() -> None:
    url = "https://example.org/a"
    assert doc_id_for(url) == hashlib.sha1(url.encode()).hexdigest()[:16]
    assert re.fullmatch(r"[0-9a-f]{16}", doc_id_for(url))


def test_chunk_id_follows_the_data_md_formula() -> None:
    assert chunk_id_for("abc", 120) == hashlib.sha1(b"abc:120").hexdigest()[:16]
    assert chunk_id_for("abc", 0) != chunk_id_for("abc", 1)


def test_point_id_is_uuid5_of_chunk_id() -> None:
    assert point_id_for("c1", NAMESPACE) == uuid5(NAMESPACE, "c1")
    assert point_id_for("c1", NAMESPACE).version == 5
    assert point_id_for("c1", NAMESPACE) == point_id_for("c1", NAMESPACE)

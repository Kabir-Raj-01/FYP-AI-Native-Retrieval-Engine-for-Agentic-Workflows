# ADR-001: Embedding model, chunking policy, sparse method, collection naming

> Status: Proposed · Date: 2026-09-25 · Updated: 2026-10-05 · Author(s): Kabir · Reviewer: Samay / Vishal

## Context
All systems share one index (EVAL_PROTOCOL §1). The embedding model also drives cache
similarity, gap coverage, novelty and redundancy, so it must be fixed before ingestion;
changing it later invalidates every result.

## Decision (proposed defaults — confirm or edit, then set Status: Accepted)
- Dense embedding: `BAAI/bge-base-en-v1.5`, 768-d, normalised; query prefix = the model
  card's retrieval instruction, passage prefix empty. (Confirm prefix text on the model card.)
- Sparse: BM25 via fastembed `Qdrant/bm25` (confirm model id and Qdrant sparse config in
  LIBRARY_NOTES).
- Chunking: 384 tokens (tokenizer of the embedding model), 48-token overlap,
  sentence-aware.
- Chunk id: `sha1(f"{doc_id}:{char_start}")[:16]`; Qdrant point id: UUIDv5 with
  `qdrant.id_namespace`.
- Collection naming: `<dataset>_v<N>`; any change to the items above → N+1.

### Pinned revisions and BM25 parameters (2026-10-05)
- `embedding.revision` and `chunking.tokenizer_revision` =
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` (BAAI/bge-base-en-v1.5; the BGE query prefix
  "Represent this sentence for searching relevant passages: " confirmed equivalent to
  sentence-transformers' `prompt=` argument). `sparse.revision` =
  `22b8d2af71a76161e18dd432d2cee0eefa66e412` (Qdrant/bm25). Re-chunking with the pinned
  tokenizer reproduced `chunks.jsonl`, `gold_map.json` and `split.json` byte-for-byte.
- BM25: `sparse.k1 = 1.2`, `sparse.b = 0.75`, `sparse.avg_len = 256.0` (fastembed's
  defaults, now explicit; measured MultiHop chunks average 259.3 words). The Qdrant sparse
  vector uses `modifier=idf`, as fastembed's Bm25 requires.

### Interfaces for sparse encoding and indexing (2026-10-05)
ARCHITECTURE §2 had no component that produces sparse vectors or writes the index. Added:
`SparseEncoder` (`engine/embeddings/interfaces.py`) and `IndexWriter`
(`engine/retrieval/interfaces.py`), with the transport types `SparseVector` and
`IndexPoint` defined next to them — like `Message`/`LLMResult` (ADR-009 §10) they never
appear in task state, so `engine/state/models.py` is unchanged. This keeps `qdrant_client`
inside `engine/retrieval` and the model libraries inside `engine/embeddings` (enforced by
`tests/unit/test_import_boundaries.py`); `data/` uses only `engine.factory`.

## Alternatives considered
- `intfloat/e5-base-v2` — comparable quality; requires "query:"/"passage:" prefixes;
  acceptable substitute if BGE underperforms on a dev spot-check.
- Larger models (large variants) — better quality but ~3× slower on CPU; not needed at
  this corpus size.
- 256-token chunks — finer evidence but more chunks per gold fact straddling boundaries;
  512-token — coarser, blurs gap coverage.

## Consequences
Keys under `embedding`, `sparse`, `chunking`, `qdrant.collection` become Frozen.

## Evidence
Before accepting: ingest the dev subset with this setting, run `check_gold_mapping.py`
(≥ 95 %) and dense_oneshot Recall@10 as a sanity check; record the numbers here.
- 2026-10-05 — chunking only (embedding and Qdrant indexing not yet run): 384-token
  chunks, 48-token whole-sentence overlap, `BAAI/bge-base-en-v1.5` tokenizer
  (transformers 5.18.0). 609 documents → 4,441 chunks (mean 344.3, max 384 tokens).
  `check_gold_mapping.py multihop`: **6,084 / 6,084 evidence entries mapped (100 %)**, all
  by exact match, none straddling a chunk boundary; 254 groups have two chunks (the fact
  lies in an overlap). Dev 100 %, test 100 %, dev subset 100 %.
- Pending: dense_oneshot Recall@10 on the dev subset (needs the Embedder, the index and
  the eval harness).

## Reversal condition
Cheap until the Week 10 checkpoint. Afterwards only with a new ADR and full re-runs.

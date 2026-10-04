# ADR-001: Embedding model, chunking policy, sparse method, collection naming

> Status: Proposed · Date: 2026-09-25 · Author(s): Kabir · Reviewer: Samay / Vishal

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

## Reversal condition
Cheap until the Week 10 checkpoint. Afterwards only with a new ADR and full re-runs.

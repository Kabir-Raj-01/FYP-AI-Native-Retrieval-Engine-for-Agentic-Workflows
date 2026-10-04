# ADR-005: Cross-encoder reranker model

> Status: Proposed · Date: 2026-09-25 · Author(s): Samay · Reviewer: Kabir

## Decision (proposed default — confirm, then Accept)
- Model: `BAAI/bge-reranker-base`; output mapped to [0,1] (sigmoid if logits — verify
  in LIBRARY_NOTES); max input 512 tokens; CPU.
- Alternative: an MS MARCO MiniLM cross-encoder (faster, weaker on long news passages).

## Evidence
Before accepting: Recall@10 before/after reranking on the dev subset for dense_oneshot,
and median rerank latency for k = 20 on a team laptop.

## Consequences
`rerank.cross_encoder_model` Frozen; used identically by all systems.

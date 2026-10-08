# Architecture Decision Records

> Status: Living index · Owner: All · Last updated: 2026-10-09

An ADR records one decision that is expensive to reverse or that affects experimental
validity. Copy `TEMPLATE.md`, number sequentially, open a PR, get one teammate's review.
Accepted ADRs are never edited except to mark them *Superseded by ADR-0XX*.

| ADR | Title | Status | Owner | Needed by |
|---|---|---|---|---|
| 001 | Embedding model, chunking policy, sparse method, collection naming | Accepted | Kabir | Week 7 (before ingestion) |
| 002 | Execution model: synchronous core; TaskStore as single state source | Accepted | Samay | Week 7 |
| 003 | Dataset source, licence, checksums, split, official answer metric | Accepted | Kabir | Week 8 |
| 004 | LLM providers and models for planner, verifier, generator; prompt versions | Accepted (qwen2.5:3b, Ollama) | Vishal | Week 11 |
| 005 | Cross-encoder reranker model | Accepted (bge-reranker-base) | Samay | Week 12 |
| 006 | Calibrated weights and thresholds (freeze) | Not started | Samay | Feb 2027 |
| 007 | HotpotQA setting and sample | Not started | Kabir | Mar 2027 |
| 008 | Curated benchmark corpora and annotation protocol | Not started | All | Mar 2027 |
| 009 | State contract amendments (pre-implementation review) | Accepted | Kabir | Week 7 |
| 010 | API request limits come from configuration | Accepted | Vishal | Week 8 |
| 011 | Recorded deviations from the approved proposal | Proposed (supervisor) | All | Week 7 |
| 012 | `DirectResult`, the return type of `Engine.retrieve_direct` (+ A1 rerank latency) | Accepted | Kabir | Week 9 |
| 013 | Preliminary (FYP-I) calibration: relevance threshold, rerank top-n, controller values | Accepted (keys stay Draft; ADR-006 freezes) | Kabir | Week 16 |

Also write an ADR for: any contract change (STATE_SPEC, API_CONTRACT, trace schema), any
change to a Frozen key, a new baseline, or anything a thesis examiner might ask "why?" about.

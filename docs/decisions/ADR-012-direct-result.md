# ADR-012: `DirectResult` — the return type of `Engine.retrieve_direct`

> Status: Accepted (2026-10-08, Kabir for the team; Samay/Vishal review after the fact) · Date: 2026-10-05 · Author(s): Kabir · Reviewer: Samay, Vishal

## Context
ARCHITECTURE §6 defines `Engine.retrieve_direct(query, mode, k, rerank)` but no return
type. The HTTP response `DirectResponse` lives in `engine/api/schemas.py` (an HTTP-only
wrapper), so the engine facade cannot return it, and CLAUDE.md rule 2 forbids returning a
raw tuple or dict across module boundaries. The direct path has three in-process callers
besides the router: the dense/hybrid one-shot baselines and fixed-iteration
(BASELINES_SPEC §2), which need the candidates, the relevance scores when reranking, and
the latency for their traces.

## Decision
1. A new shared model in `engine/state/models.py` (STATE_SPEC §6):
   ```python
   class DirectResult(BaseModel):
       model_config = ConfigDict(extra="forbid")
       candidates: list[Candidate]               # backend order, rank 0..k-1, with vectors
       scored: list[ScoredCandidate] | None = None  # rerank=True: relevance-only reranker
                                                    # output (≤ rerank.top_n_output)
       latency_ms: float                          # wall clock: search (+ rerank)
   ```
2. `Engine.retrieve_direct(query, mode, k=None, rerank=False) -> DirectResult`:
   `k=None` → `retrieval.default_k`; `k` outside `[1, retrieval.max_k]` → `ValueError`;
   `rerank=True` without a reranker → `ConfigError`. Never creates or reads task state.
3. The router maps it to `DirectResponse`: without rerank, `candidates` =
   `result.candidates`, `relevances = None`; with rerank, `candidates` = the reranked
   candidates in relevance order and `relevances` aligned with them. Vectors are serialized
   as null (ADR-009 addendum A1).
4. Until a Reranker exists (ADR-005, Weeks 11–12), `POST /v1/retrieve/direct` with
   `rerank=true` returns 422 `VALIDATION_ERROR` ("rerank is not available yet").

## Alternatives considered
- Return `DirectResponse` from the engine — couples the engine to the HTTP schema module.
- Return `list[Candidate] | list[ScoredCandidate]` depending on `rerank` — a union return
  type every caller has to branch on, and no place for the latency.
- Put `relevances` on `Candidate` — changes a model used everywhere for one path.

## Consequences
STATE_SPEC §6 gains `DirectResult` (applied in the same change). API_CONTRACT is
unchanged: `DirectRequest`/`DirectResponse` are used exactly as specified; only the
undefined `rerank=true` behaviour is fixed here. Like ADR-009, this must be Accepted before
`engine/state/models.py` is merged.

## Reversal condition
If the baselines need more per-call data (e.g. separate search and rerank latencies),
extend the model through a new ADR rather than adding ad-hoc return values.

## Amendment A1 (2026-10-08) — separate rerank latency
The case foreseen above (ADR-005 open item 4): with the reranker on, a one-shot baseline's
TraceStep put search + rerank under `latency_ms["execute"]`, while the controlled loop
records `execute` and `rerank` as separate stages. `DirectResult` gains
`rerank_latency_ms: float | None = None` (the rerank part of `latency_ms`; `None` without
rerank). The baselines record `{"execute": search, "rerank": rerank, "total": both}`
(`baselines.base.step_latency`), the same stages as the controlled loop.
`DirectResponse.latency_ms` (API_CONTRACT) is unchanged: still the total.

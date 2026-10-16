# Adaptive (State-Aware) Reranking

> Status: Stable · Owner: Samay · Last updated: 2026-10-16 · Related: EVIDENCE_MANAGER_SPEC.md, ADR-009, ADR-014

## 1. Purpose
Order candidates by usefulness **to the current trajectory**, not only by query
relevance: prefer candidates that cover open gaps and add new information, and avoid
selecting near-duplicates of each other. No reranker training — a pretrained
cross-encoder plus state-derived signals.

Interface: `Reranker.rerank(query, candidates, state: RetrievalState | None, goal: str |
None = None) -> list[ScoredCandidate]` (sorted, length ≤ `rerank.top_n_output`). `state=None` means
relevance-only mode (ADR-009 "Also"): no gap coverage, no novelty against E_t, no MMR —
`final = relevance`, the same behaviour as `ablation.state_aware_rerank=false` (§3). This
is how `Engine.retrieve_direct` and the baselines call it — they have no task state.

## 2. Signals (per candidate c)
| Signal | Definition |
|---|---|
| relevance(c) | cross-encoder score for (q, c.text) mapped to [0,1] with a sigmoid if the model outputs logits (check the model's output type — LIBRARY_NOTES). `q` per `rerank.relevance_query` (ADR-014): `subquery` → the pending query; `goal` → the task goal G (`goal` argument, else `state.goal`); `max` → the per-candidate maximum of both scores. With no goal, or goal = query (the one-shot paths), `q` = the pending query in every mode. |
| gap_coverage(c) | `max_{g ∈ OPEN gaps} cos(c.vector, emb_query(g.description))`, clipped to [0,1]; `covered_gap_id` = argmax. 0 if `ablation.gap_tracking` false. |
| novelty(c) | `1 − max_{e ∈ E_t} cos(c.vector, e.vector)`; 1 if E_t empty; 0 if c.chunk_id already in E_t. |
| redundancy(c ∣ S) | `max_{s ∈ S} cos(c.vector, s.vector)` over candidates S already selected above c in this list; 0 for the first. 0 if `ablation.redundancy_signal` false. |

Candidate vectors come from Qdrant (`with_vectors=True`) or the cache entry; gap
descriptions are embedded on the query side (RETRIEVAL_SPEC §5).

## 3. Scoring and selection (greedy, MMR-style)
```
base(c) = w_rel·relevance + w_gap·gap_coverage + w_nov·novelty
S = []
while candidates remain and |S| < rerank.top_n_output:
    pick c maximising  final(c) = base(c) − w_red·redundancy(c | S)
    append c to S (record its final_score and redundancy at selection time)
return S
```
Weights `rerank.weights.{relevance, gap_coverage, novelty, redundancy}` are non-negative.
With `ablation.state_aware_rerank` false, or when `state=None` (§1) — the case for the
direct path and all baselines: `final = relevance`, no MMR (plain cross-encoder order).

Why both novelty and redundancy: novelty compares a candidate with evidence already
accepted in earlier steps; redundancy compares it with other candidates in the same
list. Without redundancy, two near-identical new chunks could both be selected.

## 4. Cost and limits
- Cross-encoder batch size `rerank.batch_size`; one reranker call per step with
  candidates (`budget.reranker_calls_used += 1`). Candidates per call ≤ `retrieval.max_k`.
- Cross-encoder input truncated to `rerank.max_input_tokens`.
- Runs on CPU acceptably at k ≤ 50 with a base-size model; GPU optional (`rerank.device`).

**As built (2026-10-05, `engine/rerank/state_aware.py`, `cross_encoder.py`):** with
`ablation.state_aware_rerank` false *and* a state, `gap_coverage` and `novelty` are still
computed (the EvidenceManager's `LOW_NOVELTY` check reads them, so the ablation changes
only the ordering); `final = relevance`, cross-encoder order, redundancy 0. With
`state=None`: gap coverage 0, novelty 1, redundancy 0. Ties keep backend order; cosines are
clipped to [0, 1]; a state-aware call needs candidate vectors (`ValueError` otherwise). The
cross-encoder loads lazily at `rerank.revision` and maps logits through a sigmoid only for
an `Identity` head.

## 5. Outputs used downstream
EvidenceManager uses `relevance` and `novelty` for acceptance and `final_score` for
ordering; the controller reads `top_relevances` (stored in `LastResult`) for
RERANK_EXPAND ambiguity. All four signals and final score for every candidate are logged
in `rerank.done` events.

## 6. Tests
- **FR7 constructed case:** candidate A — relevance 0.9, gap_coverage 0.5, near-duplicate
  of an item in E_t (novelty 0.02); candidate B — relevance 0.6, gap_coverage 0.8,
  novelty 0.9. With the default weights in `configs/base.yaml`, final(B) > final(A)
  (0.70 vs 0.644 with weights 0.6/0.2/0.2). With state-aware off, A ranks above B.
- Two near-identical candidates: second is pushed down by redundancy.
- Empty E_t → novelty 1 for all; empty candidate list → empty output.
- Sigmoid mapping keeps relevance in [0,1] for logit outputs.

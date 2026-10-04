# Evidence Manager

> Status: Stable (formulas) · Values: Draft until calibration ADR · Owner: Samay · Last updated: 2026-09-25

## 1. Responsibilities
The only component that writes `evidence` and marks gaps RESOLVED. Per step it decides
which scored candidates become evidence, which gaps they resolve, and how much progress
the step made (evidence gain).
Interface: `EvidenceManager.update(state, scored: list[ScoredCandidate], step) -> EvidenceUpdate`.

## 2. Acceptance (for each scored candidate, in final_score order)
| Check | Reject code |
|---|---|
| chunk_id already in E_t | `DUPLICATE` |
| relevance < `evidence.relevance_threshold` | `LOW_RELEVANCE` |
| novelty < `evidence.novelty_threshold` | `LOW_NOVELTY` |
| already accepted `evidence.max_accept_per_step` this step | `STEP_CAP` |
| E_t full (`evidence.max_items`) and no evictable item with lower final_score | `CAP_REACHED` |
Otherwise accepted → new `EvidenceItem` (summary = text truncated to
`planner.evidence_summary_chars` at a sentence boundary; `vector` kept for similarity).
If E_t is full, the lowest-`final_score` **unpinned** item is evicted (logged). Pinned
items (those that resolved a gap) are never evicted.

## 3. Gap resolution
For each OPEN gap g, consider accepted items (this step and earlier) with
`relevance ≥ evidence.relevance_threshold`:
- coverage(e, g) = `cos(e.vector, emb_query(g.description))`.
- g is RESOLVED by the item with the highest coverage if that coverage
  ≥ `evidence.gap_coverage_threshold`.
- Optional verification (`llm.verifier.enabled`): for up to `evidence.verify_top_m`
  items with coverage in `[gap_coverage_threshold − evidence.verify_margin,
  gap_coverage_threshold)`, ask the verifier LLM (structured `{"supports": bool}`) whether
  the passage provides the information described by g. `supports=true` resolves g.
  Verification consumes the task's LLM budget and is skipped when it is exhausted.
- On resolution: `status = RESOLVED`, `resolved_by = evidence_id`, `closed_at_step`,
  item `pinned = True`, gap id appended to the item's `supports_gap_ids`.

## 4. Evidence gain (per step, in [0,1])
```
resolved_part = |newly resolved gaps| / max(1, |OPEN gaps at step start|)
quality_part  = mean(relevance(e) × novelty(e) for e in accepted this step)   (0 if none)
evidence_gain = w_res · resolved_part + w_q · quality_part
```
`w_res, w_q = evidence.gain_weights.{gap_resolution, quality}`, sum to 1.
With `ablation.gap_tracking` false, `evidence_gain = quality_part`.
Steps without candidates (TRANSFORM, STOP) have no gain value and do not affect
stagnation.

## 5. Sufficiency without gap tracking (ablation only)
With `ablation.gap_tracking` false there is one gap (the goal). It is RESOLVED when
E_t contains ≥ `evidence.min_items_for_sufficiency` items with relevance ≥ threshold and
the last step's gain < `monitor.stagnation_threshold` (no more new information arriving).

## 6. Null / unanswerable tasks
If no gap can be resolved, gaps become EXHAUSTED through stagnation (ACTION_POLICY §6)
and the task stops with STAGNATION, returning whatever evidence it has. The host/answer
generator decides "insufficient information". This is the intended behaviour for
MultiHop-RAG null queries and is measured by over-retrieval metrics.

## 7. Events
`evidence.accepted`, `evidence.rejected` (with code), `evidence.evicted`,
`gap.resolved`, `gap.verification` (OBSERVABILITY.md).

## 8. Tests
- Duplicate, low relevance, low novelty, step cap, capacity with and without evictable items.
- Pinned items are never evicted.
- Gap resolved exactly at threshold; not resolved just below; verification path with a
  fake LLM returning true/false; verification skipped when LLM budget is exhausted.
- Gain formula on hand-computed cases; gain ∈ [0,1] (property test).

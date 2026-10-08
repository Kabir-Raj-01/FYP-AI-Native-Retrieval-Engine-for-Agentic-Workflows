# ADR-005: Cross-encoder reranker model

> Status: Accepted (2026-10-09, Kabir for the team; Samay review after the fact) · Date: 2026-09-25 · Author(s): Samay · Reviewer: Kabir

## Decision (proposed default — confirm, then Accept)
- Model: `BAAI/bge-reranker-base`; output mapped to [0,1] (sigmoid if logits — verify
  in LIBRARY_NOTES); max input 512 tokens; CPU.
- Alternative: an MS MARCO MiniLM cross-encoder (faster, weaker on long news passages).

## Evidence
Before accepting: Recall@10 before/after reranking on the dev subset for dense_oneshot,
and median rerank latency for k = 20 on a team laptop.

### Probe 2026-10-05 (Kabir) — exploratory, dev subset only, not a reportable run
Setup: index `multihop_v1`; the 265 dev-subset tasks that have gold evidence; top-20 from
the direct path (`dense` or `hybrid`), reranked by the raw cross-encoder score
(relevance only, no gap/novelty/redundancy terms), Recall@10 / CER@10 on the reranked
top-10. `max_length = rerank.max_input_tokens` (512), `batch_size = rerank.batch_size`
(32), CPU. Scripts were scratch probes (not committed); the measurement reuses the
engine's factory-built adapter, so the candidates are exactly what `retrieve_direct` returns.

| Candidates | Reranker | Tasks | Recall@10 | CER@10 |
|---|---|---|---|---|
| dense | none | 265 | 0.477 | 0.177 |
| dense | ms-marco-MiniLM-L-6-v2 | 265 | 0.502 | 0.208 |
| hybrid | none | 265 | 0.589 | 0.291 |
| hybrid | ms-marco-MiniLM-L-6-v2 | 265 | **0.525** | 0.219 |
| dense | none | first 80 | 0.493 | 0.200 |
| dense | ms-marco-MiniLM-L-6-v2 | first 80 | 0.470 | 0.212 |
| dense | bge-reranker-base | first 80 | **0.556** | 0.237 |

Reading (exploratory; no significance test, bge on only 80 tasks because of its cost):
- `bge-reranker-base` lifts dense Recall@10 by +0.063 on its 80 tasks; MiniLM's effect on
  dense is within noise (+0.025 on 265, −0.023 on the same 80).
- MiniLM **hurts** hybrid (−0.064 on 265 tasks): a weak reranker can undo fusion gains.
  `bge-reranker-base` on hybrid candidates was not measured.
- The direct-path top-20 already holds much more gold than its top-10 (E0 trajectory
  coverage 0.661 vs Recall@10 0.477), so the head-room a good reranker can recover is real.

Model facts (sentence-transformers 6.1.0, verified; also in LIBRARY_NOTES):
- `BAAI/bge-reranker-base` @ `2cfc18c9415c912f9d8155881c133215df768a70`: `activation_fn`
  is `Sigmoid()`, `predict` returns probabilities. On these long multi-hop questions the
  raw values are tiny (observed range 0.000–0.003 for a dense top-20), so any absolute
  relevance threshold must be calibrated — rank order is informative, absolute value is not.
- `cross-encoder/ms-marco-MiniLM-L-6-v2` @ `233902d25c440f23af6f7d6e94d2946bac0bee0a`:
  `Identity()`, returns logits (needs a sigmoid to reach [0,1]).

Latency — **not valid yet.** All measurements so far ran with the laptop (i7-1185G7, 4
cores, no GPU) on battery, where the CPU was throttled (998 of 2995 MHz at one check) and
repeat runs varied by >2×. Order of magnitude only, 20 pairs per call:
MiniLM ≈ 1.4–3.2 s; bge-reranker-base ≈ 10–24 s. Re-measure on AC power ("Best
performance" mode) before quoting any number against the SRS NFRs. Even the best case
for bge (~10 s per step) makes CPU reranking the dominant latency term of every step that
reranks, which is a risk for the latency NFRs and for evaluation wall-clock.

Open for the owner (Samay) — not decided here:
1. Quality favours `bge-reranker-base` (proposed default); latency favours MiniLM, which
   showed no reliable quality gain. Options: accept bge and reduce the rerank input
   (fewer candidates, or a lower `max_input_tokens`), or accept bge and budget the time.
2. ~~Pin the reranker revision~~ — done 2026-10-05: `rerank.revision` (bge sha above), read
   by `engine/rerank/cross_encoder.py`; it must change together with the model.
3. Confirm on the full dev split (and on hybrid candidates) before the key is frozen.
4. With a reranker, the one-shot baselines' TraceStep `latency_ms["execute"]` would include
   rerank time, because `DirectResult.latency_ms` is search + rerank. Splitting it (e.g.
   `rerank_latency_ms`) is an ADR-012 amendment, to be done with the Reranker.

### Acceptance evidence 2026-10-09 — full dev subset (265 tasks with gold), paired
Reportable runs with `bge-reranker-base` (`20261008-184814_dense_oneshot_dev_f086ac2`,
`20261008-185336_hybrid_oneshot_dev_f086ac2`) against the 2026-10-05 runs without a
reranker (`20261005-074405_dense_oneshot_dev_b348d3e`, `20261005-102621_hybrid_oneshot_dev_fc73ded`);
`python -m eval.report --run <without> --run <with>`, 95 % paired bootstrap CI:

| Candidates | Recall@10 without → with | d [95 % CI] |
|---|---|---|
| dense top-20 | 0.477 → 0.565 | +0.089 [+0.059, +0.118] |
| hybrid top-20 | 0.589 → 0.619 | +0.030 [+0.001, +0.058] |

Unlike MiniLM, bge does not undo the hybrid fusion gain. (CER is not compared here: the
bundle grew from 10 to 20 items between the runs, ADR-013 Decision 2.)

Latency: uncached, 20 pairs per call, 4 eval workers sharing the CPU — median 16.9 s
per rerank in the hybrid run; the persistent score cache (`rerank.score_cache_enabled`)
makes repeat pairs ~3 ms. The NFR-level single-worker figure comes from the controller-ON
E1 run (one worker).

## Decision (accepted)
1. `BAAI/bge-reranker-base` at `rerank.revision`, sigmoid output, 512 tokens, CPU, for
   every system that reranks (open point 1: accept bge and budget the time — quality
   gain is significant on both candidate sets; MiniLM showed none on dense and hurt
   hybrid). Score cache on.
2. Open point 3 (full dev split) moves to the FYP-II calibration (ADR-006).
3. Open point 4 is done (ADR-012 A1, `DirectResult.rerank_latency_ms`).

## Consequences
`rerank.cross_encoder_model` Frozen; used identically by all systems.

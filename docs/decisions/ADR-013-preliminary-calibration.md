# ADR-013: Preliminary (FYP-I) calibration of the evidence threshold and controller values

> Status: Accepted (2026-10-08, Kabir for the team; Samay review after the fact) · Date: 2026-10-08 · Author(s): Kabir · Reviewer: Samay
> Scope: dev subset only (EVAL_PROTOCOL §1.2). The keys stay **Draft**: the full
> ACTION_POLICY §9 calibration (stage-1 grid of 56 weight points × thresholds, stage 2 on
> the full dev split) remains the FYP-II task that freezes them (`v0.8-frozen-config`).

## Context
Planner JSON validity on the calibration sample (ADR-004 reversal check): 45 of 47 INIT
attempts parsed (95.7 %), 37 of 40 tasks got an accepted plan, 3 the fallback.

The first live controller-ON runs (qwen2.5:3b planner, ADR-004) were unusable with the
documented placeholder values:
- `evidence.relevance_threshold` 0.5 rejected every candidate on many tasks, leaving empty
  evidence bundles. bge-reranker-base (ADR-005) is a probability head whose absolute values
  are low on long multi-part questions; gold passages score a median 0.44.
- With the uniform utility weights and `controller.min_utility` 0.05, a second search on a
  subgoal scored U ≈ 0.045 and the task stopped; TRANSFORM can never be chosen while
  δ > 0, because its latency signal is 1.0 by construction (it is the slowest action:
  14.5 s measured vs ~0.12 s for a search, `configs/latency_profile.yaml`).

## Decision 1 — `evidence.relevance_threshold` = 0.2 (§9 step 1)
Data: the bge relevance of every passage in the reranked top-10 of `dense_oneshot` on the
first 77 dev-subset questions (64 with gold; 640 scored passages, 86 gold). Acceptance =
relevance ≥ t; gold precision / recall:

| t | recall | precision | F1 | F2 | accepted / task | questions with none ≥ t |
|---|---|---|---|---|---|---|
| 0.05 | 0.791 | 0.176 | 0.288 | 0.465 | 6.0 | 17 / 77 |
| 0.1 | 0.756 | 0.208 | 0.327 | 0.495 | 4.9 | 21 / 77 |
| 0.15 | 0.721 | 0.217 | 0.333 | 0.492 | 4.5 | – |
| **0.2** | **0.698** | **0.233** | **0.349** | **0.499** | 4.0 | 26 / 77 |
| 0.3 | 0.605 | 0.241 | 0.344 | 0.465 | 3.4 | – |
| 0.5 (old) | 0.477 | 0.248 | 0.327 | – | 2.6 | 34 / 77 |

0.2 maximises both F1 and the recall-weighted F2. The same key gates fixed_iteration's
acceptance, so both multi-step systems use it. `evidence.novelty_threshold` and
`evidence.gap_coverage_threshold` are unchanged (they need controlled-run traces; FYP-II).

**Confirmed on controller-ON traces** (§9 step 1 asks for "dev traces"): best reranker
relevance per (task, chunk) over every `rerank.done` event of the uniform-weight run below
(40 tasks, 1,799 scored pairs, 78 gold) — recall-weighted F2 is highest at 0.2 again:

| t | 0.05 | 0.1 | 0.15 | **0.2** | 0.3 |
|---|---|---|---|---|---|
| gold recall | 0.756 | 0.705 | 0.628 | **0.615** | 0.538 |
| F2 | 0.337 | 0.352 | 0.338 | **0.358** | 0.354 |

## Decision 2 — `rerank.top_n_output` = 20 (was 10)
In the first calibration run 23 of 78 retrieved gold passages were scored by the reranker
and then cut by the top-10 *before* the EvidenceManager saw them (gold kept: 38 / 78). At
20 (= `retrieval.default_k` = `evidence.max_items`) no scored candidate is dropped before
acceptance, at no extra cost (all 20 were already scored), and the one-shot baselines'
bundle capacity (`top_n_output`) equals controller-ON's (`max_items`), so CER compares
like with like. CER@10 (all gold groups in the top 10) is reported as the size-independent
companion metric.

## Decision 3 — controller weights and `min_utility` unchanged (§9 step 3, reduced grid)
Grid: first 40 dev-subset tasks, B = 4, `max_workers` 2, commit `903ff1c`, objective
`J = CER − λ · mean_retrieval_calls / B` with λ = `controller.calibration_lambda` (0.1).
Run folders: `eval/runs/20261008-155217…161251_controller_on_dev_903ff1c`.

| weights α/β/γ/δ | min_utility | J | CER | R@10 | traj. cov. | calls | TRANSFORMs | planner REWRITE dup. / fallbacks |
|---|---|---|---|---|---|---|---|---|
| **.25/.25/.25/.25** | **0.05** | **0.085** | 0.152 | 0.467 | 0.854 | 2.65 | 0 | – |
| .25/.25/.25/.25 | 0 | 0.083 | 0.152 | 0.477 | 0.854 | 2.73 | 0 | – |
| .4/.3/.3/0 | 0.05 and 0 | 0.083 | 0.152 | 0.477 | 0.854 | 2.73 | 17 | 11 / 9 |
| .4/.2/.2/.2 | 0.05 | 0.082 | 0.152 | 0.477 | 0.854 | 2.78 | 0 | – |
| .6/.2/.2/0 | 0.05 and 0 | 0.075 | 0.152 | 0.437 | 0.790 | 3.08 | 34 | 23 / 15 |
| .4/.2/.2/.2 | 0 | 0.051 | 0.121 | 0.429 | 0.798 | 2.83 | 17 | 11 / 9 |
| .6/.2/.1/.1 | 0.05 | 0.047 | 0.121 | 0.429 | 0.798 | 2.95 | 18 | 11 / 9 |

(.6/.2/.1/.1 with min_utility 0 was lost to a run-id collision; two independent runs of
.6/.2/.1/.1 at 0.05 gave identical metrics — a reproducibility check of the stack.)

Reading: the defaults (uniform weights, `min_utility` 0.05) maximise J. Every weighting
that lets TRANSFORM win does no better and often worse: qwen2.5:3b's REWRITE proposals
are often a previous query again (`EMPTY_OR_DUPLICATE`) and end in the deterministic
fallback, so the extra planning buys no coverage. With δ > 0 TRANSFORM's latency signal
is 1.0 (14.5 s vs ~0.12 s per search), which in effect disables it — the right call for
this model. Revisit with a stronger planner model (ADR-004 reversal) in FYP-II.

## Consequences
Results reported in FYP-I are preliminary and say so; the FYP-II calibration ADR
supersedes this one and freezes the keys.

## Reversal condition
Superseded by the full §9 calibration in FYP-II, or earlier if a dev run shows the
threshold leaving more than a third of the questions with no accepted evidence.

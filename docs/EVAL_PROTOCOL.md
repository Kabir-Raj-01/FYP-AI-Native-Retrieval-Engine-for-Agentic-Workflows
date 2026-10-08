# Evaluation Protocol

> Status: Stable · Owner: Vishal · Last updated: 2026-10-04 · Related: BASELINES_SPEC.md, OBSERVABILITY.md

## 1. Rules that make results valid
1. **Shared infrastructure** for every system (BASELINES_SPEC §1).
2. **Dev vs test.** All development, debugging, calibration and prompt iteration uses the
   dev split. The test split is evaluated once per reported configuration, with frozen
   config (tag `v0.8-frozen-config` or later). Any second test run with changed settings is
   recorded as such in the report.
3. **Equal budget first.** The primary comparison caps all multi-step systems at the same
   `max_retrieval_calls`. Natural-policy results are reported alongside, never instead.
4. **Task-local cache only** (CACHE_SPEC). Leakage test must pass before any run is reported.
5. **Same generator** (role `generator`, temperature 0, same prompt) for all systems.
6. **Manifests.** A run without a complete manifest is not reportable.

## 2. Experiments
| ID | Question | Systems | Data | Phase |
|---|---|---|---|---|
| E0 | Pipeline sanity | dense_oneshot | dev subset | Week 10 |
| E1 | VQ1 main comparison, equal budget B ∈ {2,3,4} | all four | dev (FYP-I), test (FYP-II) | Week 16 / FYP-II |
| E2 | VQ1 natural policy | all four | dev, test | FYP-II |
| E3 | VQ2 ablations (one component off at a time) | controller_on variants | dev + test | FYP-II |
| E4 | Sensitivity of weights/thresholds | controller_on | dev | FYP-II (calibration) |
| E5 | VQ3 generalisation | all four | HotpotQA sample | FYP-II |
| E6 | VQ3 representative tasks | controller_on vs fixed_iteration | curated benchmark | FYP-II |
Ablation variants (E3): `planner` off (single gap = goal, no TRANSFORM), `cache` off,
`state_aware_rerank` off, `gap_tracking` off, `redundancy_signal` off, `budget_awareness`
off, `adaptive_actions` off. Each is one experiment config overriding one `ablation.*` key.

## 3. Answer generation (`eval/generator.py`)
Input: goal + bundle evidence (in bundle order), each as `[i] (title, date) text`,
truncated to `eval.generator_max_context_tokens` total. Prompt instructs a short answer and
`Insufficient information.` if the evidence does not support one. Prompt template
`eval/prompts/answer.j2`, hash recorded in the manifest.

## 4. Metrics (`eval/metrics.py`)
Gold groups G = list of chunk-id groups (DATA §4); a group is *hit* if any member is in
the considered set. Tasks with no gold groups (null queries) are excluded from retrieval
and evidence metrics.

**Retrieval quality** (on the final bundle, ordered by score)
- Hit@k = 1 if ≥ 1 group hit in top k; Recall@k = hit groups in top k / |G|;
  MRR = 1 / rank of first hit (0 if none). k ∈ `eval.recall_ks`.
- NDCG@k (binary gain): a rank position gains 1 if it holds a member of a gold group not
  already hit at a higher rank; DCG@k = Σ gain_i / log2(i + 1) over ranks i = 1..k;
  NDCG@k = DCG@k / IDCG@k, where IDCG@k places min(|G|, k) gains at the top ranks.
- Reranking effect: the same metrics on the pre-rerank candidate order of each step vs
  post-rerank (controller_on and baselines).

**Evidence quality**
- Complete-evidence recall (CER) = mean over tasks of 1[all groups hit in bundle].
- Partial coverage = mean(hit groups / |G|) in bundle.
- Trajectory coverage = same, over everything retrieved during the task (not just the bundle).
- Missing-critical-evidence rate = 1 − CER.
- Repeated-document rate = returned candidates already seen in the task / all returned.
- Redundant-evidence rate = accepted items whose max cosine to an earlier accepted item
  ≥ `eval.redundancy_cos` / accepted items.
- Evidence gain per retrieval = mean evidence_gain over candidate-producing steps.

**Task performance**
- MultiHop-RAG: accuracy per the benchmark's own evaluation (`qa_evaluate.py`; the exact
  definition is recorded in ADR-003); null queries are scored against their gold answer
  in the same way. Also report per question type.
- HotpotQA: EM and token F1 (standard normalisation).
- Curated: evidence metrics; answers judged by rubric by two team members.

**Efficiency and trajectory**
- Retrieval calls / task; retrievals per correct answer; cache-hit rate = REUSE steps /
  candidate-producing steps; LLM calls, tokens and estimated cost / task; end-to-end
  latency (median, p90); zero-gain retrieval rate.
- Premature stop = stopped with reason ALL_GAPS_RESOLVED, LOW_MARGINAL_UTILITY or
  STAGNATION while CER = 0 for that task and trajectory coverage < 1.
- Over-retrieval = retrieval calls made after the step at which all groups were first hit.
- Stagnation episodes = count of steps with stagnation_count ≥ patience.

**Implementation status (v0, 2026-10-05):** `eval/metrics.py` computes Hit/Recall/NDCG@k,
MRR, CER, partial and trajectory coverage and the efficiency counts from traces, bundles
and the gold map. **Update 2026-10-08:** answers and answer accuracy are produced post hoc
by `python -m eval.generator --run <id>` (`b<B>/answers.csv`, `answers.json`: the
official ADR-003 rule as `accuracy` plus normalised `exact_match`; prompt
`eval/prompts/answer.j2`, its hash in `answers.json`); `eval/stats.py` implements §5 and
`eval/report.py` §7 (tables, stop reasons, action sequences, `quality_vs_calls.png`,
`stop_reasons.png`, and `comparison.md` for several runs). Still to add: redundancy rates,
over-retrieval and premature-stop rates. **Latency caveat:** with
`eval.max_workers` > 1 the workers share one Embedder (which serialises model calls), so a
task's latency includes waiting for the others — report latency from a single-worker run
(`--set eval.max_workers=1`, non-reportable) or from `scripts/latency_profile.py`, not from
a parallel quality run.

## 5. Statistics (`eval/stats.py`)
- Differences between two systems on per-task metrics: paired bootstrap
  (`eval.bootstrap_resamples`, seed `project.seed`), 95 % CI of the mean difference.
- Binary per-task outcomes (correct, CER): exact McNemar test.
- Latency: medians with bootstrap CIs.
- Report every configuration tested in E4, not only the best.

## 6. Repeats and seeds
- Deterministic components are run once. For LLM-dependent systems, final test results
  use `eval.repeats` (default 3) runs with `llm.cache_namespace = repeat-<n>` so each
  repeat queries the provider freshly; report mean ± std across repeats and the CI above
  on the pooled per-task means.
- Seeds fixed in config: `project.seed`, `data.split_seed`.

## 7. Reporting (`eval/report.py`, `python -m eval.report --run ...`)
Outputs to `eval/runs/<run_id>/report/`: `metrics.json`, `per_task.csv`, markdown tables,
and figures: quality vs retrieval calls (per B), ablation bar chart, stop-reason
distribution, action-sequence histogram, per-question-type accuracy.
Mapping to validation questions: E1+E2 → VQ1; E3+E4 → VQ2; E5+E6 → VQ3.

## 8. Reproducibility checklist (before a number enters the report)
- [ ] Manifest complete: git commit (clean tree), config hash, resolved config, library
      versions, prompt hashes, collection name, split hash, seeds.
- [ ] Same collection and split hash across all compared runs.
- [ ] controller_on config matches the frozen calibration ADR.
- [ ] Leakage test passed on that commit.
- [ ] Repeats done where required.
- [ ] Test split used only for the frozen configuration.

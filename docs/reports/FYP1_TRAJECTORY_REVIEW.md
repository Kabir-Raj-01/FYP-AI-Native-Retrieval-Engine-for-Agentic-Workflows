# FYP-I trajectory review: controller-ON failure categories

> Status: Draft · Owner: All · Last updated: 2026-10-16 · Related: ACTION_POLICY.md §9, EVIDENCE_MANAGER_SPEC.md §3, PLANNER_SPEC.md, ADR-013, ADR-014

TASKS.md Weeks 14–15: "read ≥ 20 trajectories by hand with `show_trace.py`; log failure
categories". Input to the FYP-II refinement items (evidence manager, planner guardrails,
stopping policy) and to the ACTION_POLICY §9 calibration.

## Source
Run `20261008-155217_controller_on_dev_903ff1c` (commit `903ff1c`, ADR-013 values:
uniform controller weights, `controller.min_utility` 0.05, `evidence.relevance_threshold`
0.2, `evidence.gap_coverage_threshold` 0.6), first 40 dev-subset tasks, B = 4,
qwen2.5:3b planner. Not reportable (`--max-tasks`, `--set`); dev data only.

21 trajectories read by hand (`python scripts\show_trace.py <run_id> <task_id>`):
multihop:0013, 0014, 0053, 0061, 0071, 0077, 0089, 0098, 0115, 0136, 0161, 0168, 0182,
0184, 0214, 0231, 0242, 0265, 0290, 0304, 0367. The counts below cover all 40 tasks
(33 with gold, 89 gold groups, 107 queries) and come from the traces, bundles and gold
groups of that run.

## Where the gold evidence went
| Gold group (89) | n | share |
|---|---|---|
| in the final bundle | 38 | 43 % |
| retrieved, then not kept | 37 | 42 % |
| never retrieved | 14 | 16 % |

Of the 37 retrieved-not-kept groups, the best rejection reason was `LOW_RELEVANCE` for 30,
`STEP_CAP` for 5 and `LOW_NOVELTY` for 2. **Losing retrieved gold at acceptance is the
largest single loss**, larger than retrieval misses.

## Failure categories
| # | Category | Evidence | Owner / FYP-II item |
|---|---|---|---|
| F1 | **Relevance is scored against the planner's subgoal query, not the gap or the question.** Short keyword queries give the cross-encoder little to match, so gold passages fall under `evidence.relevance_threshold`. | 30 of 37 lost gold groups (`LOW_RELEVANCE`); e.g. 0214, 0231, 0265 accept nothing on whole steps | S: rerank/evidence refinement — score relevance against the gap description (or the goal) as well; calibrate the threshold on controlled traces (§9) |
| F2 | **Queries name the publisher, which is metadata only.** Chunk text never contains "TechCrunch", "The Verge", …; the source lives in chunk `metadata.source`. | 69 of 107 queries (64 %) contain a source name; 0115 (`'The Verge TechCrunch'`) and 0265 (`'The Verge Fortune'`) are *only* source names and retrieve nothing relevant | V: planner guardrail (strip or reject source-only queries); K: optional `source` payload filter in the adapter (RETRIEVAL_SPEC filters) |
| F3 | **False gap resolution.** Coverage = cosine(passage, gap description) ≥ `evidence.gap_coverage_threshold`; bge-base cosines are high for anything on the topic, so one search resolves several gaps, including gaps about other sources. Leads to an early `ALL_GAPS_RESOLVED`. | 18 of 34 resolving items (53 %) are not gold; 0136 and 0367 resolve all 3 gaps in one step, 0184/0242 resolve a TechCrunch and a The Verge gap from one query | S: calibrate `evidence.gap_coverage_threshold` on controlled traces (deferred by ADR-013); consider the verifier (`llm.verifier`) for the band below the threshold |
| F4 | **Same query re-run in another mode.** Step 2 is usually DENSE with step 1's exact query; most candidates are duplicates and the gain is near 0, then a min-utility STOP ends the gap. | 16 consecutive identical queries in 40 tasks; 0013, 0061, 0077, 0115, 0168, 0214, 0265 | S: controller — prefer TRANSFORM (needs a planner that rewrites well, see F6) or SPARSE↔DENSE only when the first mode returned little |
| F5 | **Planner over-splits.** A single comparison question becomes near-identical gaps ("Find / Check / Verify if …"); or a gap that only says "articles by X and Y". | 0367 (3 identical gaps), 0098 g2 ("identify if the articles agree") | V: planner validator — reject near-duplicate gap descriptions (rapidfuzz, as for queries) |
| F6 | **Planner output rejected.** INIT rejected 9 times (`UNDECLARED_ENTITY` ×4, `COUNT` ×3, `SCHEMA` ×2); 3 tasks fell back to the single-gap plan. REWRITE proposals often repeat an earlier query (ADR-013). | 0182, 0230, 0304 (fallback); 0038, 0184, 0290 (retry accepted) | V: planner failure analysis; ADR-004 reversal check with a stronger model |
| F7 | **Null queries spend budget.** Unanswerable questions run 2–4 retrieval calls before stopping with an empty bundle. The empty bundle is correct; the calls are the cost. | 7 null tasks: 1×2, 4×3, 2×4 calls | S: stopping — stop a gap after a first search that accepts nothing *and* scores low relevance across the board |

## Observations worth keeping
- The planner's **fallback** (the whole question as one gap) did well where it happened:
  0304 reached CER 1.0 in one step and 0182 kept a gold passage. Together with F1/F2 this
  suggests that the question text is a stronger retrieval query than qwen2.5:3b's keyword
  subqueries. Worth an explicit FYP-II comparison (e.g. subquery + goal as the rerank
  query).
- Gap-level STOP (ACTION_POLICY §2 amendment, 2026-10-08) behaves as designed: a
  min-utility STOP closes the current gap and the loop moves to the next one (0013,
  0061, 0071).
- No invariant violations, crashes or REUSE loops in the 40 trajectories; REUSE appeared
  once (0161) and correctly returned 0 new candidates.

## Not a defect in code
F1–F7 are calibration, prompt or policy questions. The implementation matches the specs
(EVIDENCE_MANAGER_SPEC §3 coverage rule, ACTION_POLICY §2–§6); none of these is changed for
FYP-I because the keys involved are Draft and their calibration is the FYP-II task
(ACTION_POLICY §9, ADR-013 "Scope").

**Update 2026-10-16 (F1):** part of F1 was a mismatch, not a calibration choice: the 0.2
threshold was calibrated on question-vs-passage scores (ADR-013) and applied to
subquery-vs-passage scores. ADR-014 scores relevance against the better of the subquery and
the question (`rerank.relevance_query = max`). On the full dev subset it raised controller-ON
Recall@10 by +0.076 to +0.086 and CER by +0.068 to +0.079 at B = 2–4 (FYP1_REPORT_DRAFT §5.2).
The rest of F1 — gold under the threshold even against the question, and the per-step
acceptance cap — stays with the FYP-II evidence-manager calibration.

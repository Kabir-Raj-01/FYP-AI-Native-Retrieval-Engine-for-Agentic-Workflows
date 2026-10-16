# ADR-014: Score passage relevance against the subquery and the goal (`rerank.relevance_query`)

> Status: Accepted (2026-10-16, Kabir for the team; Samay review after the fact) · Date: 2026-10-16 · Author(s): Kabir · Reviewer: Samay
> Scope: dev subset only (EVAL_PROTOCOL §1.2). The key stays **Draft**; the FYP-II
> calibration ADR (ADR-006) may revisit it together with the thresholds it interacts with.

## Context
The reportable E1 controller-ON run (`20261010-163330_controller_on_dev_8b966ec`) scored
below the one-shot hybrid baseline on every evidence metric (B = 4: Recall@10 0.418 vs
0.619, CER 0.170 vs 0.445), although it retrieved more of the gold evidence (trajectory
coverage 0.755 vs 0.729). The trajectory review (`docs/reports/FYP1_TRAJECTORY_REVIEW.md`,
F1) traced the largest loss to acceptance: the cross-encoder scored passages against the
planner's short keyword subquery, so 30 of 37 retrieved-but-dropped gold groups fell
under `evidence.relevance_threshold`. That threshold (0.2, ADR-013 Decision 1) was
calibrated on **question**-vs-passage scores from `dense_oneshot`, but was applied to
**subquery**-vs-passage scores. fixed_iteration has the same mismatch (its hops rerank
against subgoal queries); the one-shot baselines do not (query = question).

## Decision
New key `rerank.relevance_query: subquery | goal | max`, default **`max`**:

- `subquery` — relevance = CE(pending query, passage) (the previous behaviour);
- `goal` — relevance = CE(task goal G, passage);
- `max` — relevance = max(CE(pending query, passage), CE(G, passage)) per passage.

With no goal, or goal = query (the one-shot paths and the HTTP direct endpoint), all three
are the pending query, so the one-shot baselines are unchanged. The key applies to both
multi-step systems so that E1 compares controllers, not relevance definitions:
controller_on reads G from `state.goal`; fixed_iteration passes it through the new
`goal=None` argument of `Engine.retrieve_direct`. Interface change:
`Reranker.rerank(query, candidates, state, goal: str | None = None)`. The cross-encoder
score cache (`rerank.score_cache_enabled`) keys by query text, so `max` costs one extra
CE pass per step only for pairs not yet scored. `rerank.done` events log the mode.

## Alternatives considered
- `goal` alone — measured below `max` (Evidence): a passage that answers one gap's
  subquery precisely can score low against a long multi-part question.
- Lower `evidence.relevance_threshold` instead — treats the symptom: the threshold would
  then mean something different for subquery and question scores. ADR-013's reversal
  condition (more than a third of questions with no accepted evidence) is not met
  (61 of 300 dev-subset tasks = 20 % at B = 2 in the reportable `subquery` run; 9 of the 40
  probe tasks = 23 % with `goal`), so the threshold stays at 0.2.
- Score against the gap description — gap descriptions are planner text with the same
  brevity problem as subqueries; left for the FYP-II evidence-manager work (Samay).
- Leave it for FYP-II — rejected: the mismatch is a defect in how an ADR-013 value is
  applied, not a calibration choice, and it decides the main FYP-I comparison.

## Evidence
Probe: first 40 dev-subset tasks (33 with gold, 89 gold groups), B = 4, commit `6182b67`
plus this change, `--set` (non-reportable). Controller-ON:

| relevance_query | R@10 | CER | gold groups kept | retrieved, not kept | calls | run |
|---|---|---|---|---|---|---|
| subquery | 0.467 | 0.152 | 38 (43 %) | 37 (28 LOW_RELEVANCE, 9 STEP_CAP) | 2.65 | `20261010-163330_controller_on_dev_8b966ec` (same tasks) |
| goal | 0.500 | 0.242 | 40 (45 %) | 32 (30 / 2) | 2.63 | `20261016-062842_controller_on_dev_6182b67` |
| **max** | **0.553** | **0.303** | **46 (52 %)** | 27 (18 / 9) | 2.55 | `20261016-064441_controller_on_dev_6182b67` |
| hybrid_oneshot (reference) | 0.609 | 0.364 | 60 (67 %) | — | 1 | `20261008-185336_hybrid_oneshot_dev_f086ac2` |

Selection rule, fixed before the probe results were read: the mode with the highest
controller-ON CER on the probe, Recall@10 as the tie-break, adopted only if it beats
`subquery`. The reportable effect is the full-subset E1 rerun (FYP1_REPORT_DRAFT §5.2).

## Consequences
- Changed in the same PR: `engine/config.py`, `configs/base.yaml`, CONFIG_SCHEMA,
  `engine/rerank/{interfaces,state_aware}.py`, `engine/core.py`,
  `baselines/fixed_iteration.py`, RERANK_SPEC §1–§2, BASELINES_SPEC (fixed_iteration),
  ARCHITECTURE §2/§6 + change log, `tests/unit/test_interfaces.py` (pinned signature).
- `LastResult.top_relevances` (RERANK_EXPAND ambiguity) now holds `max` scores.
- E1 runs made before this ADR (`…_8b966ec`) are the `subquery` setting and stay as the
  "before" row; E1 is rerun for both multi-step systems.

## Reversal condition
The full-subset E1 rerun shows `max` no better than `subquery` on CER and Recall@10, or the
FYP-II calibration (ADR-006) finds a better relevance target (e.g. gap descriptions).

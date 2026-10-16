# AI-Native Retrieval Engine for Agentic Workflows — FYP-I report (draft)

> Status: Draft · Owner: All · Last updated: 2026-10-16 · Related: PRD.md, ARCHITECTURE.md, EVAL_PROTOCOL.md, decisions/ (ADR-013, ADR-014)

FAST-NUCES Karachi, BS(CS), Fall 2026. Kabir Raj · Samay Kumar · Vishal Kothari.
Sources, per TASKS.md Week 16: methodology ← `docs/`, design ← ARCHITECTURE + ADRs,
results ← `eval/runs/`. Every number here names its run folder; all results are on the
**dev subset** (300 questions) and are **preliminary** (ADR-013, ADR-014): no test-split run
has been made, and the FYP-II calibration (ACTION_POLICY §9, ADR-006) has not happened.

## 1. Problem and objective
Retrieval-augmented agents usually retrieve once (one-shot RAG) or loop for a fixed
number of hops. Multi-hop questions need evidence from several documents; one-shot
retrieval misses parts of it, and fixed loops spend calls whether or not they help. The
project builds a **retrieval-control middleware**: a service between a host application
and a vector database that decides, step by step and under an explicit budget, which
retrieval action to take next, which candidates become evidence, and when to stop. It
returns an **EvidenceBundle** (evidence, open/resolved gaps, trace summary, stop reason);
the host generates the answer.

Requirements: SRS FR1–FR12 and NFRs, summarised in `PRD.md`. Scope exclusions (flagged,
not built): cross-task caching, learned policies, extra backends (PRD "Out of scope").

## 2. Methodology
- **Two paths.** *Controller-OFF* (`retrieve_direct`): one search (+ rerank), no state —
  the baselines and the API's direct endpoint. *Controller-ON*: a stateful loop over the
  task state S_t (STATE_SPEC): plan → select action → cache/REUSE gate → retrieve →
  rerank → update evidence → monitor → continue / replan / stop (ARCHITECTURE §4).
- **Actions** (ACTION_POLICY §2–§5): DENSE, SPARSE_HYBRID, RERANK_EXPAND, REUSE (task-local
  cache), TRANSFORM (planner REWRITE/SPLIT), STOP. The policy is deterministic code:
  preconditions, then `U = α·gain − β·cost − γ·redundancy − δ·latency`, argmax with a
  fixed tie order, and a minimum-utility stop per gap. No LLM chooses actions.
- **Planner** (PLANNER_SPEC): an LLM decomposes the goal into 1–4 gaps (subgoals) with
  queries; output is schema-validated and grounded (no undeclared entities), with retries
  and a deterministic fallback (the goal as a single gap).
- **Evidence** (EVIDENCE_MANAGER_SPEC): candidates are accepted on relevance, novelty and
  duplication rules (relevance = cross-encoder score against the better of the step's
  subquery and the question, `rerank.relevance_query = max`, ADR-014); a gap is resolved by the accepted item with the highest embedding
  coverage of the gap description above a threshold; evidence gain feeds the controller's
  per-action moving average.
- **Stopping** (ACTION_POLICY §6): all gaps resolved, budget exhausted, stagnation, or low
  marginal utility on every open gap.
- **Baselines** (BASELINES_SPEC): `dense_oneshot`, `hybrid_oneshot` (BM25 + dense, RRF),
  `fixed_iteration` (N = B planner-driven hops, no adaptive choice).
- **Evaluation** (EVAL_PROTOCOL): MultiHop-RAG (609 news articles, 2,556 queries; ADR-003),
  split 1,789 dev / 767 test, dev subset 300. Metrics: Recall/Hit/NDCG@k, MRR, complete
  evidence recall (CER, all gold groups in the bundle), CER@10, partial and trajectory
  coverage, premature stop, over-retrieval, repeated-document rate, evidence gain per
  retrieval; efficiency = retrieval calls, steps, latency. Equal-budget comparisons
  (B ∈ {2, 3, 4}); paired bootstrap CIs and exact McNemar tests (EVAL_PROTOCOL §5).
  Every run writes a manifest; a run is *reportable* only from a clean tree with no
  overrides and a complete pipeline.

## 3. Design (ARCHITECTURE, ADRs)
- Python 3.11, FastAPI, LangGraph (one iteration per graph invoke; `Engine.run` loops),
  Qdrant 1.19.1 native Windows binary (dense HNSW + sparse BM25 named vectors), SQLite
  TaskStore as the single source of state (ADR-002, ADR-009), JSONL traces.
- Every component sits behind an interface and is chosen only in `engine/factory.py`
  from configuration; every threshold, weight, model name and k is a config key
  (`configs/base.yaml`, CONFIG_SCHEMA).
- Models (ADR-001, ADR-004, ADR-005): `BAAI/bge-base-en-v1.5` embeddings (pinned
  revision), Qdrant/bm25 sparse, `BAAI/bge-reranker-base` cross-encoder (CPU),
  `qwen2.5:3b` on a local Ollama for the planner and the answer generator (no API cost).
- Chunking: 384-token chunks, 4,441 chunks, gold mapping 100 % (DATA §4).
- Security: retrieved text is untrusted, wrapped in delimiters, never in the system role
  (SECURITY.md); the planner injection test is part of the suite.
- Deviations from the proposal are recorded in ADR-011 (Windows-only deployment, no
  LangGraph checkpointer, the gain estimate's terms, planner inputs, SQLite-only cache) —
  awaiting the supervisor's approval.

## 4. Implementation status (end of FYP-I)
| FR | Item | Status |
|---|---|---|
| FR1 | Ingestion, chunking, Qdrant index, dense/sparse/hybrid adapter | done |
| FR2 | Task state S_t, SQLite TaskStore (version check, WAL) | done |
| FR3 | Controller-OFF direct path + `/v1/retrieve/direct` (NFR-P1: median 104–142 ms) | done |
| FR4 | Planner INIT/REWRITE/SPLIT, validator, fallback | done |
| FR5 | Task-local cache + REUSE, cross-task leakage test | done |
| FR6 | Utility controller, preconditions, fixed-policy ablation | done |
| FR7 | State-aware reranking (prototype; weights Draft) | done |
| FR8 | Evidence manager, gap resolution, trajectory monitor | done (LLM verifier not built) |
| FR9 | Engine facade: create/step/run/result, timeouts, per-task lock | done |
| FR10 | Traces, events, manifests, eval harness, metrics, statistics, reports | done (redundant-evidence rate: FYP-II) |
| FR11 | HTTP API `/v1/tasks/*` + Python `RetrievalClient`; FYP-I demo `examples/demo_fyp1.py` | done (agent demo: FYP-II) |
| FR12 | MultiHop-RAG loader | done (HotpotQA, curated set: FYP-II) |

Quality gates (2026-10-16, `62bb0be`): 1,012 unit tests on fakes (including fault injection),
15 integration tests against a live Qdrant and 2 golden-trajectory e2e tests, all green;
`ruff`, `mypy --strict engine` and pre-commit clean.

## 5. Results (dev subset, preliminary)
### 5.1 E0 — one-shot baselines (reportable)
Clean tree, no overrides, reranker on, 300 questions (265 with gold, 35 null queries).
Runs: `20261008-184814_dense_oneshot_dev_f086ac2`, `20261008-185336_hybrid_oneshot_dev_f086ac2`.
Reproduced at `62bb0be` for E1 (`20261016-073418_dense_oneshot_dev_62bb0be`,
`20261016-073536_hybrid_oneshot_dev_62bb0be`): dense identical on every task; hybrid differs
on 1 of 300 tasks (multihop:1888, a different 20th hybrid candidate), R@10 0.620, CER@10 0.317.

| System | R@5 | R@10 | Hit@10 | NDCG@10 | MRR | CER@10 | CER (bundle 20) | Coverage |
|---|---|---|---|---|---|---|---|---|
| dense_oneshot | 0.459 | 0.565 | 0.917 | 0.487 | 0.633 | 0.226 | 0.328 | 0.661 |
| hybrid_oneshot | 0.497 | 0.619 | 0.917 | 0.522 | 0.643 | 0.313 | 0.445 | 0.729 |

Hybrid − dense, paired over 265 tasks (95 % bootstrap CI; McNemar exact on CER):
Recall@10 +0.053 [+0.029, +0.078]; CER@10 +0.087 [+0.049, +0.128], p < 0.0001;
CER +0.117 [+0.068, +0.166], p < 0.0001; partial coverage +0.068 [+0.041, +0.095].
Hybrid is significantly better on every evidence metric; the gain is largest on
comparison and inference questions:

| Question type (tasks) | dense R@10 / CER | hybrid R@10 / CER |
|---|---|---|
| comparison (101) | 0.683 / 0.475 | 0.752 / 0.653 |
| inference (96) | 0.470 / 0.125 | 0.543 / 0.240 |
| temporal (68) | 0.525 / 0.397 | 0.527 / 0.426 |

Reranking (ADR-005, paired against the same systems without a reranker): Recall@10
dense +0.089 [+0.059, +0.118], hybrid +0.030 [+0.001, +0.058].

Latency is not reported from these runs: four workers share the CPU, and the dense run
re-used cached reranker scores (EVAL_PROTOCOL §4 latency caveat). Search-only latency
(NFR-P1): median 104 ms (ADR-012) / 142 ms (integration test, 24 queries, 2026-10-09).

### 5.2 E1 — equal budget B ∈ {2, 3, 4} (preliminary)
Commit `62bb0be` (ADR-014: relevance against the better of subquery and question), clean
tree, no overrides, reranker on, 300 questions (265 with gold). Reportable runs:
`20261016-073418_dense_oneshot_dev_62bb0be`, `20261016-073536_hybrid_oneshot_dev_62bb0be`,
`20261016-073716_controller_on_dev_62bb0be`. **fixed_iteration:**
`20261016-112205_fixed_iteration_dev_62bb0be` is still running (its B = 3 and B = 4 hops
need fresh planner REWRITE calls on CPU); its rows are added when it finishes. The
`subquery` rows are the earlier reportable run `20261010-163330_controller_on_dev_8b966ec`
(same code apart from ADR-014).

| System | B | R@10 | Hit@10 | NDCG@10 | MRR | CER@10 | CER | Partial cov. | Traj. cov. | Calls | Premature stop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_oneshot | 1 (one-shot) | 0.565 | 0.917 | 0.487 | 0.633 | 0.226 | 0.328 | 0.661 | 0.661 | 1.00 | – |
| hybrid_oneshot | 1 (one-shot) | 0.620 | 0.917 | 0.522 | 0.643 | 0.317 | 0.445 | 0.729 | 0.729 | 1.00 | – |
| controller_on | 2 | 0.431 | 0.732 | 0.400 | 0.551 | 0.196 | 0.196 | 0.431 | 0.631 | 1.75 | 0.226 |
| controller_on | 3 | 0.473 | 0.774 | 0.431 | 0.569 | 0.208 | 0.215 | 0.476 | 0.698 | 2.21 | 0.347 |
| controller_on | 4 | 0.494 | 0.792 | 0.447 | 0.586 | 0.234 | 0.249 | 0.504 | 0.744 | 2.52 | 0.419 |
| controller_on, `subquery` (before ADR-014) | 2 | 0.345 | 0.630 | 0.321 | 0.457 | 0.121 | 0.121 | 0.345 | 0.631 | 1.78 | 0.204 |
| controller_on, `subquery` (before ADR-014) | 3 | 0.390 | 0.683 | 0.354 | 0.488 | 0.143 | 0.147 | 0.393 | 0.706 | 2.31 | 0.317 |
| controller_on, `subquery` (before ADR-014) | 4 | 0.418 | 0.713 | 0.378 | 0.512 | 0.162 | 0.170 | 0.427 | 0.755 | 2.65 | 0.389 |

Paired differences over the 265 gold tasks (95 % bootstrap CI, 10,000 resamples, seed 42;
exact McNemar on CER):

| Difference (paired, 265 gold tasks) | Recall@10 | CER | Partial coverage |
|---|---|---|---|
| controller_on − hybrid_oneshot, B = 2 | -0.189 [-0.226, -0.152] | -0.249 [-0.306, -0.192]; McNemar p < 0.0001 | -0.298 [-0.336, -0.260] |
| controller_on − hybrid_oneshot, B = 3 | -0.147 [-0.184, -0.110] | -0.230 [-0.291, -0.174]; McNemar p < 0.0001 | -0.253 [-0.291, -0.215] |
| controller_on − hybrid_oneshot, B = 4 | -0.126 [-0.163, -0.090] | -0.196 [-0.253, -0.140]; McNemar p < 0.0001 | -0.225 [-0.263, -0.186] |
| controller_on `max` − `subquery` (ADR-014), B = 2 | +0.086 [+0.059, +0.114] | +0.075 [+0.042, +0.113]; McNemar p < 0.0001 | +0.086 [+0.059, +0.114] |
| controller_on `max` − `subquery` (ADR-014), B = 3 | +0.083 [+0.057, +0.110] | +0.068 [+0.034, +0.106]; McNemar p = 0.0003 | +0.083 [+0.057, +0.111] |
| controller_on `max` − `subquery` (ADR-014), B = 4 | +0.076 [+0.050, +0.103] | +0.079 [+0.045, +0.117]; McNemar p < 0.0001 | +0.077 [+0.052, +0.104] |

Where the gold evidence went (gold groups, all 300 questions; reasons are the rejection
closest to acceptance):

| System | Gold groups kept | Retrieved, not kept (reason) | Never retrieved | Mean bundle | Empty bundles (of 300) |
|---|---|---|---|---|---|
| hybrid_oneshot (one-shot) | 517 (70%) | 0 (0%; –) | 218 (30%) | 20.0 | 0 |
| controller_on, B = 4 | 345 (47%) | 191 (26%; LOW_RELEVANCE 146, STEP_CAP 35, LOW_NOVELTY 10) | 199 (27%) | 5.9 | 38 |
| controller_on, `subquery`, B = 4 | 292 (40%) | 253 (34%; LOW_RELEVANCE 216, STEP_CAP 28, LOW_NOVELTY 9) | 190 (26%) | 5.4 | 42 |

**Reading.**
1. ADR-014 removes the relevance mismatch: controller-ON improves at every budget
   (Recall@10 +0.076 to +0.086, CER +0.068 to +0.079, McNemar p ≤ 0.0003) and uses slightly
   fewer retrieval calls (2.52 vs 2.65 at B = 4).
2. Controller-ON stays significantly below one-shot hybrid at every budget (B = 4: Recall@10
   −0.126 [−0.163, −0.090], CER −0.196 [−0.253, −0.140]). **VQ1 is not supported at FYP-I.**
3. The loss is at acceptance, not retrieval: by B = 4 controller-ON has retrieved about as
   much gold as hybrid puts in its bundle (trajectory coverage 0.744 vs 0.729) but keeps 47 %
   of gold groups against 70 %. 146 retrieved groups fall under `evidence.relevance_threshold`
   even against the question, and 35 hit `evidence.max_accept_per_step`; bundles average 5.9
   items against the one-shot systems' fixed 20 (§7). Both keys are Draft and belong to the
   FYP-II evidence-manager calibration (§8).
4. By question type (B = 4, Recall@10 / CER): comparison 0.611 / 0.366, temporal
   0.456 / 0.265, inference 0.398 / 0.115 — inference questions remain the weakest, as in E0.

How ADR-014's setting was chosen (first 40 dev-subset tasks, B = 4, `--set`, non-reportable;
rule fixed beforehand: highest controller-ON CER, Recall@10 as tie-break):

| System, `relevance_query` | R@10 | CER | Run |
|---|---|---|---|
| controller_on, subquery | 0.467 | 0.152 | `20261010-163330_controller_on_dev_8b966ec` (same 40 tasks) |
| controller_on, goal | 0.500 | 0.242 | `20261016-062842_controller_on_dev_6182b67` |
| controller_on, **max** | **0.553** | **0.303** | `20261016-064441_controller_on_dev_6182b67` |
| fixed_iteration, subquery | 0.533 | 0.364 | `20261016-064545_fixed_iteration_dev_6182b67` |
| fixed_iteration, **max** | **0.553** | **0.455** | `20261016-071127_fixed_iteration_dev_62bb0be` |
| hybrid_oneshot (reference) | 0.609 | 0.364 | `20261008-185336_hybrid_oneshot_dev_f086ac2` |

## 6. Trajectory analysis
`docs/reports/FYP1_TRAJECTORY_REVIEW.md`: 21 controller-ON trajectories read by hand,
counts over 40 tasks. Main findings:
1. 42 % of gold groups were retrieved but not kept, 30 of 37 of them for low reranker
   relevance against the planner's short subgoal query.
2. 64 % of planner queries name the publisher, which exists only as chunk metadata.
3. 53 % of gap resolutions were made by a non-gold passage (embedding coverage threshold
   too permissive), producing early `ALL_GAPS_RESOLVED` stops.
4. The second search on a gap often repeats the first query in another mode with ~0 gain.
These are calibration/prompt issues, not code defects; they define the FYP-II work.
**Update 2026-10-16:** F1 was partly a mismatch rather than a calibration question — the
0.2 threshold was calibrated on question-vs-passage scores (ADR-013) but applied to
subquery-vs-passage scores. ADR-014 scores relevance against the better of the two; its
full-subset effect is in §5.2.

## 7. Limitations
- Dev subset only; one run per configuration; preliminary calibration (ADR-013) on a
  40-task sample; no test-split numbers.
- CPU-only reranking and a 3B local LLM: planner calls take 15–50 s, so latency NFRs for
  the controlled path are not met on the team laptop; quality numbers are unaffected.
- The planner model is weak at REWRITE (often repeats a previous query), which in effect
  disables TRANSFORM under the calibrated weights (ADR-013 Decision 3).
- **Bundles are not the same size.** A one-shot bundle is always its top 20 reranked
  candidates (BASELINES_SPEC); a multi-step bundle holds only accepted items (relevance
  ≥ 0.2, at most `evidence.max_accept_per_step` = 5 per step, novelty and duplicate rules).
  CER and partial coverage therefore reward the one-shot systems' larger bundles; CER@10 and
  Recall@10 are less sensitive to this but not immune (§5.2 gives bundle sizes).
- Latency is not compared: runs share caches (LLM, reranker scores) from earlier runs, the
  controller-ON E1 run was paused for ~111 min by Windows Modern Standby (one task's latency
  and the run's wall time include the pause; no metric depends on it), and the CPU is shared.
- Answer accuracy/EM are not reported: the qwen2.5:3b generator needs ~1–2 min per answer on
  CPU (ADR-004), too slow for 300 questions × 8 E1 sub-runs in FYP-I.
- One small reproducibility gap: re-running hybrid_oneshot changed one of 300 bundles
  (§5.1); dense_oneshot reproduced exactly.

## 8. FYP-II plan
| Month | Work | Owner |
|---|---|---|
| Jan 2027 | Evidence manager: acceptance vs bundle size (per-step cap, threshold under ADR-014's relevance), relevance against gap descriptions, coverage threshold, verifier band (F1, F3); stopping for unanswerable questions (F7) | Samay |
| Jan 2027 | Planner guardrails: no source-only queries, no near-duplicate gaps; failure analysis; stronger-model check (F2, F5, F6) | Vishal |
| Jan 2027 | Cache/REUSE experiments; optional `source` metadata filter; backend robustness | Kabir |
| Feb 2027 | ACTION_POLICY §9 calibration on the full dev split; ADR-006; freeze keys (`v0.8-frozen-config`) | Samay + Vishal |
| Mar 2027 | E1/E2 on the test split, repeats = 3; E3 ablations; statistics | Vishal + Samay |
| Mar–Apr 2027 | HotpotQA (E5) and curated benchmark (E6) | Kabir + Vishal / All |
| Apr–May 2027 | LangGraph agent demo / MCP; wheel + Windows deployment docs | Vishal / Kabir |
| May–Jun 2027 | Code freeze, reproducibility package, thesis, viva | All |

## 9. Acknowledgement of tool use
Per CONTRIBUTING.md ("Working with Claude Code") and FAST's policy on AI assistance: parts
of the code, tests and documentation were written with Claude Code (Anthropic) under the
team's review — plan first, tests with every change, real command output checked, every
line explainable by its author. Check the department's current policy for the exact wording
required in the submitted report.

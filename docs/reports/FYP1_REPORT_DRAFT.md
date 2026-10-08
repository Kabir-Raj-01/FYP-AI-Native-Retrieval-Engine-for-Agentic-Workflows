# AI-Native Retrieval Engine for Agentic Workflows — FYP-I report (draft)

> Status: Draft · Owner: All · Last updated: 2026-10-09 · Related: PRD.md, ARCHITECTURE.md, EVAL_PROTOCOL.md, decisions/

FAST-NUCES Karachi, BS(CS), Fall 2026. Kabir Raj · Samay Kumar · Vishal Kothari.
Sources, per TASKS.md Week 16: methodology ← `docs/`, design ← ARCHITECTURE + ADRs,
results ← `eval/runs/`. Every number here names its run folder; all results are on the
**dev subset** (300 questions) and are **preliminary** (ADR-013): no test-split run has
been made, and the FYP-II calibration (ACTION_POLICY §9, ADR-006) has not happened.

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
  duplication rules; a gap is resolved by the accepted item with the highest embedding
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

Quality gates: ~1,000 unit tests on fakes plus integration (live Qdrant), e2e (golden
trajectories) and fault-injection tests; `ruff`, `mypy --strict engine`, pre-commit.

## 5. Results (dev subset, preliminary)
### 5.1 E0 — one-shot baselines (reportable)
Clean tree, no overrides, reranker on, 300 questions (265 with gold, 35 null queries).
Runs: `20261008-184814_dense_oneshot_dev_f086ac2`, `20261008-185336_hybrid_oneshot_dev_f086ac2`.

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
Pending: `controller_on_dev` and `fixed_iteration_dev` are running (started 2026-10-09 00:19,
one worker, local Ollama). Filled in when both manifests are written.

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

## 7. Limitations
- Dev subset only; one run per configuration; preliminary calibration (ADR-013) on a
  40-task sample; no test-split numbers.
- CPU-only reranking and a 3B local LLM: planner calls take 15–50 s, so latency NFRs for
  the controlled path are not met on the team laptop; quality numbers are unaffected.
- The planner model is weak at REWRITE (often repeats a previous query), which in effect
  disables TRANSFORM under the calibrated weights (ADR-013 Decision 3).

## 8. FYP-II plan
| Month | Work | Owner |
|---|---|---|
| Jan 2027 | Evidence manager: relevance against gap/goal, coverage threshold, verifier band (F1, F3); stopping for unanswerable questions (F7) | Samay |
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

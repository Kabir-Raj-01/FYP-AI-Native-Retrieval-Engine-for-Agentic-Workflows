# Tasks and Milestones

> Status: Living document · Owner: All · Last updated: 2026-10-09
> Mirrors the proposal's milestone plan (§4), Table II allocation and Fig. 2 Gantt chart.
> Week 1 = 17 Aug 2026; dates are approximate — align with the department calendar.
> Mark items `[x]` in the PR that completes them. Each item names its owner, FR and doc.
> If the team uses GitHub Issues, put the issue number after each item and keep this file
> as the milestone overview.

Legend: **K** = Kabir · **S** = Samay · **V** = Vishal · **All** = pair/mob work

## Week 6 (21–27 Sep) — Proposal defense ✅
- [x] Proposal, literature review, requirements, methodology, architecture, plan

## Week 7 (28 Sep–4 Oct) — Foundations and contracts
- [ ] All: read all docs; resolve open questions in PR comments
- [x] K: repo skeleton, pyproject, Qdrant Windows binary setup, CI, pre-commit (SETUP, TESTING §6)
      — skeleton, pyproject fixes and pre-commit done (branch `k/foundations-week7`);
      Qdrant v1.19.1 binary installed and pinned (branch `k/data-download`); CI `unit` job
      green on GitHub 2026-10-10 (run #1, `8b966ec`). The `integration` job runs only on PRs
      to `main`, which does not exist yet
- [x] K: ADR-001 embedding + chunking + sparse method (accept) · ADR-003 dataset source, licence, split (draft)
      — both Accepted 2026-10-08 (Kabir for the team); ADR-003 also fixes EM next to accuracy
- [x] S: `engine/state/models.py` exactly per STATE_SPEC; invariant tests skeleton
      (also `serialization.py`, `invariants.py`; merge waits for ADR-009 acceptance)
- [x] V: `engine/config.py` EngineConfig + loader + config tests (CONFIG_SCHEMA)
- [x] V: test fakes + conftest fixtures (TESTING §2) (`engine_with_fakes` skips until the
      Engine exists; `planner_responses/*.json` belong with the planner, Week 11)
- [x] All: all `interfaces.py` files with docstrings, bodies `NotImplementedError`
- [x] S+V: review ADR-009 and mark it Accepted (its amendments are already applied to
      STATE_SPEC, ARCHITECTURE and API_CONTRACT) · review ADR-010
      — both Accepted 2026-10-08 by Kabir for the team; Samay/Vishal review after the fact
- [ ] All: show ADR-011 (deviations from the proposal) to the supervisor; mark Accepted once approved
- [x] Tag `v0.1-skeleton` — local tag at `0aac8b8` (tip of `k/foundations-week7`); push with the branch

## Weeks 8–9 (5–18 Oct) — Vertical slice
- [x] K: download + ingestion pipeline, UUIDv5 point ids, Qdrant collection (FR1, DATA §3, RETRIEVAL_SPEC)
      — `python -m data.download` (pinned, SHA-256 verified) and
      `python -m data.ingest --dataset multihop --index`: 4,441 chunks (deterministic),
      collection `multihop_v1` built in 59 min, resumable (branch `k/data-download`)
- [x] K: gold map + `check_gold_mapping.py` ≥ 95 % (DATA §4); split.json committed (DATA §5)
      — 100 % (6,084 / 6,084); split 1,789 / 767, subset 300 (branch `k/data-download`)
- [x] K: QdrantAdapter dense + sparse; Embedder (RETRIEVAL_SPEC §1–5) — plus hybrid (RRF),
      BM25 SparseEncoder and IndexWriter (ADR-001); unit + integration tests
- [x] K: controller-OFF direct path — `Engine.retrieve_direct` + `/retrieve/direct` endpoint (FR3, ARCHITECTURE §6, API_CONTRACT)
      — with a minimal app shell (`/v1/health`); `rerank=true` → 422 only on an engine built
      without a Reranker (bge-reranker-base wired since 2026-10-08); NFR-P1 median 104 ms;
      ADR-012 Accepted
- [x] V: FastAPI app, `/health`, `/tasks` create/state (FR2, API_CONTRACT)
      — `/v1/tasks` routes built by K (`engine/api/routers/tasks.py`, branch `k/controlled-path`)
- [x] V: eval harness v0: dataset loader, dense_oneshot, metrics (Recall/Hit/MRR/CER), run folder + manifest (EVAL_PROTOCOL, OBSERVABILITY)
      — built by K: `python -m eval.run --config configs\experiments\dense_oneshot_dev.yaml`;
      reranks whenever the engine has a Reranker (ADR-005 model still Proposed)
- [x] S: SQLite TaskStore (WAL, version check) + LangGraph skeleton with stub nodes; `Engine.step/run` loop with budget cap (ARCHITECTURE §3–4)
      — built by K (branch `k/controlled-path`): TaskStore (`engine/state/store.py`:
      version-checked save, cache entries in the same transaction, foreign-task entries
      refused); graph nodes (`engine/graph/nodes.py`, real node logic per ARCHITECTURE §4-§5)
      and `Engine.create_task/step/run/result` (lock, timeout, FAILED path); tested on
      scripted fakes (`tests/fakes/controlled.py`) until the real components exist
- [x] V: trace writer (TraceStep) shared by engine and baselines (OBSERVABILITY)
      — built by K: `engine/monitor/trace.py` (JSONL traces + bundles, context stamping,
      step-order check, `summarize`); events.jsonl + manifest stay with the eval harness

## Week 10 (19–25 Oct) — Working prototype checkpoint
- [x] K+V: E0 — dense_oneshot results on the dev subset through the service
      — reportable run 2026-10-09 `20261008-184814_dense_oneshot_dev_f086ac2` (clean tree,
      bge reranker): Recall@10 0.565, Hit@10 0.917, MRR 0.633, CER@10 0.226, CER 0.328.
      `eval.run` calls the Engine in-process (it has no HTTP mode); the HTTP path is the
      same `retrieve_direct`, verified live by `examples/demo_fyp1.py` and the integration
      tests (NFR-P1 median 142 ms)
- [x] All: demo script + short progress slides
      — `examples/demo_fyp1.py` (run live 2026-10-09) + `docs/reports/FYP1_SLIDES.md`
- [x] Tag `v0.2-week10` — local tag at `f086ac2` (code of the reportable E0 runs)

## Weeks 11–12 (26 Oct–8 Nov) — Baselines, state, planner → Mid-term
- [x] K: hybrid mode (prefetch + RRF); hybrid_oneshot baseline (BASELINES_SPEC)
      — `python -m eval.run --config configs\experiments\hybrid_oneshot_dev.yaml`;
      shares `run_oneshot` with dense_oneshot; reranks like dense_oneshot
- [x] K: retrieval instrumentation (latency, calls) in traces
      — calls and latency per step in TraceStep (`budget_after`, `latency_ms`) for the
      one-shot baselines; `events.jsonl` lines carry `run_id`/`task_id`/`step` with fields
      under `data` (OBSERVABILITY §3); `retrieval.call.latency_ms` is search-only. The
      controlled loop's `execute` node records its own (S, with the graph)
- [x] V: LLMClient (providers, structured output, cache, caps) (LLM_CLIENT_SPEC); ADR-004 model choices
      — client built by K (`engine/llm/`); ADR-004 Accepted 2026-10-08: qwen2.5:3b on a local
      Ollama v0.40.1 (`tools\ollama`, SETUP §2), live-verified with `python -m engine.llm.smoke`
- [x] V: planner v1: INIT/REWRITE/SPLIT, validator, fallback, prompts (FR4, PLANNER_SPEC)
      — built by K (`engine/planner/`; injection test included); interpretations in
      PLANNER_SPEC §7a. Needs ADR-004 (model) for live runs
- [x] S: real `maybe_plan` node → gaps in state; explicit state updates per ownership matrix
      — built by K with the graph (`engine/graph/nodes.py`), now driven by the real planner
- [x] V: cross-encoder relevance-only Reranker (used by baselines); ADR-005
      — built by K (`engine/rerank/`, branch `k/controlled-path`) with the config default
      bge-reranker-base at `rerank.revision`; ADR-005 Accepted 2026-10-09 (Kabir for the
      team; Samay review after the fact). `DirectResult.rerank_latency_ms` splits rerank
      from search (ADR-012 Amendment A1, implemented in `engine/core.py` /
      `baselines/base.py`); E0 re-run with the reranker is the Week 10 reportable run.
- [x] All: early MultiHop-RAG results table (dense vs hybrid)
      — `docs/reports/FYP1_REPORT_DRAFT.md` §5.1: hybrid − dense Recall@10 +0.053
      [+0.029, +0.078], CER +0.117 (McNemar p < 0.0001), 265 tasks; ADR-005 Accepted
- [x] Tag `v0.3-midterm` — local tag at `5163f78`

## Week 13 (9–15 Nov) — Controller core
> Per the proposal Gantt chart, cache/controller work starts alongside Weeks 11–12; this
> week completes it so that Weeks 14–15 stay free for integration and testing.
- [x] V: fixed_iteration baseline (BASELINES_SPEC)
      — built by K (`baselines/fixed_iteration.py`); `eval.run` now runs all four systems
      (`controller_on` and `fixed_iteration` need ADR-004's planner model)
- [x] K: task-local cache v1 + REUSE + leakage test (FR5, CACHE_SPEC)
      — cache v1 + leakage test done (`engine/cache/task_cache.py`, shared
      `engine/state/sqlite.py`, `engine/text.py`); REUSE needs the controller and the
      `execute`/`score_and_update` nodes (Week 13, S)
- [x] S: controller v1: preconditions, signals, utility, tie-break, fixed-policy ablation (FR6, ACTION_POLICY §2–4, §7) + worked-example test
      — built by K (`engine/controller/policy.py`, `signals.py`; worked example exact);
      interpretations listed under ACTION_POLICY §6 "As built" for Samay's review
- [x] S: EvidenceManager: acceptance, gap resolution, evidence gain (FR8, EVIDENCE_MANAGER_SPEC)
      — built by K (`engine/evidence/manager.py`); the optional LLM verifier is not built
      (needs a shared output model via ADR + a verifier model in ADR-004)
- [x] S: TrajectoryMonitor: stagnation, gap advancement, stop checks (ACTION_POLICY §6)
      — built by K (`engine/monitor/trajectory.py`) with ADR-009 A6 decided as recommended.
      Risk: with bge-reranker-base, relevance on long MultiHop questions was 0.000–0.003
      (ADR-005 probe) vs `evidence.relevance_threshold` 0.5 — measure on subgoal queries
      once the planner runs; calibration (ACTION_POLICY §9) owns the fix
- [x] S: state-aware reranking prototype (FR7, RERANK_SPEC) — may run into FYP-II per Gantt
      — built by K with the Reranker (`engine/rerank/state_aware.py`; FR7 constructed case
      reproduced exactly); weights stay Draft until the calibration ADR
- [x] V: `/step`, `/run`, `/result`, `/trace` endpoints + RetrievalClient (FR11)
      — built by K (`engine/api/routers/tasks.py`, `client/`); server traces in
      `paths.runs_dir/server/` (OBSERVABILITY §4)

## Weeks 14–15 (16–29 Nov) — Integration and test buffer (no new features)
- [x] All: end-to-end controller_on runs on dev subset; fix defects
      — reportable run `20261010-163330_controller_on_dev_8b966ec` (300 tasks × B = 2, 3, 4,
      clean tree, no crashes); defects found on the way fixed in `3f6257d`, `903ff1c`, `3111f87`
- [x] All: read ≥ 20 trajectories by hand with `show_trace.py`; log failure categories
      — 21 read, 7 categories: `docs/reports/FYP1_TRAJECTORY_REVIEW.md`
- [x] V: golden-trajectory and fault-injection tests
      — built by K: `tests/e2e/test_golden.py` (+ `tests/fixtures/golden/mini_tasks.json`),
      `tests/unit/graph/test_faults.py`, OpenAPI snapshot `tests/fixtures/openapi.json`;
      `scripts/show_trace.py`. The golden test found a REUSE loop (fixed, ACTION_POLICY §6
      "As built"). Coverage: 98 % overall, ≥ 94 % per target package
- [x] K: latency profile (`python scripts\latency_profile.py`)
      — script done and `configs/latency_profile.yaml` generated (REUSE 3.8, DENSE 139,
      SPARSE_HYBRID 154, RERANK_EXPAND 168 ms median); re-run for TRANSFORM once the
      planner exists — done 2026-10-08: TRANSFORM measured 14.5 s (live planner REWRITE)
- [x] Fallback if controller-ON slips: baselines + modular component demos remain the FYP-I demo
      — not needed: controller-ON runs end to end (demo and E1)

## Week 16 (30 Nov–6 Dec) — FYP-I draft
- [x] V+All: E1 on dev subset (equal budget B ∈ {2,3,4}), preliminary; initial report/dashboard
      — all four systems reportable (2026-10-16/17, `62bb0be` + docs-only `161ee2b`), results and
      paired statistics in `docs/reports/FYP1_REPORT_DRAFT.md` §5.2. VQ1 not supported yet:
      controller-ON is below fixed_iteration and hybrid one-shot at every budget; the loss is at
      acceptance/stopping (ADR-014 fixed the relevance mismatch, +0.08 R@10 / CER)
- [x] All: FYP-I report draft (methodology ← docs/, design ← ARCHITECTURE + ADRs, results ← eval/runs)
      — `docs/reports/FYP1_REPORT_DRAFT.md` (team edits it into the submitted report)
- [x] Tag `v0.5-fyp1` — annotated, on the commit that records the E1 results (2026-10-17)

## FYP-I final evaluation / defense (Dec–early Jan)
- [ ] Live demo: same question controller-OFF vs ON, trace + stop reason; recorded backup video
      — demo verified and runbook in `docs/reports/FYP1_SLIDES.md`; since 2026-10-12 the demo
      ends with the host's answer from the bundle (`No[2][1]`, gold: No); recording the video
      is left
- [ ] Every member can explain every module
- [x] FYP-II plan slide — `docs/reports/FYP1_SLIDES.md` §10

---

## FYP-II — January 2027: refinement
- [ ] S: evidence manager + semantic novelty/redundancy refinement from FYP-I failure categories
- [ ] S: reranking and stopping-policy refinement
- [ ] V: planner guardrail refinement + planner failure analysis
- [ ] K: cache/reuse refinement + experiments; backend robustness

## February 2027: calibration and freeze
- [ ] S+V: calibration per ACTION_POLICY §9 on dev; sensitivity tables (E4)
- [ ] S: calibration ADR; mark keys Frozen; tag `v0.8-frozen-config`

## March 2027: full MultiHop-RAG evaluation
- [ ] V: E1 + E2 on test split, frozen config, repeats = 3; statistics (EVAL_PROTOCOL §5–6)
- [ ] S+V: E3 ablations; interpretation
- [ ] K: performance optimisation (batching, caching); reproducibility checks

## March–April 2027: generalisation
- [ ] K+V: HotpotQA ingestion (ADR for setting/sample) + E5
- [ ] All: curated benchmark 40–60 tasks, double-annotated (DATA §7) + E6

## April–May 2027: integration and packaging
- [ ] V: LangGraph agent demo (AGENT_INTEGRATION_SPEC); optional MCP adapter
- [ ] K: installable Python package (wheel) for the engine; Windows deployment + setup docs
- [ ] V: automated test hardening; API/client docs final

## May–June 2027: finalisation
- [ ] All: code freeze (≥ 4 weeks before final defense); tag `v1.0`
- [ ] K: reproducibility package tested on a teammate's clean machine
- [ ] All: thesis, presentation, scripted demo + backup video, viva preparation

## Stretch (only when the above is on schedule)
- [ ] Adaptive-RAG-style router baseline · learned policy/classifier · AgenticRAGTracer-compatible diagnostics · cross-task cache experiment · extra retrieval adapters

## Current status
| Date | Note |
|---|---|
| 2026-09-25 | Documentation set complete; Week 7 starting. |
| 2026-10-04 | Windows-only setup; audit against the proposal: owners per Table II, Week 13 controller core, ADR-010/011 added, ADR-003 partly filled, all experiment configs for E1–E3, pre-commit config. |
| 2026-10-05 | First code (K with Claude Code, branch `k/foundations-week7`): errors, EngineConfig + loader, state models/serialization/invariants, all interfaces + factory stub, fakes/fixtures, import-boundary test. 475 unit tests green, 100 % coverage of `engine/`. Open: ADR-009/010 acceptance, an ADR line for the checkpoint interpretations, LangGraph import blocked by Smart App Control on K's machine (LIBRARY_NOTES). |
| 2026-10-05 | LangGraph unblocked (uuid-utils 0.17.0, `requirements.lock`); nested GraphState verified (ADR-009 A7). Branch `k/data-download`: Qdrant v1.19.1 pinned; MultiHop-RAG downloaded and verified (609 docs, 2,556 queries; ADR-003 filled except split hashes). |
| 2026-10-05 | scikit-learn 1.9.0 / pandas 3.0.3 held (Smart App Control). Ingestion step 1 (chunks, gold map, split): 4,441 chunks, gold coverage 100 %, split.json + gold_map.json committed, ADR-003 split hashes filled. Next: Embedder + BM25 + Qdrant upsert + adapter (step 2). |
| 2026-10-05 | Index `multihop_v1` built (4,441 points, 59 min). Direct path + `/v1/retrieve/direct` + `/v1/health` (ADR-012). Preliminary dev-subset Recall@10: dense 0.477, BM25 0.608, hybrid 0.589 (ADR-001). NFR-P1 met (median 104 ms). Risk: CPU cross-encoder vs NFR-P3/P7 (ADR-005). |
| 2026-10-05 | Trace writer + eval harness v0 + dense_oneshot. Preliminary E0 (dev subset, no rerank): Recall@10 0.477, CER 0.177; trajectory coverage 0.661 shows the top-20 holds far more gold than the retrieval-order top-10 — evidence for adding the reranker (ADR-005). Cache + latency profile done. |
| 2026-10-05 | ADR-005 probe (dev subset, exploratory): bge-reranker-base lifts dense Recall@10 0.493 → 0.556 (80 tasks); MiniLM no reliable gain on dense and hurts hybrid (0.589 → 0.525). Rerank latency not yet valid (measured on battery); bge ≈ 10–24 s per 20 pairs. Decision stays with ADR-005's owner. |
| 2026-10-05 | hybrid_oneshot baseline. Preliminary dev-subset run (no rerank, non-reportable): Recall@10 0.589, Hit@10 0.898, MRR 0.604, CER 0.291, trajectory coverage 0.729 — vs dense_oneshot 0.477 / 0.823 / 0.491 / 0.177 / 0.661. |
| 2026-10-08 | ADR-001/003/004/009/010/012 Accepted (Kabir for the team; Samay/Vishal review after the fact); ADR-013 preliminary calibration (relevance threshold 0.2, `rerank.top_n_output` 20, controller defaults kept). qwen2.5:3b planner on local Ollama; first live controller-ON runs on 40 dev-subset tasks (B = 4, non-reportable): CER 0.152, R@10 0.467, trajectory coverage 0.854, 2.65 calls. Latency profile with TRANSFORM 14.5 s. FYP-I demo `examples/demo_fyp1.py`. `eval.run` now marks a run interrupted by Ctrl+C or any crash as aborted. Open: ADR-005 (Samay), ADR-011 (supervisor), CI on GitHub, reportable E0/E1 runs on a clean tree, hand-read trajectories. |
| 2026-10-16 | ADR-014: relevance scored against the better of subquery and question (`rerank.relevance_query = max`), both multi-step systems; fixes the subquery/question mismatch behind trajectory-review F1. Full dev subset, controller-ON: Recall@10 +0.08, CER +0.07–0.08 at every budget (p ≤ 0.0003), still below hybrid one-shot (B = 4: R@10 0.494 vs 0.620, CER 0.249 vs 0.445) — loss is at acceptance (bundle 5.9 vs 20 items). E1 one-shot + controller_on reportable at `62bb0be`; fixed_iteration running. Long runs paused/killed by Windows Modern Standby (SETUP §7). 1,012 unit / 15 integration / 2 e2e tests green. |
| 2026-10-17 | E1 complete: fixed_iteration reportable (`20261016-153022_fixed_iteration_dev_161ee2b`). B = 4, R@10 / CER: hybrid one-shot 0.620 / 0.445 (1 call), fixed_iteration 0.555 / 0.411 (4), controller-ON 0.494 / 0.249 (2.52). Controller-ON is significantly below fixed_iteration at every budget (CER −0.11 to −0.18) with 13–37 % fewer calls; fixed_iteration matches hybrid's CER at B = 3–4 (n.s.). Gap is at acceptance and stopping (bundles 5.9 vs 11.4 vs 20 items) → FYP-II evidence-manager/stopping calibration. Report §5.2, slides §7 updated; tag `v0.5-fyp1`. |
| 2026-10-17 | FYP-I requirements audit (proposal, SRS/PRD, docs) — gaps fixed: ADR-015 optional bearer token (SRS NFR-S3; was documented but not built); ADR-016 performance targets from the profiled runs (SRS TBD-4: P1/P2 met, P3/P7 CPU targets); live-LLM test layer `tests/llm/` (TESTING §1) and an NFR-R3 test. Coverage 97 %. Open: people items (ADR-011 supervisor, backup video, module walkthroughs, ADR reviews), the submitted report/deck, the PR/`main` workflow and pushing to GitHub. |

# Glossary

> Status: Stable · Owner: All · Last updated: 2026-09-28 · Amended by ADR-009

| Term | Meaning in this project |
|---|---|
| **Host application** | The external RAG or agent app that calls the engine. Owns final answer generation. |
| **Engine** | This project: the retrieval-control middleware (`engine/`). |
| **Controller-OFF / direct mode** | One-shot retrieval through the same RetrieverAdapter with no state, planning, cache or control. `POST /v1/retrieve/direct`. |
| **Controller-ON / controlled mode** | Stateful, iterative retrieval governed by the controller. `POST /v1/tasks/...`. |
| **Task** | One information need (e.g. one benchmark question), identified by `task_id`. All state and cache are scoped to it. |
| **S_t / RetrievalState** | Task state at step t: goal G, current subgoal g_t, evidence E_t, query history Q_t, retrieval history R_t, budget B_t (plus control bookkeeping). See STATE_SPEC.md. |
| **Step** | One controller iteration: (optional plan) → select one action → execute it → rerank/evidence (if candidates) → monitor → persist. Every action, including REUSE and TRANSFORM, consumes one step. |
| **Retrieval call** | One backend search request (DENSE, SPARSE_HYBRID, RERANK_EXPAND). REUSE, TRANSFORM and STOP make none. The equal-budget comparison caps retrieval calls. |
| **Action** | One of REUSE, DENSE, SPARSE_HYBRID, TRANSFORM, RERANK_EXPAND, STOP. |
| **Retrieval mode** | Backend search type: `dense`, `sparse`, `hybrid`. SPARSE_HYBRID executes `sparse` for lexical needs and `hybrid` for mixed needs. |
| **Need type** | Planner's classification of an information need: `semantic`, `lexical`, `mixed`. |
| **Gap / EvidenceGap** | A sub-question the task still needs answered. Created by the planner; resolved only by the EvidenceManager. Status: open, resolved, superseded, exhausted. |
| **Current gap / current subgoal (g_t)** | The open gap being worked on now. |
| **Pending query** | The concrete query text the next retrieval action will use for the current gap. |
| **Candidate** | A chunk returned by the backend (or cache) for a query, before acceptance. |
| **Evidence / EvidenceItem** | A candidate accepted into E_t by the EvidenceManager. |
| **Relevance** | Cross-encoder score of (pending query, chunk), mapped to [0,1]. |
| **Gap coverage** | Cosine similarity between a chunk and an open gap description, clipped to [0,1]. |
| **Novelty** | 1 − max cosine similarity of a chunk to any evidence already in E_t. |
| **Redundancy (rerank)** | Similarity of a candidate to candidates already selected higher in the same reranked list (MMR-style). Distinct from novelty, which is measured against E_t. |
| **Redundancy signal (controller)** | Predicted repetition of a proposed action relative to query/result history; code identifier `redundancy_signal`, not `R`, to avoid clashing with S_t's `R_t` (retrieval history) — `ActionScores.R` is still its field name in that one frozen model (ADR-009 "Also"). |
| **GraphState / IterationScratch** | LangGraph's per-run state: `RetrievalState` plus a `scratch` (`IterationScratch`) holding transient per-iteration data — cache hit, fresh/scored candidates, plan outcome, queued cache entries — discarded by `persist`, never saved (STATE_SPEC §4; ADR-009 §2). |
| **RunContext** | `{run_id, system}`, passed to `Engine` and the trace writer rather than stored in task state (ADR-009 §6). |
| **PlanLLMOutput** | The schema actually requested from the LLM for the planner role: `PlanResult` minus `mode` and minus the length/count constraints, which the planner validator enforces from config instead (STATE_SPEC §3; ADR-009 §10). |
| **Evidence gain** | Per-step scalar in [0,1] measuring progress: newly resolved gaps + quality of newly accepted evidence. |
| **Stagnation** | Consecutive retrieval steps with evidence gain below `monitor.stagnation_threshold`. |
| **Utility** | `α·G − β·C − γ·R − δ·L` for an eligible action; the controller picks the argmax. |
| **Budget** | Per-task caps: steps, retrieval calls, LLM calls, tokens. |
| **Evidence bundle** | Terminal output: accepted evidence, stop reason, trace summary. |
| **Trace / TraceStep** | Per-step structured log record, identical schema for every system. |
| **Run** | One execution of an experiment config over a dataset split; has a `run_id` and a folder under `eval/runs/`. |
| **Equal-budget experiment** | All systems capped at the same `max_retrieval_calls`. |
| **Natural-policy experiment** | Each system uses its own stopping behaviour within a generous cap. |
| **Dev split / test split** | Dev: everything used while building and calibrating. Test: touched once, with frozen config. |
| **Frozen** | A config value or doc that may change only via an ADR. |
| **ADR** | Architecture Decision Record in `docs/decisions/`. |
| **Complete-evidence recall** | Fraction of tasks where all gold evidence chunks are in the final bundle. |
| **Gold evidence** | Benchmark-annotated supporting evidence, mapped to chunk IDs (DATA.md). |

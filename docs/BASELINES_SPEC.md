# Baselines

> Status: Stable · Owner: Vishal · Last updated: 2026-10-05 · Related: EVAL_PROTOCOL.md

## 1. Shared conditions (all systems, including controller_on)
Same Qdrant collection, embedding model, sparse encoder, cross-encoder (relevance-only
for baselines), planner prompts/LLM (where a planner is used), answer generator and
generator prompt, dataset split and task list. Every system produces TraceSteps and an
EvidenceBundle of at most `rerank.top_n_output` items sorted by score — the generator
sees the same maximum amount of evidence from every system.

Implementation: `baselines/<name>.py`, each exposing
`run_task(task: Task, ctx) -> EvidenceBundle` and writing traces through the same logger
as the engine. Selected by `experiment.system` in the experiment config.

## 2. Systems
### dense_oneshot
1. `retrieve_direct(goal, mode=dense, k=retrieval.default_k, rerank=True)`.
2. Bundle = top `rerank.top_n_output` by relevance. One TraceStep (action DENSE).
   `stop_reason = ONE_SHOT_COMPLETE`. Retrieval calls = 1, LLM calls = 0.

Implementation: `baselines/dense_oneshot.py`, `run_task(task, ctx: BaselineContext)`
(`baselines/base.py`: the shared Engine, the run's TraceWriter, the config and the
sub-run's retrieval budget B). TraceStep: `rule="one_shot"`, `eligible_actions=[DENSE]`,
empty `utility_scores`, `accepted_ids` = the bundle, `rejected={}`, `evidence_gain=0.0`,
`budget_after` with `max_steps=1`, `max_retrieval_calls=B`, `max_llm_calls=0`. Bundle `gaps`
are empty (a one-shot baseline plans none); `summary` = PLANNER_SPEC §3 truncation.
**Without a reranker** (an Engine built without one, e.g. in tests): no rerank — the bundle is the top
`rerank.top_n_output` in retrieval order with `final_score` = the raw retrieval score and
`relevance = 0.0`; `eval/run.py` marks such runs non-reportable.

### hybrid_oneshot
Identical with `mode=hybrid` (action SPARSE_HYBRID in the trace).

### fixed_iteration (Agentic RAG with a fixed hop budget)
1. `Planner.plan(INIT)` on the goal — same planner, prompts and validation as the engine.
2. For hop i = 1..N where N = `budget.max_retrieval_calls` of the experiment:
   - query = subgoal i's query if i ≤ number of subgoals; otherwise a planner REWRITE of
     the last subgoal given accepted-evidence summaries (costs an LLM call; if the LLM
     budget is exhausted, reuse the last query with k increased by
     `retrieval.expand_step_k`).
   - retrieve with `fixed_iteration.mode` (default `hybrid`), k = `retrieval.default_k`.
   - rerank relevance-only; accept candidates with relevance ≥
     `evidence.relevance_threshold` that are not duplicates (no novelty/gap logic).
3. No cache, no utility, no early stopping: always exactly N retrieval calls.
   `stop_reason = FIXED_ITERATIONS_COMPLETE`.
4. Bundle = top `rerank.top_n_output` accepted items by relevance.
This isolates "more retrieval" from "controlled retrieval".

### controller_on
The full engine via `Engine.run` (in-process, same code path as the API).

### adaptive_rag_router (stretch)
A query-complexity classifier routes each question to: no retrieval, dense_oneshot, or
fixed_iteration. Only attempted if the core schedule is on track; must use the same
infrastructure. Classifier approach recorded in an ADR before implementation.

## 3. Budget alignment
| Experiment | dense/hybrid_oneshot | fixed_iteration | controller_on |
|---|---|---|---|
| Equal-budget | 1 call (reference) | N = B | `max_retrieval_calls = B` |
| Natural-policy | 1 call | N = B | `max_retrieval_calls = B_natural` (generous), stops on its own |
B values (e.g. 2, 3, 4) are listed in the experiment configs; results are reported for each.

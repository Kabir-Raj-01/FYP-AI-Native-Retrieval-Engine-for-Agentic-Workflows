# ADR-011: Recorded deviations from the approved proposal

> Status: Proposed (to be shown to the supervisor) · Date: 2026-10-04 · Author(s): All · Reviewer: Supervisor

## Context
The FYP proposal (18 September 2026) is the approved baseline. During design the team
made five decisions that differ from what the proposal states. This ADR lists them in
one place so the supervisor can approve them and the FYP-I report can cite them. Once
approved, set Status to Accepted.

## Decision
| # | Proposal says | The design now does | Reason | Details |
|---|---|---|---|---|
| 1 | Table III: "pytest, Docker, Git/GitHub"; §10b: "Docker/deployment and setup documentation" | Windows-only development and deployment: Qdrant runs as the native Windows binary `qdrant.exe`; CI uses Windows runners; FYP-II packaging is an installable Python package (wheel) plus a Windows setup guide. No Docker. | Team decision to build and run the whole project on Windows 10/11. | SETUP.md, TESTING §6, SDS §8.1, TASKS.md |
| 2 | Table III: LangGraph for "checkpointing" | LangGraph runs without a persistent checkpointer; the SQLite TaskStore is the single persisted copy of the state. | One source of truth for state; the API contract does not depend on LangGraph's internal checkpoint format. | ADR-002, ARCHITECTURE §3 |
| 3 | §5e: evidence gain estimated from "gap similarity, previous evidence gain, retriever/planner confidence and query type" | Gain = gap alignment + moving average of realised gain + need-type match; no confidence term. | Retrieval scores are not comparable across modes (RETRIEVAL_SPEC §3) and the planner produces no calibrated confidence, so a confidence term would be an uncalibrated signal. Revisit in FYP-II if dev traces show a need (new ADR, and the ACTION_POLICY §8 worked example must be updated). | ACTION_POLICY §4 |
| 4 | §5c: the planner "receives only the task goal, current subgoal, and accepted evidence summaries" | For REWRITE/SPLIT it also receives the current gap's previous queries and the other open gaps' descriptions. | Needed to avoid repeating failed queries and to split without overlapping existing gaps. Both are produced by the engine itself, so no new external facts enter the prompt; entity grounding is still enforced. | PLANNER_SPEC §3, §5 |
| 5 | Table III: intermediate cache = "Python in-memory cache + SQLite persistence" | Cache entries live in SQLite only; query embeddings are cached in memory (Embedder LRU). | Entries must be committed atomically with the state (ADR-009 §9); with at most `cache.max_entries_per_task` entries per task, SQLite lookups are cheap; an in-memory copy would need invalidation after failed steps. | CACHE_SPEC §4–6 |

Aligned with the proposal in the same change (not deviations): task ownership follows
Table II (Kabir owns the controller-OFF direct path; Vishal owns trajectory logging);
NDCG@k added to the retrieval metrics (§5i); Weeks 14–15 kept for integration and
testing only.

## Consequences
The FYP-I report's methodology and design chapters cite this ADR for each difference.
Any further deviation from the proposal is added here (or in a new ADR) before it is
implemented.

## Reversal condition
If the supervisor rejects an item, that item reverts to the proposal's wording, and the
affected docs and tests are updated in one PR.

# FYP-I progress slides and demo runbook

> Status: Draft · Owner: All · Last updated: 2026-10-09 · Related: FYP1_REPORT_DRAFT.md, FYP1_TRAJECTORY_REVIEW.md

Outline for the Week 10 progress slides and the FYP-I defense (TASKS.md). One slide per
heading; bullets are the slide text, *notes* are what the speaker says. Numbers come from
`FYP1_REPORT_DRAFT.md` §5 — update them there first.

## 1. Title
- AI-Native Retrieval Engine for Agentic Workflows
- Kabir Raj · Samay Kumar · Vishal Kothari — FAST-NUCES Karachi, FYP-I

## 2. Problem
- Multi-hop questions need evidence from several documents
- One-shot RAG misses parts; fixed loops spend calls whether or not they help
- *Notes:* MultiHop-RAG example — a question comparing two publishers' articles.

## 3. What we built
- Retrieval-control middleware between a host app and Qdrant (FastAPI + Python client)
- Controller-OFF: one search + rerank · Controller-ON: stateful, budgeted loop
- Output: EvidenceBundle (evidence, gaps, stop reason, trace) — the host writes the answer
- *Notes:* architecture diagram from ARCHITECTURE §1.

## 4. The controlled loop
- plan → select action → cache/REUSE → retrieve → rerank → evidence → monitor → stop?
- Actions: DENSE · SPARSE_HYBRID · RERANK_EXPAND · REUSE · TRANSFORM · STOP
- Policy = deterministic utility `U = α·gain − β·cost − γ·redundancy − δ·latency` — no LLM picks actions
- *Notes:* show one trace from `show_trace.py` (worked example).

## 5. Engineering
- Every component behind an interface; every threshold a config key; ADRs for every decision
- ~1,000 unit tests on fakes + live Qdrant integration, golden trajectories, fault injection
- Every run writes a manifest; only clean-tree runs are reportable

## 6. Results — E0 (dev subset, 265 questions with gold)
- Table: dense vs hybrid (R@10 0.565 vs 0.619; CER 0.328 vs 0.445; both p < 0.0001)
- Reranker (bge-reranker-base): dense R@10 +0.089, hybrid +0.030 (paired CIs)
- *Notes:* say "preliminary, dev subset, one run".

## 7. Results — E1 equal budget (preliminary)
- Table: controller-ON vs fixed_iteration vs one-shot at B = 2, 3, 4 (report §5.2)
- Calls used per task (controller-ON stops early) vs CER

## 8. What the trajectories taught us
- 42 % of gold was retrieved then rejected (relevance scored against short subqueries)
- 64 % of planner queries name a publisher (metadata only, never in the text)
- 53 % of gap resolutions came from non-gold passages (coverage threshold)
- → concrete FYP-II fixes, not guesses

## 9. Live demo
- Same question, controller-OFF vs controller-ON, trace + stop reason (runbook below)

## 10. FYP-II plan
| Month | Work |
|---|---|
| Jan | Evidence/relevance fixes (Samay) · planner guardrails (Vishal) · cache/REUSE experiments (Kabir) |
| Feb | Full calibration on the dev split, freeze config (ADR-006) |
| Mar | Test-split E1/E2 (×3), ablations E3, statistics |
| Mar–Apr | HotpotQA (E5), curated benchmark (E6) |
| Apr–May | LangGraph agent demo / MCP, installable package |
| May–Jun | Code freeze, reproducibility package, thesis, viva |

---

## Demo runbook (PowerShell, repo root, `.venv` active)
1. Start Qdrant (SETUP §3) and Ollama (SETUP §2 "Local LLM"); check
   `Invoke-RestMethod http://127.0.0.1:6333` and `http://127.0.0.1:11434/api/version`.
2. Start the API: `python -m uvicorn engine.api.app:app --host 127.0.0.1 --port 8000`.
3. In a second window: `$env:PYTHONIOENCODING = "utf-8"; python -m examples.demo_fyp1`.
   Default question: dev task multihop:0082 (Scorsese, Polygon vs The Independent).
   Verified 2026-10-09: controller-OFF returns 20 reranked candidates in ~0.3 s;
   controller-ON resolves both gaps in 2 steps (`ALL_GAPS_RESOLVED`), ~1 s with a warm
   LLM cache (the first run of a new question waits ~15–30 s for the planner).
   The demo ends as the host app: qwen2.5:3b answers from the top 8 bundle items with
   passage citations — verified 2026-10-12: `No[2][1]` (gold: No). The first answer takes
   ~1–2 min on CPU (3k prompt tokens), a repeat ~1 s while Ollama keeps the model loaded:
   **run the demo once before the audience arrives**, and start Ollama with
   `$env:OLLAMA_KEEP_ALIVE = "2h"` so the model is not unloaded after 5 idle minutes.
   `--no-answer` skips this step.
4. Optional: `python scripts\show_trace.py server <task_id>` for the full trajectory.
5. Backup: record the screen during a rehearsal (Win+Alt+R, Xbox Game Bar) and keep the
   video next to the slides in case Ollama or Qdrant fails on the day.

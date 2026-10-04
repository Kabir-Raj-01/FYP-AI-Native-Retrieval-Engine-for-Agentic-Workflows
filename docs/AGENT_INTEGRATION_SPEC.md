# External Agent Integration (FR11)

> Status: Draft (FYP-II) · Owner: Vishal · Last updated: 2026-10-04

## 1. Goal
Show that an open-source agent can use the engine as an external service with a small
adapter and **no engine logic inside the agent** — demonstrating the pluggable claim.

## 2. Demo agent (`examples/langgraph_agent/`)
- Separate LangGraph graph: `receive_question → gather_evidence → answer → END`.
- `gather_evidence` is a tool wrapping `RetrievalClient`:
  `create_task(goal=question, budget=...)` → `run(task_id)` → returns the bundle's
  evidence texts with ids and the stop reason.
- `answer` calls the agent's own LLM (its own config, not the engine's) with the evidence,
  wrapped per SECURITY §3, and cites evidence ids.
- CLI flag `--mode direct|controlled`: `direct` calls `retrieve_direct` instead — the same
  agent shows controller-OFF vs controller-ON behaviour live.
- Optional step mode: the agent calls `step()` repeatedly and prints each action — useful
  in the defense demo.

## 3. Constraints
- Imports only `client` and `client.models` (import-boundary test, TESTING §4.9).
- Runs against a server started with `python -m uvicorn engine.api.app:app --host 127.0.0.1 --port 8000`.
- `README.md` in the folder: setup, one command to run, expected output.

## 4. Acceptance
- Runs end-to-end on 5 fixture questions in both modes.
- Output shows actions taken, evidence ids, stop reason and final answer.
- Recorded as a short screen capture for the final defense backup.

## 5. Stretch: MCP adapter
`examples/mcp_server/` exposes one tool, `gather_evidence(goal, max_retrieval_calls)`,
implemented with `RetrievalClient`, so MCP-compatible agents can call the engine. Only
after the LangGraph demo is complete; check current MCP SDK docs before implementing.

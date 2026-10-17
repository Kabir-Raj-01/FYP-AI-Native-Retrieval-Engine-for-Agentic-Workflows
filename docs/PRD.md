# Product Requirements

> Status: Stable · Owner: All · Last updated: 2026-10-17 · Source: FYP Proposal (18 Sep 2026)
> Developer summary of the official SRS (`../SRS — AI-Native Retrieval Engine for Agentic
> Workflows.docx`). The SRS is authoritative: if the two disagree, fix this file.

## 1. Purpose
Build a standalone, pluggable retrieval-control engine that augments an existing RAG or
agent application — without replacing its retriever or generator — so that multi-step
information gathering plans follow-up queries, reuses intermediate results, adaptively
selects and reranks retrieval using trajectory state, and stops efficiently. A
controller-OFF bypass keeps normal RAG working and makes the engine's added value
directly measurable.

Project type: **Development** (not Research). Benchmarks, ablations and statistics
validate and debug the engineered artifact; no new learning algorithm is claimed.

## 2. Validation questions (from proposal)
- **VQ1:** Does controller-ON maintain or improve complete-evidence recall and task
  accuracy while reducing retrieval calls, redundancy and iterative latency versus
  one-shot and fixed-iteration baselines?
- **VQ2:** Which components (planning, caching, state-aware reranking, gap tracking,
  redundancy awareness, budget signals) materially affect quality, efficiency or
  robustness?
- **VQ3:** How well does the engine transfer from MultiHop-RAG to HotpotQA and curated
  agentic tasks while keeping an adapter-based architecture that needs no host changes?
  (Multi-backend validation beyond Qdrant is out of scope; backend independence is a
  design goal only.)

**Validation target (provisional):** ≈15–25 % fewer retrieval calls or redundant
retrievals than the fixed-iteration baseline, with task accuracy and complete-evidence
recall within ≈2–3 percentage points of the best comparable baseline — or higher
evidence recall at comparable retrieval cost. Missing the target is reported as a
measured limitation, not treated as implementation failure.

## 3. Functional requirements and acceptance criteria

**FR1 — Ingestion and indexing.** Reproducible chunking, embedding, metadata and Qdrant
indexing of benchmark corpora. (DATA.md, RETRIEVAL_SPEC.md)
- Running `python -m data.ingest --dataset multihop` twice from a clean state yields identical chunk
  IDs, identical point counts and identical gold-evidence maps (checked by
  `tests/integration/test_ingest_reproducible.py`).
- `scripts/check_gold_mapping.py` reports ≥ 95 % of gold evidence items mapped to at
  least one chunk; the unmapped list is saved for inspection.

**FR2 — Task-scoped state.** Create and persist S_t keyed by `task_id`. (STATE_SPEC.md)
- `POST /v1/tasks` returns a `task_id` and persists S_0; `GET /v1/tasks/{id}/state`
  returns it; state survives a server restart.

**FR3 — Two modes, one service.** (API_CONTRACT.md, ARCHITECTURE.md)
- `POST /v1/retrieve/direct` supports modes `dense`, `sparse`, `hybrid` and creates no
  state. Controlled endpoints use the same RetrieverAdapter instance.

**FR4 — Bounded query planner.** (PLANNER_SPEC.md)
- LLM output validates against `PlanLLMOutput` (ADR-009 §10); ungrounded entities cause rejection; at most
  `planner.max_attempts` LLM attempts per planning call; deterministic fallback after
  that; every accept/reject/fallback logged with a reason code.

**FR5 — Task-local cache/reuse.** (CACHE_SPEC.md)
- An equivalent pending query within the same task yields REUSE with zero backend calls
  and a `cache.hit` event. An identical query under a different `task_id` never hits.

**FR6 — Action selection.** (ACTION_POLICY.md)
- `Controller.select_action(state, cache_hit)` is a pure function of its inputs + config, returns the
  chosen action, the eligible set and per-action signal values; covered by unit tests
  constructing a synthetic state for every branch.

**FR7 — State-aware reranking.** (RERANK_SPEC.md)
- A constructed test where a lower-relevance but gap-covering, novel candidate must
  outrank a higher-relevance duplicate of existing evidence passes; with
  `ablation.state_aware_rerank=false` order equals cross-encoder order.

**FR8 — Evidence management and stopping.** (EVIDENCE_MANAGER_SPEC.md, ACTION_POLICY.md)
- Gaps are updated after every step; STOP is issued with a distinct `StopReason` for
  each stopping condition; budget caps are never exceeded (invariant test).

**FR9 — Return contract.** (STATE_SPEC.md `EvidenceBundle`)
- Every terminated task returns an EvidenceBundle; the engine never returns generated
  answer text.

**FR10 — Tracing and metrics.** (OBSERVABILITY.md, EVAL_PROTOCOL.md)
- Every system writes TraceSteps in the shared schema; `eval/metrics.py` computes all
  metrics from traces + bundles + gold map only, with no system-specific branches.

**FR11 — Service, client, agent adapter.** (API_CONTRACT.md, AGENT_INTEGRATION_SPEC.md)
- OpenAPI docs render at `/docs`; `RetrievalClient` covers every endpoint; the demo
  agent in `examples/` imports only `client/` (enforced by a test).

**FR12 — Benchmark formats.** (DATA.md)
- MultiHop-RAG, HotpotQA and the curated benchmark all load as `eval.datasets.Task`;
  `eval/run.py` has no dataset-specific branches.

## 4. Non-functional requirements
| NFR | Requirement | Verified by |
|---|---|---|
| Reproducibility | Versioned data/prompts/configs, fixed seeds, saved traces, run manifest with git commit + config hash + library versions | Manifest check in `eval/run.py`; reproducibility checklist (EVAL_PROTOCOL.md) |
| Modularity | Planner, cache, policy, retriever, reranker, evidence manager, LLM provider, agent integration each replaceable via config + one class | Ablation configs run without code edits |
| Observability | Every planned query, cache decision, action, result set, rerank decision, state update, latency, token count and stop reason logged | Trace schema completeness test |
| Efficiency | No fine-tuning/RL; per-task and per-run LLM call/token/cost caps; runs on student hardware (CPU acceptable for embeddings at MultiHop-RAG scale) | Budget tests; `llm.max_cost_usd_per_run` |
| Reliability | Backend/LLM failure never corrupts stored state; direct mode stays available | Fault-injection tests (TESTING.md) |
| Usability | Host integration needs only API_CONTRACT.md + client + a small adapter | Agent demo |
| Security | No secrets in git; retrieved text treated as untrusted; API bound to localhost by default, optional bearer token for remote demos (ADR-015) | SECURITY.md checks; `tests/unit/api/test_auth.py` |

## 5. Scope
**In scope:** textual multi-hop retrieval control; Qdrant dense/sparse/hybrid backend
behind an adapter; bounded planning; task-local caching; rule-based utility controller;
state-aware reranking with a pretrained cross-encoder; evidence management; stopping;
FastAPI service + client; one LangGraph agent demo; MultiHop-RAG (primary), HotpotQA
(secondary), compact curated benchmark (40–60 tasks).

**Out of scope — flag, do not build:** cross-task semantic caching (stretch experiment
only); graph/web retrieval adapters (stretch); training a reranker or foundation model;
RL/learned policies (stretch only after the rule-based controller is stable);
long-term personal memory; multi-agent orchestration; multimodal RAG; custom ANN
algorithms; production-scale distributed infrastructure; zero-code integration with
closed RAG products; comprehensive adversarial-content detection.

**Stretch (if time permits):** Adaptive-RAG-style router baseline; lightweight learned
policy/classifier; AgenticRAGTracer-compatible diagnostics; MCP adapter; cross-task
cache experiment; additional retrieval adapters.

## 6. Traceability matrix
| FR | Module(s) | Governing doc | Owner | First milestone |
|---|---|---|---|---|
| FR1 | `data/`, `engine/retrieval` | DATA, RETRIEVAL_SPEC | Kabir | Week 10 |
| FR2 | `engine/state` | STATE_SPEC | Samay | Week 10 (scaffold), Week 12 |
| FR3 | `engine/core` (`retrieve_direct`), `engine/api` | API_CONTRACT, ARCHITECTURE §6 | Kabir | Week 10 |
| FR4 | `engine/planner` | PLANNER_SPEC | Vishal | Week 12 |
| FR5 | `engine/cache` | CACHE_SPEC | Kabir | Week 16 |
| FR6 | `engine/controller` | ACTION_POLICY | Samay | Week 16 |
| FR7 | `engine/rerank` | RERANK_SPEC | Samay | Week 16 (prototype) |
| FR8 | `engine/evidence`, `engine/monitor` | EVIDENCE_MANAGER_SPEC, ACTION_POLICY | Samay | Week 16 (basic) |
| FR9 | `engine/core` | STATE_SPEC | Samay | Week 12 |
| FR10 | `engine/monitor`, `eval/` | OBSERVABILITY, EVAL_PROTOCOL | Vishal | Week 12 |
| FR11 | `engine/api`, `client/`, `examples/` | API_CONTRACT, AGENT_INTEGRATION_SPEC | Vishal | Week 16 (client), FYP-II (agent) |
| FR12 | `eval/datasets` | DATA | Vishal / Kabir | Week 10 (MultiHop), FYP-II (others) |

# Documentation Index

> Status: Stable · Owner: All · Last updated: 2026-10-04

This folder is the single source of truth for *what* the system does and *why*.
Code implements these documents. If code and a doc disagree, one of them is a bug —
fix it in the same PR.

## Reading order for a new teammate (or a fresh Claude Code session)
1. `../README.md` → `GLOSSARY.md` → `ARCHITECTURE.md`
2. `STATE_SPEC.md` (the shared data models everything depends on)
3. The spec for the module you are working on (table below)
4. `CONFIG_SCHEMA.md`, `TESTING.md`, `../CONTRIBUTING.md`

## Document catalogue
| Doc | Purpose | Owner | Needed from |
|---|---|---|---|
| `../SRS — …Workflows.docx` / `.pdf` / `.md` | Official Software Requirements Specification (submitted); authoritative for requirements; the `.md` copy (diagrams in `../assets/srs/`) is for reading on GitHub and by Claude Code | All | Week 7 |
| `../SDS — …Workflows.docx` / `.pdf` / `.md` | Official Software Design Specification (submitted); design summary; `.md` copy with diagrams in `../assets/sds/` | All | Week 7 |
| `PRD.md` | Developer summary of the SRS: FR1–FR12, NFRs, acceptance criteria, scope, traceability | All | Week 7 |
| `GLOSSARY.md` | Shared vocabulary | All | Week 7 |
| `ARCHITECTURE.md` | Layers, modules, Engine facade, LangGraph graph, loop semantics | All | Week 7 |
| `STATE_SPEC.md` | All shared Pydantic models, enums, invariants, field ownership | Samay | Week 7 |
| `RETRIEVAL_SPEC.md` | RetrieverAdapter, Qdrant collection layout, modes, hybrid fusion | Kabir | Week 8 |
| `DATA.md` | Datasets, ingestion, chunking, IDs, gold mapping, splits, curated benchmark | Kabir | Week 7 |
| `PLANNER_SPEC.md` | Bounded planner, output schema, grounding, retry/fallback | Vishal | Week 10 |
| `CACHE_SPEC.md` | Task-local cache, normalization, lookup, REUSE semantics | Kabir | Week 12 |
| `ACTION_POLICY.md` | Actions, preconditions, utility signals, monitor, stopping, calibration | Samay | Week 12 |
| `RERANK_SPEC.md` | Cross-encoder + state-aware scoring, MMR selection | Samay | Week 13 |
| `EVIDENCE_MANAGER_SPEC.md` | Acceptance, gap resolution, evidence gain, verification | Samay | Week 12 |
| `LLM_CLIENT_SPEC.md` | Providers, structured output, caching, budgets, cost | Vishal | Week 10 |
| `API_CONTRACT.md` | HTTP endpoints, errors, client | Vishal | Week 8 |
| `BASELINES_SPEC.md` | Exact definitions of the comparison systems | Vishal | Week 9 |
| `EVAL_PROTOCOL.md` | Experiments, metrics, fairness, statistics, reporting | Vishal | Week 8 |
| `OBSERVABILITY.md` | Trace schema semantics, events, run folders, manifests | Vishal | Week 8 |
| `CONFIG_SCHEMA.md` | Every config key, default, owner, frozen status | All | Week 7 |
| `TESTING.md` | Test layers, fakes, fixtures, CI | Vishal | Week 7 |
| `SECURITY.md` | Secrets, prompt-injection handling, API exposure | All | Week 10 |
| `SETUP.md` | Windows environment, Qdrant, command reference, troubleshooting | Kabir | Week 7 |
| `AGENT_INTEGRATION_SPEC.md` | External LangGraph agent demo, MCP stretch | Vishal | FYP-II |
| `LIBRARY_NOTES.md` | Pinned versions + verified API usage | All | Ongoing |
| `TASKS.md` | Milestone checklist per week and owner | All | Ongoing |
| `decisions/` | Architecture Decision Records (ADRs) | Author of each | Ongoing |
| `reports/` | FYP-I report draft, results, trajectory review, slides outline | All | Week 16 |

## Maintenance rules
1. **Header line.** Every doc starts with
   `> Status: Draft | Stable | Frozen · Owner: <name> · Last updated: YYYY-MM-DD`.
   - *Draft*: may change freely. *Stable*: change via PR with owner review.
     *Frozen*: change only via ADR (used for contracts and calibrated values).
2. **Same-PR rule.** A PR that changes behaviour described in a doc updates that doc.
3. **Contracts.** `STATE_SPEC.md`, `API_CONTRACT.md` and the trace schema in
   `OBSERVABILITY.md` change only with an ADR in `decisions/`.
4. **No duplication.** A fact lives in one doc; others link to it. Config defaults live
   in `CONFIG_SCHEMA.md` / `configs/base.yaml` only — other docs cite the key name
   (e.g. `monitor.stagnation_threshold`), never a copy of the number, except in clearly
   labelled worked examples.
5. **Official documents.** The SRS is the authoritative requirements document; `PRD.md`
   summarises it for developers and is fixed whenever the two disagree. The SDS
   summarises the design; where it differs from a spec in this folder (with its ADRs),
   the spec is current and the SDS is updated in the same PR. For both documents the Word
   file is the master: edit it, re-export the PDF, and update the `.md` copy in the same PR.
6. **Monthly review.** First week of each month, each owner re-reads their docs against
   the code and bumps "Last updated".

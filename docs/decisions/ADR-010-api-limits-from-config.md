# ADR-010: API request limits come from configuration

> Status: Accepted (2026-10-08, Kabir for the team; Samay/Vishal review after the fact) · Date: 2026-10-04 · Author(s): Vishal · Reviewer: Kabir, Samay

## Context
API_CONTRACT hard-coded request limits in the HTTP schemas: goal length 3–2,000
characters, query length ≤ 2,000, `k` default 10 and maximum 100. This conflicted with:
- CLAUDE.md rule 4 (no literal thresholds or k values in code), the same rule ADR-009
  §10 applied to the planner schema;
- `retrieval.default_k` (20), which the baselines and SRS NFR-P1 use for direct retrieval;
- RERANK_SPEC §4, which caps reranker input at `retrieval.max_k` (60) — a direct call
  with `k = 100, rerank = true` would have broken that cap.

## Decision
1. HTTP schemas keep only structural bounds: non-empty strings, `k ≥ 1`, and
   `k: int | None = None`.
2. The router validates against configuration and returns 422 `VALIDATION_ERROR` on
   failure:
   - goal length within [`api.goal_min_chars`, `api.goal_max_chars`];
   - query length ≤ `api.query_max_chars`;
   - `k` omitted → `retrieval.default_k`; `k > retrieval.max_k` → rejected.
3. New config keys `api.goal_min_chars = 3`, `api.goal_max_chars = 2000`,
   `api.query_max_chars = 2000` keep the previous length limits unchanged.

## Consequences
Behaviour change for `/v1/retrieve/direct` only: the default `k` becomes 20 (was 10)
and the maximum becomes 60 (was 100). Updated in the same change: API_CONTRACT.md,
CONFIG_SCHEMA.md, `configs/base.yaml`, SDS §6.1. The OpenAPI snapshot
(`tests/fixtures/openapi.json`, TESTING §4.7) is created from the amended schemas.

## Reversal condition
If a host application needs a different limit, change the config keys; reverting to
schema literals requires a new ADR.

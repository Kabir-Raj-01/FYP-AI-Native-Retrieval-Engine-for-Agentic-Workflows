# Security and Robustness

> Status: Stable · Owner: All · Last updated: 2026-10-04

## 1. Secrets
- API keys only in `.env` (gitignored); `.env.example` lists variable names with empty
  values. Config files hold env var **names** (`llm.<role>.api_key_env`), never values.
- A pre-commit hook (`detect-secrets`, configured in `.pre-commit-config.yaml` with the
  baseline `.secrets.baseline`) runs on every commit.
- Keys are never logged; the LLM client redacts headers in errors.
- Leaked key → revoke immediately at the provider, rotate, and note it in the team chat.

## 2. API exposure
- Default bind `127.0.0.1`. The API has no authentication in the core scope; do not bind
  to `0.0.0.0` on shared networks. For a remote demo, put it behind an SSH tunnel, or set
  `ENGINE_API_TOKEN` to enable a simple bearer-token check (optional middleware).
- Request size limits (goal/query length, `k`) come from configuration and are checked by
  the router, which returns 422 `VALIDATION_ERROR` (API_CONTRACT, ADR-010).

## 3. Untrusted retrieved content (indirect prompt injection)
Retrieved chunks may contain text like "ignore previous instructions". Mitigations built
into the design:
1. **The controller is not an LLM.** Action selection, caching and stopping are
   deterministic code, so injected text cannot directly choose actions.
2. **Role separation.** Retrieved text never appears in a system message. In user
   messages it is wrapped:
   ```
   <evidence untrusted="true">
   <doc id="c_1a2b">…text…</doc>
   </evidence>
   ```
   and the system prompt states that content inside `<evidence>` is data to analyse, never
   instructions. Angle brackets inside chunk text are escaped before wrapping.
3. **Planner output is validated** (schema + entity grounding). Known limitation: an
   injected entity *inside accepted evidence* is groundable, so injection can still steer
   a rewrite toward that entity. Logged traces make this inspectable.
4. **The verifier** returns only a boolean; its influence is bounded to gap resolution.
5. **Answer generation** (host/eval) uses the same wrapping.
A test (`tests/unit/planner/test_injection.py`) uses `fixtures/injection_doc.txt` and
checks the prompt structure and that planner output requesting an out-of-schema action is
rejected. Comprehensive adversarial detection is out of scope (future work).

## 4. Data and licences
Use datasets under their licences (record in ADR-003). Do not commit third-party corpora.
Curated benchmark sources must be publicly shareable or excluded from the public repo.

## 5. Dependency hygiene
Pinned versions; `pip-audit` run monthly; models downloaded only from the
official Hugging Face repos named in ADRs.

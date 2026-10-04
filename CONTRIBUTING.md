# Contributing

> Status: Stable · Owner: All · Last updated: 2026-10-04

## Branches
- `main` is protected: PR only, CI green, one approving review from another member.
- Branch names: `<initial>/<area>-<short-topic>` e.g. `k/retrieval-hybrid`, `s/controller-signals`, `v/planner-validator`.
- Keep branches short-lived (≤ 1 week). Rebase on `main` before requesting review.

## Commits — Conventional Commits
`feat(controller): add strategy-switch precondition` · `fix(cache): scope lookup by task_id`
· `test(planner): hallucinated entity rejected` · `docs(state): add EvidenceGap.status`
· `chore(ci): start qdrant.exe for integration tests`. One logical change per commit.

## Pull requests
Use `.github/pull_request_template.md`. A PR must:
1. Reference the TASKS.md item / issue and the governing doc.
2. Include tests (TESTING.md) and pass `ruff check .`, `mypy --strict engine` and the unit
   tests (`pytest -m "not integration and not llm and not e2e"`).
3. Update docs affected by the change (same-PR rule, docs/README.md).
4. Include an ADR if it changes a contract or a Frozen key.
5. Stay reviewable: aim for < 400 changed lines excluding tests/fixtures.

Reviewers check: matches the spec, respects the field-ownership matrix and dependency
rule, no literals that should be config, logging present, tests meaningful (not just
coverage), and **the author can explain every line** — AI-generated code included.

## Working with Claude Code (or any AI assistant)
- Run it from the repo root so `CLAUDE.md` loads. One feature per session; start fresh
  sessions often.
- Plan first (plan mode), then tests, then implementation. Review every diff yourself.
- Paste real command output back; never accept "tests pass" without seeing it.
- When the assistant uses a library API, check it against `docs/LIBRARY_NOTES.md`; record
  newly verified usage there.
- Do not let the assistant edit Frozen docs/keys or golden snapshots without an explicit,
  reviewed reason.
- Follow FAST's current policy on AI assistance and acknowledge tool use in the report.

## Releases / tags
`v0.1-skeleton` (Wk 7) · `v0.2-week10` · `v0.3-midterm` (Wk 12) · `v0.5-fyp1` (Wk 16)
· `v0.8-frozen-config` (calibration) · `v1.0` (code freeze FYP-II).
Reported runs must come from a tagged or clean committed state.

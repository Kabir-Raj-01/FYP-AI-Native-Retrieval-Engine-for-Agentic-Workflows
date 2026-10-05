# Testing Strategy

> Status: Stable · Owner: Vishal (framework) — each module owner writes their module's tests · Last updated: 2026-10-04
> §4 categories 2 and 5 amended by ADR-009.

## 1. Layers
| Layer | Folder | Marker | Needs | Runs |
|---|---|---|---|---|
| Unit | `tests/unit/` | (none) | nothing — fakes only | every commit, every PR (CI) |
| Integration | `tests/integration/` | `@pytest.mark.integration` | local Qdrant (`qdrant.exe`), real embedder/cross-encoder (small models OK) | PR to `main` (CI Windows runner), locally via `pytest -m integration` |
| LLM | `tests/llm/` | `@pytest.mark.llm` | real LLM (local or API) | manually before milestones; never in CI |
| End-to-end | `tests/e2e/` | `@pytest.mark.e2e` | Qdrant + fixture corpus + fake or local LLM | before milestones |
The unit-test command `pytest -m "not integration and not llm and not e2e"` runs unit tests only.

## 2. Fakes (`tests/fakes/`) — implement the real interfaces
| Fake | Behaviour |
|---|---|
| `FakeEmbedder` | deterministic vectors: explicit `{text: vector}` map for tests that need controlled similarities; otherwise hash-seeded random unit vectors (identical text → identical vector, different text → near-orthogonal) |
| `FakeRetriever` | in-memory corpus; scripted results per `(query, mode)`, or a simple token-overlap scorer; can be told to raise `BackendError` on the n-th call |
| `FakeReranker` | relevance from a `{chunk_id: score}` map; state-aware logic can be tested with the real Reranker + FakeEmbedder + fake cross-encoder scorer |
| `FakeLLMClient` | queue of scripted responses per role; records every request; can raise `LLMError` / `LLMOutputError`; reports configurable token usage |
| `FakeClock` | controllable time for latency fields |
| `FakePlanner`, `FakeController`, `FakeEvidenceManager`, `FakeMonitor` (`tests/fakes/controlled.py`) | scripted plans / action sequence / acceptance (optionally resolving the current gap) / ACTION_POLICY §6.3-§6.4 stop checks without stagnation logic — for graph and loop tests before the real components exist |
`tests/conftest.py` provides fixtures: `cfg` (base config + test overrides), `state_factory`
(builds `RetrievalState` with given gaps/evidence/budget), `engine_with_fakes`,
`tmp_state_db`.

## 3. Fixtures (`tests/fixtures/`)
- `mini_corpus.jsonl` — ~30 short handcrafted chunks covering 5 two-hop questions, with
  one exact-entity question, one paraphrase-heavy question and one null question.
- `mini_tasks.jsonl` — those 5 questions with gold groups.
- `planner_responses/*.json` — valid, hallucinated, malformed planner outputs. (As built,
  the planner tests script these inline in `tests/unit/planner/test_planner.py`; move them
  to fixture files if other tests need them.)
- `injection_doc.txt` — a chunk containing instructions aimed at the planner (SECURITY §3).

## 4. Required test categories
1. **Per-module unit tests** listed in each spec's "Tests" section.
2. **State invariants** (`test_state_invariants.py`): run the engine with fakes over many
   synthetic tasks (Hypothesis property tests where practical) and assert
   `check_invariants` (STATE_SPEC §5; ADR-009 §5 — called by `persist`, not a Pydantic
   validator) after every completed iteration. Includes:
   - **`max_steps` exact iteration count** (ADR-009 §12): with `max_steps=3`, the engine
     performs exactly 3 iterations, all real (non-STOP-only) actions, and ends with
     `steps_used == 3` and `stop_reason == MAX_STEPS`; the controller's own
     `steps_used >= max_steps` guard (ACTION_POLICY §2 step 1) never fires.
   - **Concurrent step rejected:** two concurrent `step()`/`run()` calls on the same
     `task_id` — the in-process lock rejects the second with `TaskConflict` (API_CONTRACT:
     409; the lock is non-blocking, so it does not queue the call); if the lock is
     bypassed (e.g. two separate `Engine` instances against the same DB), the second
     `TaskStore.save` raises `TaskConflict` on a stale `expected_version`.
   - **Serialization round-trip** (`engine/state/serialization.py`; ADR-009 §4a
     correction): for a `RetrievalState` with gaps, evidence, vectors and a non-null
     `current_gap_id`, `load_with_vectors(RetrievalState, dump_with_vectors(state))`
     equals `state` field-for-field, including the three computed properties
     (`current_subgoal`, `pending_query`, `current_need_type`) recomputing to their
     original values even though `dump_with_vectors` excludes them from the payload.
     Also covers `current_gap_id = None` (computed properties fall back to
     `""`/`""`/`MIXED`).
3. **Worked example** (`tests/unit/controller/test_worked_example.py`) reproducing
   ACTION_POLICY §8 exactly. Invariant 9's static part is
   `tests/unit/test_import_boundaries.py::test_components_do_not_store_state_references`
   (no `self.<attr> = …state…` in component methods).
4. **Golden trajectories** (`tests/e2e/test_golden.py`): with fakes and the fixture
   corpus, the action sequence and stop reason per task match a stored snapshot
   (`tests/fixtures/golden/*.json`). Snapshot updates require a PR comment explaining why.
5. **Fault injection:** retriever raises on step 2 → stored state equals state after step
   1; LLM raises during TRANSFORM → same (as built: an `LLMError` is absorbed by the
   planner's fallback, PLANNER_SPEC §6, so the step completes; the per-run cap
   `LLMBudgetExceeded` propagates and leaves the stored state — `tests/unit/graph/test_faults.py`); API returns the mapped error code; **no orphan
   cache entries after a failed step** (ADR-009 §9) — a fresh `CacheEntry` queued in
   `scratch.queued_cache_entries` before the failure is never committed, so a subsequent
   retry's `Cache.lookup` cannot return it.
6. **Leakage:** identical queries in two tasks never produce a cross-task cache hit.
7. **Contract tests:** API responses validate against the schemas; OpenAPI schema
   snapshot (`tests/fixtures/openapi.json`) changes only with an ADR.
8. **Metric tests:** `eval/metrics.py` on hand-built traces with known answers
   (Recall@k, NDCG@k, CER with groups, over-retrieval, premature stop).
9. **Import-boundary test:** no file in `examples/` imports `engine`; no module outside
   `engine/llm` imports `anthropic` or `openai`; only `engine/factory.py` imports concrete
   component classes (checked by parsing imports with `ast`).
10. **Config tests:** base.yaml loads; every experiment file loads; invalid weight sums
    are rejected; `budget_defaults.max_llm_calls = 0` loads and `Budget.pressure()` does
    not raise (ADR-009 §8).

## 5. Conventions
- File mirrors module: `engine/cache/task_cache.py` → `tests/unit/cache/test_task_cache.py`.
- Test names describe behaviour: `test_reuse_not_eligible_without_cache_hit`.
- Arrange/act/assert; one behaviour per test; no sleeps (use FakeClock).
- Coverage target: ≥ 85 % lines for `engine/controller`, `engine/evidence`,
  `engine/monitor`, `engine/planner`, `engine/cache`; ≥ 70 % overall (add `--cov=engine --cov-report=term-missing` to the unit-test command).

## 6. CI (`.github/workflows/ci.yml`)
Job 1 (every push/PR): install, `ruff check`, `ruff format --check`, `mypy --strict engine`,
unit tests with coverage. Job 2 (PRs to `main`): downloads and starts the native Windows
`qdrant.exe`, then runs the integration tests. Both jobs run on `windows-latest`.
Branch protection: both jobs green + one approving review.

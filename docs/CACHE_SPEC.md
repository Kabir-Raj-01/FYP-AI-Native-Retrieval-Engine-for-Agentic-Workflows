# Task-Local Cache and REUSE

> Status: Stable · Owner: Kabir · Last updated: 2026-10-05 · Related: ACTION_POLICY.md §3, ADR-009
> Implementation: `engine/cache/task_cache.py` (`TaskCache`), `engine/text.py` (`normalize`),
> `engine/state/sqlite.py` (schema + entry write rules shared with the TaskStore).

## 1. Purpose
Avoid backend calls when the pending query asks for something this task already
retrieved. Scope is **one task only** — cross-task caching is out of scope because it
would leak results between benchmark items.

## 2. Normalisation
`normalize(q)`: Unicode NFC → lowercase → strip punctuation except inside tokens with
digits (keeps "3.5", "COVID-19") → collapse whitespace → trim. Used for exact matching
and for duplicate-query checks in the planner and controller.
Precisely (`engine/text.py`): punctuation = Unicode category `P*`. In a whitespace token
that contains a digit, only leading/trailing punctuation is removed (`"(2023),"` →
`2023`, `covid-19` and `3.5` kept); in any other token each punctuation character becomes
a space (`state-of-the-art` → `state of the art`, `U.S.` → `u s`).

## 3. Lookup (`Cache.lookup(task_id, query, need_type) -> CacheHit | None`)
1. If `ablation.cache` is false → None.
2. Candidate entries = entries with the same `task_id` (SQL filter — never scan others).
3. **Exact:** an entry with identical `normalized_query` → hit, similarity 1.0.
4. **Semantic:** embed query (Embedder, query side); best entry by cosine with
   `query_vector`; hit if ≥ `cache.similarity_threshold`.
5. **Compatibility:** an entry is usable only if its mode can serve the need:
   `dense` entries serve semantic/mixed; `sparse` serve lexical; `hybrid` serve all.
6. Return the best compatible hit, or None. Emit `cache.hit` / `cache.miss`.

Compatibility filters both exact and semantic matches. Exact matching is tried first and
needs no embedding. Semantic similarity is the cosine of `Embedder.embed_queries([query])`
with the stored `query_vector`; the threshold is inclusive (`≥`). Ties prefer the entry
with more candidates, then the newest (`created_at_step`), then `entry_id`. Producers must
compute `query_vector` with `Embedder.embed_queries` (the same model and query prefix).

## 4. Put
After every fresh DENSE / SPARSE_HYBRID / RERANK_EXPAND retrieval, `score_and_update`
builds a `CacheEntry` with all candidates (with vectors) and the ids accepted as
evidence, and appends it to `scratch.queued_cache_entries` — it is **not** written to
the DB at this point. `persist` passes the queued entries to
`TaskStore.save(state, expected_version, cache_entries=scratch.queued_cache_entries)`,
which commits the state save and every queued entry (insert, or replace if an entry with
the same normalised query and mode exists — the deeper result wins) in **one atomic
transaction** (ADR-009 §9, signature corrected — the `cache_entries` parameter on `save`
itself, not a save-then-separately-committed write). If the iteration fails before
`persist`, `save` is never called, so no entry is written and a retried step can never
REUSE a result from a step that was never actually saved. If `save` raises
`TaskConflict` (stale version), none of it is written either. Max
`cache.max_entries_per_task`; beyond that, the oldest entry is dropped (lowest
`created_at_step`, then `entry_id`; logged `cache.evict`). ("Oldest non-exact entry" in
earlier drafts could not be implemented: exact/semantic is a property of a hit, not of an
entry.) "The deeper result wins": a new entry with the same task, normalised query and mode
replaces the stored one only if it has at least as many candidates; otherwise the stored
entry is kept (logged `cache.put` with `kept`). These rules live in
`engine/state/sqlite.write_cache_entries`, used by `Cache.put` and by `TaskStore.save`
inside its transaction.

## 5. REUSE semantics
- REUSE feeds the hit's stored candidates through the normal rerank → evidence path,
  reranked against the **current** pending query (not the cached entry's original
  normalised query — ADR-009 §9). Already-accepted items are rejected as DUPLICATE, so
  REUSE naturally shows low gain when the information is already in E_t — the
  controller's gain signal for REUSE accounts for that (ACTION_POLICY §4).
- REUSE consumes one step and zero retrieval calls.
- REUSE is eligible only when `lookup` returned a hit for the current pending query.
- Its `RetrievalRecord`/`LastResult` use `mode = entry.mode`, `k = len(entry.candidates)`,
  `query = pending_query`; accepted items get `source_action = REUSE`;
  `RetrievalRecord.cache_hit = True` (ADR-009 §9). A `lookup` hit that exists but is
  **not** chosen by the controller (e.g. a fresh search won on utility instead) leaves no
  `RetrievalRecord` at all — it is visible only as the `cache.hit` event.

## 6. Persistence
Entries live in the `cache_entries` table of `paths.state_db` (ARCHITECTURE §7, schema
SDS §4.6), so a task can continue after a restart. The cache table is never read by a
different task (every read is `WHERE task_id = ?`). Foreign keys are enforced: an entry for
an unknown task is rejected (`TaskNotFound` from `Cache.put`). Rows whose `schema_version`
this code does not know raise an error instead of being misread (ADR-009 §6). With
`ablation.cache` false, `lookup` returns None without touching the database; `put` still
writes (the ablation disables the policy, not the data). One SQLite connection per
thread.

## 7. Tests
- Exact normalised match → hit 1.0; paraphrase above threshold → semantic hit; below → miss.
- Same query, different `task_id` → miss (explicit leakage test).
- Mode compatibility: dense entry does not serve a lexical need.
- Replacement and max-entries eviction.
- `ablation.cache=false` → always miss, no DB reads.

# Data Specification

> Status: Draft (freeze after ADR-001 and ADR-003 are accepted) · Owner: Kabir · Last updated: 2026-10-05

## 1. Common task interface (`eval/datasets/base.py`)
Every benchmark loads into the same shape so `eval/run.py` has no dataset branches.
```python
class Task(BaseModel):
    task_id: str               # "<dataset>:<original_id>"
    dataset: str               # "multihop" | "hotpotqa" | "curated"
    question: str              # becomes RetrievalState.goal
    answer: str | None         # gold answer (None for tasks without one)
    question_type: str | None  # e.g. inference / comparison / temporal / null
    gold_groups: list[list[str]]  # one inner list per evidence entry (§4); any member satisfies it
    split: str                 # "dev" | "test"
    meta: dict[str, str | int | None] = {}
```

## 2. MultiHop-RAG (primary)
Source: Tang & Yang, *MultiHop-RAG*, COLM 2024. A news-article corpus (≈600 articles)
with ≈2.5k multi-hop queries of four types — inference, comparison, temporal and null
(insufficient-information) queries — each annotated with supporting evidence.
- **Source and licence** (verified 2026-10-04, ADR-003): files `corpus.json` and
  `MultiHopRAG.json` from the Hugging Face dataset `yixuantt/MultiHopRAG` (they are no
  longer in the GitHub repository); licence ODC-BY 1.0. Reference SHA-256 values are
  in ADR-003. Counts above are approximate; record the numbers you observe there.
- **Field names** (verified from the downloaded files, 2026-10-05; counts in ADR-003):
  queries have `query`, `answer`, `question_type` (`inference_query`,
  `comparison_query`, `temporal_query`, `null_query`) and `evidence_list`, whose entries
  hold a supporting `fact` plus the source article's `title`, `url`, `source`,
  `published_at`, `author`, `category`. Corpus items carry `title`, `body` (the article
  text), `url`, `source`, `published_at`, `author`, `category` — no document id, so
  `doc_id` = sha1 of `url` (§3; `url` is unique). Download:
  `python -m data.download --dataset multihop` (pinned revision and checksums in
  `data.sources.multihop`; files land in `data/raw/multihop/`).
- **Null queries** have no answer in the corpus. They are kept: they test whether the
  engine stops without over-retrieving. Their gold evidence list is empty; they are
  excluded from recall metrics and included in answer accuracy and efficiency metrics.

## 3. Ingestion pipeline (`data/ingest.py`, `python -m data.ingest --dataset ...`)
1. **Load** raw files from `data/raw/<dataset>/`; verify checksums from ADR-003.
2. **Normalise** text: Unicode NFC, collapse whitespace, strip boilerplate. No lowercasing.
3. **Chunk** per ADR-001: `chunking.chunk_size_tokens`, `chunking.chunk_overlap_tokens`,
   tokenizer `chunking.tokenizer`, sentence-aware (never split inside a sentence unless a
   single sentence exceeds the chunk size).
4. **IDs:** `doc_id` = stable id from the source (or sha1 of url/title if none);
   `chunk_id = sha1(f"{doc_id}:{char_start}")[:16]`; Qdrant point id = UUIDv5(chunk_id).
5. **Embed** passages with the Embedder (batch `embedding.batch_size`); encode sparse
   vectors with the configured sparse encoder.
6. **Upsert** to Qdrant in batches; idempotent (same point ids).
7. **Write** `data/processed/<dataset>/chunks.jsonl` (chunk_id, doc_id, text, spans,
   metadata) and `data/processed/<dataset>/ingest_manifest.json` (config hash, model,
   chunk count, doc count, collection name, git commit).
8. **Build gold map** (§4) and **split** (§5).

## 4. Gold-evidence mapping
Output: `data/processed/<dataset>/gold_map.json` = `{task_id: [chunk_id, ...]}`.
Algorithm for MultiHop-RAG:
1. For each evidence entry, find the source document (match url, else title).
2. Within that document's chunks, find the chunk(s) containing the fact snippet:
   exact substring match first; else fuzzy match (`rapidfuzz.partial_ratio` ≥
   `data.gold_fuzzy_threshold`) → take best-scoring chunk; if the snippet straddles a
   boundary, include both chunks.
3. Unmapped entries are written to `gold_unmapped.json`.
`scripts/check_gold_mapping.py` prints coverage. **Acceptance: ≥ 95 % of evidence
entries mapped.** Below that, fix chunking/matching before any evaluation.
Metrics count a task's gold evidence as the set of chunk ids; if an entry maps to two
chunks, retrieving either satisfies that entry (the map stores groups:
`{task_id: [[chunk_a, chunk_b], [chunk_c]]}` — one inner list per evidence entry).

## 5. Splits
- Stratified by `question_type`, fraction `data.dev_fraction` to dev, seed
  `data.split_seed`. Written once to `data/processed/<dataset>/split.json` and committed
  (small file). Never regenerated after ADR-003 is accepted.
- **FYP-I working subset:** `data.dev_subset_size` dev tasks (stratified, same seed) for
  fast/cheap iteration. Full dev split for calibration; test split once (EVAL_PROTOCOL).

## 6. HotpotQA (secondary, FYP-II)
- Setting: **distractor** dev set. Corpus = the pooled set of all context paragraphs of
  the sampled questions (`data.hotpot_sample_size`, seed-fixed); each paragraph is one
  chunk (they are short). Gold = paragraphs named in `supporting_facts`.
- Answer metrics: standard EM/F1. Record the exact sample and setting in an ADR before
  running.

## 7. Curated benchmark (FYP-II, 40–60 tasks)
File: `data/curated/tasks.jsonl` + `data/curated/annotations.csv`.
```python
class CuratedTask(BaseModel):
    task_id: str
    task_type: Literal["research_synthesis", "codebase_navigation"]
    goal: str
    source_corpus: str            # collection name it runs against
    expected_evidence: list[list[str]]   # groups of chunk ids, same semantics as gold map
    info_path: list[str]          # ordered description of the multi-step path
    reference_answer: str | None
    author: str
    reviewer: str
```
Annotation guide:
- 2–5 evidence groups per task; at least two groups must require information found only
  after reading an earlier piece (true multi-step).
- Research synthesis: corpus of papers/articles chunked like MultiHop-RAG.
- Codebase navigation (exploratory only): an open-source repo chunked by top-level
  function/class blocks with `path` and `line_start/line_end` metadata. No claim that
  vector retrieval beats symbol/AST search.
- Every task reviewed by a second member; disagreements resolved and logged in the CSV.

## 8. Storage conventions
`data/raw/` and `data/processed/` are gitignored except `split.json`, `gold_map.json`
(small) and `data/curated/`. Raw data is re-downloadable via `data/download.py`.

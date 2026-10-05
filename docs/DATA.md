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
1. **Load** raw files from `data/raw/<dataset>/`; verify checksums from ADR-003
   (`data.sources.<dataset>.files`; a missing or mismatching file stops ingestion).
2. **Normalise** text: Unicode NFC, collapse every whitespace run to one space, trim. No
   lowercasing and **no boilerplate stripping**: every MultiHop-RAG gold fact matches its
   article verbatim after exactly this normalisation, and stripping could delete facts
   (verified 2026-10-05; `data/chunking.py:normalize_text`).
3. **Chunk** per ADR-001: `chunking.chunk_size_tokens`, `chunking.chunk_overlap_tokens`,
   tokenizer `chunking.tokenizer`, sentence-aware (never split inside a sentence unless a
   single sentence exceeds the chunk size). Implementation (`data/chunking.py`):
   - Sentences: a deterministic regex ends a sentence at `.`/`!`/`?` (plus closing
     quotes or brackets) followed by whitespace and a capital letter, digit or opening
     quote/bracket. Imperfect on abbreviations; that only moves chunk boundaries.
   - Packing: whole sentences are added greedily while the token count of the candidate
     chunk text itself (tokenizer, no special tokens) stays ≤ `chunk_size_tokens`.
   - Overlap: the next chunk starts with the previous chunk's trailing whole sentences,
     as many as fit in `chunk_overlap_tokens` (never all of them).
   - A sentence longer than a chunk is cut into token windows of `chunk_size_tokens`
     overlapping by `chunk_overlap_tokens`.
   - Chunk text is the exact slice of the normalised document; the title is metadata,
     not prepended.
4. **IDs** (`data/ids.py`): `doc_id` = stable id from the source, or `sha1(url)[:16]` if
   none (MultiHop-RAG: none; `url` is unique);
   `chunk_id = sha1(f"{doc_id}:{char_start}")[:16]` with `char_start` into the
   normalised text; Qdrant point id = UUIDv5(`qdrant.id_namespace`, chunk_id).
   MultiHop-RAG task ids are `multihop:<index>` (0-based position in the pinned queries
   file, zero-padded to 4 digits; the file has no ids).
5. **Embed** passages with the Embedder (batch `embedding.batch_size`); encode sparse
   vectors with the configured sparse encoder (RETRIEVAL_SPEC §5).
6. **Upsert** to Qdrant in batches (`qdrant.upsert_batch_size`); idempotent (same point
   ids); resumable.
7. **Write** `data/processed/<dataset>/chunks.jsonl` — one line per chunk:
   `{chunk_id, doc_id, text, metadata: {title, source, url, published_at, author,
   category, dataset, char_start, char_end, n_tokens}}` (the same shape as
   `tests/fixtures/mini_corpus.jsonl`) — and `ingest_manifest.json` (config hash, git
   commit and dirty flag, raw-file SHA-256s, chunking settings, counts, gold coverage,
   split hashes, SHA-256 of every output; collection name and embedding model are added
   with steps 5–6).
8. **Build gold map** (§4) and **split** (§5).

Status (2026-10-05): all steps are implemented. Steps 5–6 run with
`python -m data.ingest --dataset multihop --index` (`data/index.py`): passages are
embedded with the Embedder (`embedding.batch_size`) and BM25-encoded with the
SparseEncoder, then upserted through the IndexWriter in batches of
`qdrant.upsert_batch_size`; the components come from `engine.factory`, so `data/` never
imports a model library or the Qdrant client. Indexing is resumable (chunks already in the
collection are skipped) and refuses a collection holding chunk ids that are not in
`chunks.jsonl` (an older chunking needs a new collection name, ADR-001). The manifest gains
an `index` section (collection, counts, duration, model names and revisions, BM25
parameters). Two step-1 runs from the same raw files and config produce byte-identical
outputs (manifest timestamp aside).

## 4. Gold-evidence mapping
Output: `data/processed/<dataset>/gold_map.json` = `{task_id: [[chunk_id, ...], ...]}`
(one group per evidence entry; `[]` for null queries). Algorithm for MultiHop-RAG
(`data/gold.py`):
1. For each evidence entry, find the source document (match url, else title).
2. Find the (normalised) fact in the (normalised) document text: every exact
   occurrence; else the best fuzzy alignment (`rapidfuzz.fuzz.partial_ratio_alignment`
   ≥ `data.gold_fuzzy_threshold`). The entry's group is every chunk that fully contains
   an occurrence; for an occurrence that straddles a chunk boundary (no chunk contains
   it), every chunk it overlaps. With overlapping chunks a fact can lie in two chunks;
   a fact repeated in its article contributes the chunks of every occurrence.
3. Unmapped entries are written to `gold_unmapped.json` (reason `NO_DOCUMENT` or
   `NO_MATCH`) and contribute no group.
`scripts/check_gold_mapping.py <dataset>` prints coverage overall and per split.
**Acceptance: ≥ `data.gold_min_coverage` (95 %) of evidence entries mapped** — exit
code 1 below it. Below that, fix chunking/matching before any evaluation.
MultiHop-RAG result (2026-10-05): 6,084 / 6,084 (100 %), all exact (ADR-001 Evidence).
Metrics count a task's gold evidence as the set of chunk ids; if an entry maps to two
chunks, retrieving either satisfies that entry (the map stores groups:
`{task_id: [[chunk_a, chunk_b], [chunk_c]]}` — one inner list per evidence entry).

## 5. Splits
- Stratified by `question_type`, fraction `data.dev_fraction` to dev, seed
  `data.split_seed`. Written once to `data/processed/<dataset>/split.json` and committed
  (small file). Never regenerated after ADR-003 is accepted. Algorithm (`data/split.py`):
  within each type, sort the task ids, shuffle with `random.Random(f"{seed}:{type}")`
  (each type independent of the others) and put the first `round-half-up(n ×
  dev_fraction)` in dev.
- **FYP-I working subset:** `data.dev_subset_size` dev tasks (stratified, same seed) for
  fast/cheap iteration: exact proportional allocation per type (largest remainder, ties
  by type name), each type shuffled with `random.Random(f"{seed}:{type}:subset")`. Full
  dev split for calibration; test split once (EVAL_PROTOCOL).
- `split.json` = `{dataset, seed, dev_fraction, dev_subset_size, dev, test,
  dev_subset}` with sorted id lists; its SHA-256 and the subset's are in ADR-003.

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

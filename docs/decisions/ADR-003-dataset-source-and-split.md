# ADR-003: MultiHop-RAG source, licence, checksums, split and answer metric

> Status: Accepted (2026-10-08, Kabir for the team; Samay/Vishal review after the fact) · Date: 2026-09-25 · Updated: 2026-10-05 · Author(s): Kabir · Reviewer: Vishal

## Context
Results must be reproducible from a named, checksummed dataset version, and the dev/test
split must never change after calibration begins.

## Decision
Verified on 2026-10-04 from the authors' repository (commit `c1c1287`, 16 Sep 2026) and
its git history. Items marked **(to fill)** need the downloaded files.

- **Source.** Paper: Tang & Yang, *MultiHop-RAG*, COLM 2024 (arXiv:2401.15391).
  Repository: https://github.com/yixuantt/MultiHop-RAG. The dataset files were removed
  from the repository in commit `33d3ad0`; they are now distributed through the
  Hugging Face dataset `yixuantt/MultiHopRAG` (linked from the repository README).
  `data/download.py` downloads from Hugging Face. · Access date: 2026-10-04.
- **Pinned revision** (verified 2026-10-05): Hugging Face dataset commit
  `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82` (last modified 2024-01-30; public, not
  gated; licence tag `odc-by`). Both files are stored directly in the dataset's git
  repository (not LFS). Config: `data.sources.multihop` (CONFIG_SCHEMA).
- **Licence:** ODC-BY 1.0 (Open Data Commons Attribution), per the repository README.
  Attribution is given in the report; the corpus is not committed (SECURITY §4).
- **Files and SHA-256** (Git LFS pointers of the last repository-hosted version,
  commit `3dd4d4e`, 3 Apr 2025):

  | File | Size (bytes) | SHA-256 |
  |---|---|---|
  | `corpus.json` | 6,785,567 | `20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f` |
  | `MultiHopRAG.json` | 5,171,312 | `03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff` |

  `data/download.py` computes the SHA-256 of the downloaded files. If they match the
  values above, record "identical to repository version"; if not, record the new values
  here and note the difference before any ingestion.
  **Result (2026-10-05): identical to repository version** — the Hugging Face files at the
  pinned revision have exactly the sizes and SHA-256 values above
  (`python -m data.download --dataset multihop`; manifest in
  `data/raw/multihop/download_manifest.json`).
- **Observed counts** (2026-10-05, from the downloaded files). The README states 2,556
  queries with evidence spread over 2–4 documents — confirmed. Corpus documents **609**;
  queries **2,556** (by type: inference **816**, comparison **856**, temporal **583**,
  null **301**). Evidence entries per query: 2 (1,079 queries), 3 (778), 4 (398); null
  queries have none. All 6,084 evidence entries name a `url` present in the corpus.
- **Field names** (verified from the official scripts and, 2026-10-05, from the files):
  queries — `query`, `answer`, `question_type`, `evidence_list`; question-type labels
  `inference_query`, `comparison_query`, `temporal_query`, `null_query`; each evidence
  entry has `fact`, `title`, `url`, `source`, `published_at`, `author`, `category`.
  Corpus items (a JSON list) — `title`, `body` (the article text; 4,770–71,034
  characters, median 7,836), `url`, `source`, `published_at`, `author`, `category`.
  `url` and `title` are each unique across the corpus; there is no separate document id
  field, so `doc_id` falls back to the DATA §3 rule (sha1 of url).
- **Split:** stratified by question type, dev fraction `data.dev_fraction`, seed
  `data.split_seed` (algorithm: DATA §5). Generated 2026-10-05 by
  `python -m data.ingest --dataset multihop` with base.yaml values (0.7, 42, subset 300):
  dev **1,789** (comparison 599, inference 571, null 211, temporal 408), test **767**
  (257 / 245 / 90 / 175), dev subset **300** (101 / 96 / 35 / 68).
  `data/processed/multihop/split.json` SHA-256
  `ddb6a0381c578632360b4eccf2957757b35683fe532569fe0ce1254a2b65fce4`.
  Dev subset SHA-256 (of its sorted task ids joined by `\n`)
  `48b5322a9fde09b39460054949e0a9d8d6be086dd65855e38cfefd512e8f466a`.
  Task ids are `multihop:<index>`, the 0-based position in the checksum-pinned
  `MultiHopRAG.json` (the file has no ids; DATA §1). Two runs produced byte-identical
  files.
- **Official answer-accuracy definition** (`qa_evaluate.py`): if the model output contains
  `The answer to the question is "…"`, the quoted text is taken as the prediction. A
  prediction is correct when the set of whitespace-separated tokens of the lower-cased
  prediction shares at least one token with the lower-cased gold answer; accuracy is
  reported overall and per question type. Null queries are scored the same way against
  their gold answer. Note: any shared token counts (including stop-words), so this metric
  is lenient; decided 2026-10-08: a normalised exact match (lower-case, punctuation and
  articles removed) is reported alongside it (`eval/generator.py`, `exact_match`).
- **Official retrieval metrics** (`retrieval_evaluate.py`, for reference): Hits@4,
  Hits@10, MAP@10 and MRR@10, matching gold `fact` strings as substrings of retrieved
  text, with null queries excluded. Our metrics use the gold chunk map instead (DATA §4,
  EVAL_PROTOCOL §4).

## Consequences
`data.*` split keys Frozen once the split hash is filled in and this ADR is Accepted;
`split.json` committed and never regenerated.

## Reversal condition
Only if the dataset is found to be corrupted or mis-parsed; requires re-running all results.

# ADR-003: MultiHop-RAG source, licence, checksums, split and answer metric

> Status: Proposed · Date: 2026-09-25 · Updated: 2026-10-04 · Author(s): Kabir · Reviewer: Vishal

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
- **Observed counts.** The README states 2,556 queries with evidence spread over 2–4
  documents. Corpus documents ___, queries ___ (by type: inference ___, comparison ___,
  temporal ___, null ___) **(to fill)**.
- **Field names** (verified from the official scripts): queries — `query`, `answer`,
  `question_type` (null queries use the value `null_query`), `evidence_list` (each entry
  has `fact`, plus source-article fields); corpus metadata — `title`, `source`,
  `published_at`. Exact labels of the other three question types and the corpus body
  field **(to fill)**.
- **Split:** stratified by question type, dev fraction `data.dev_fraction`, seed
  `data.split_seed`; `split.json` hash ___ **(to fill)**. Dev subset
  (`data.dev_subset_size`) hash ___ **(to fill)**.
- **Official answer-accuracy definition** (`qa_evaluate.py`): if the model output contains
  `The answer to the question is "…"`, the quoted text is taken as the prediction. A
  prediction is correct when the set of whitespace-separated tokens of the lower-cased
  prediction shares at least one token with the lower-cased gold answer; accuracy is
  reported overall and per question type. Null queries are scored the same way against
  their gold answer. Note: any shared token counts (including stop-words), so this metric
  is lenient; the team decides before E1 whether to also report a normalised exact match.
- **Official retrieval metrics** (`retrieval_evaluate.py`, for reference): Hits@4,
  Hits@10, MAP@10 and MRR@10, matching gold `fact` strings as substrings of retrieved
  text, with null queries excluded. Our metrics use the gold chunk map instead (DATA §4,
  EVAL_PROTOCOL §4).

## Consequences
`data.*` split keys Frozen once the split hash is filled in and this ADR is Accepted;
`split.json` committed and never regenerated.

## Reversal condition
Only if the dataset is found to be corrupted or mis-parsed; requires re-running all results.

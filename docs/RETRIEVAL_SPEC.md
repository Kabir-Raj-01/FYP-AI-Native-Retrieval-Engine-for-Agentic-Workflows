# Retrieval Adapter and Qdrant Backend

> Status: Stable · Owner: Kabir · Last updated: 2026-10-05 · Related: DATA.md, LIBRARY_NOTES.md, ADR-001, ADR-009

## 1. Interface (`engine/retrieval/interfaces.py`)
```python
SearchFilters = Mapping[str, str | int | None]   # payload field -> value

class RetrieverAdapter(ABC):
    def search(self, query: str, mode: RetrievalMode, k: int,
               filters: SearchFilters | None = None, with_vectors: bool = True) -> list[Candidate]: ...
    def health(self) -> bool: ...
```
`filters` was a bare `dict`; it is typed so the interface passes `mypy --strict` and
follows CLAUDE.md rule 2 (Week 7).
`fetch(chunk_ids)` is dropped from the interface (ADR-009 "Also") until a concrete
caller needs it — nothing in the current controlled loop or baselines calls it.
- `search` returns at most `k` candidates ordered by backend score, `rank` 0..k-1.
- `with_vectors=True` returns the **dense** vector of each chunk in `Candidate.vector`
  (needed for novelty / gap coverage without re-embedding).
- Raises `BackendError` on timeout, connection failure or malformed response. Never
  returns partial results silently.
- The adapter is stateless; it knows nothing about tasks, budgets or caches.

This interface is the only thing the engine knows about the backend. Adding another
backend (stretch) = one new class + a factory entry.

Implementation: `engine/retrieval/qdrant_adapter.py` (`QdrantAdapter`, built by
`factory.build_retriever(cfg, embedder, sparse_encoder)`). Client failures
(`ResponseHandlingException`, `UnexpectedResponse`, `httpx.HTTPError`, `OSError`) become
`BackendError`; a point without `chunk_id`/`doc_id`/`text` in its payload, or without its
dense vector when `with_vectors=True`, is a malformed response (`BackendError`). `k < 1`
raises `ValueError`. `health()` = the collection exists (False on any backend error).

### Write side: `IndexWriter` (ingestion only; ADR-001)
```python
class IndexPoint(BaseModel):          # transport type of this interface, not task state
    chunk_id: str; doc_id: str; text: str
    metadata: dict[str, str | int | None] = {}
    dense: list[float]; sparse: SparseVector

class IndexWriter(ABC):
    def ensure_collection(self) -> None: ...   # create, or verify an existing one
    def existing_ids(self) -> set[str]: ...    # chunk_ids already stored (resume)
    def upsert(self, points: Sequence[IndexPoint]) -> None: ...
```
`engine/retrieval/qdrant_index.py` (`QdrantIndexWriter`, `factory.build_index_writer`).
`ensure_collection` refuses (`ConfigError`) an existing collection whose dense size,
distance or sparse IDF modifier differ — an index change needs a new collection name.
Upserts go in batches of `qdrant.upsert_batch_size`. Only `engine/retrieval` imports
`qdrant_client` (import-boundary test).

## 2. Qdrant collection layout
| Item | Value (config key) |
|---|---|
| Collection | `qdrant.collection` (one per dataset, e.g. `multihop_v1`) |
| Dense named vector | `qdrant.dense_vector_name` (`dense`), size = `embedding.dimension`, distance Cosine |
| Sparse named vector | `qdrant.sparse_vector_name` (`sparse`), BM25 with `modifier=idf` — required: fastembed's BM25 carries only the term-frequency part and expects Qdrant to apply IDF (verified, LIBRARY_NOTES) |
| HNSW | collection-level `qdrant.hnsw.m`, `qdrant.hnsw.ef_construct`; search-time `qdrant.search_ef` (dense mode) |
| Point ID | UUIDv5(namespace = `qdrant.id_namespace`, name = chunk_id). **Qdrant point IDs must be unsigned ints or UUIDs** — the string chunk_id is stored in the payload. |
| Payload | `chunk_id, doc_id, text` plus the chunk's metadata flattened: `title, source, url, published_at, author, category, dataset, char_start, char_end, n_tokens` (DATA §3 step 7) |
| Payload indexes | keyword index on `chunk_id`, `doc_id` (needed by filters; also ready for a future `fetch`) |

Collection name is versioned: any change to embedding model, chunking or sparse method
creates a new collection name (e.g. `multihop_v2`) — never overwrite a collection that
reported results depend on.

## 3. Modes
| Mode | Execution |
|---|---|
| `dense` | Query embedded with `Embedder.embed_queries`; HNSW search on the dense vector. |
| `sparse` | Query encoded with the same sparse encoder used at ingestion; search on the sparse vector. |
| `hybrid` | Qdrant Query API: prefetch `retrieval.hybrid_prefetch_k` from dense **and** sparse, fuse with Reciprocal Rank Fusion (`retrieval.hybrid_fusion = rrf`), return top k. |

Action → mode mapping lives in the controller (ACTION_POLICY §5): SPARSE_HYBRID uses
`sparse` if `current_need_type == lexical`, else `hybrid`.

Scores are **not comparable across modes** (cosine vs BM25 vs RRF). Nothing downstream
may threshold on `retrieval_score`; relevance thresholds use the cross-encoder only.

## 4. Latency and instrumentation
`search` measures wall-clock latency (embedding + request) and returns it via the calling
node, which records it in `RetrievalRecord.latency_ms` and the TraceStep. Emits
`retrieval.call` events (OBSERVABILITY.md) whose `latency_ms` is the search alone, never
reranking; on the direct path `DirectResult.latency_ms` is search + rerank, with the rerank
part in `DirectResult.rerank_latency_ms` (ADR-012 A1).

## 5. Embedder (`engine/embeddings/`)
```python
class Embedder(ABC):
    dimension: int
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...   # applies embedding.query_prefix
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...  # applies embedding.passage_prefix
```
- One model (`embedding.model_name`) for indexing, queries, cache similarity, gap
  descriptions and all similarity signals. Vectors L2-normalised so dot = cosine.
- Query/passage prefixes matter for E5/BGE-family models; set them in config per the
  model card and record them in ADR-001.
- In-process LRU cache (`embedding.lru_size`) for repeated query/gap texts.
- Gap descriptions and queries are embedded as *queries*; chunks as *passages*.
- Implementation: `engine/embeddings/sentence_transformer.py` (sentence-transformers,
  `embedding.model_name` at the pinned `embedding.revision`; `factory.build_embedder`).
  The constructor fails with `ConfigError` if the model's dimension differs from
  `embedding.dimension`. CPU throughput on a 4-core laptop (i7-1185G7): ≈ 0.9–1.25
  chunks/s for ~344-token chunks — the full MultiHop-RAG index (4,441 chunks) took 59 min;
  direct dense retrieval (query embedding + search, k = 20) has a median of 104 ms (NFR-P1).

### Sparse encoder (`engine/embeddings/interfaces.py`; ADR-001)
```python
class SparseVector(BaseModel):        # transport type, not task state
    indices: list[int]; values: list[float]

class SparseEncoder(ABC):
    def encode_documents(self, texts: list[str]) -> list[SparseVector]: ...
    def encode_queries(self, texts: list[str]) -> list[SparseVector]: ...
```
`engine/embeddings/bm25.py` (`FastEmbedBM25Encoder`, `factory.build_sparse_encoder`):
fastembed `Qdrant/bm25` with `sparse.k1`, `sparse.b`, `sparse.avg_len`. fastembed cannot
pin a revision, so the pinned `sparse.revision` snapshot is downloaded with
`huggingface_hub.snapshot_download` into `paths.model_cache_dir` and passed as
`specific_model_path`. Documents get term-frequency weights, queries weight 1 per term;
IDF is applied by Qdrant.

## 6. Tests
- Unit: a `FakeRetriever` (TESTING.md) implements the interface over an in-memory corpus.
  `QdrantAdapter` and `QdrantIndexWriter` are unit-tested against Qdrant's in-memory client
  (`QdrantClient(":memory:")`, exact search) with `FakeEmbedder` and `FakeSparseEncoder`
  (`tests/unit/retrieval/test_qdrant.py`).
- Integration (needs local Qdrant, `qdrant.exe`): create a tiny collection from
  `tests/fixtures/mini_corpus.jsonl`; assert dense/sparse/hybrid each return the expected
  top-1 for three handcrafted queries (one semantic, one exact-entity, one mixed);
  `search` round-trips payloads and vectors; timeouts raise `BackendError`.
  Implemented in `tests/integration/test_qdrant_live.py` with the real pinned models; an
  unreachable server (connection refused) stands in for the timeout case.

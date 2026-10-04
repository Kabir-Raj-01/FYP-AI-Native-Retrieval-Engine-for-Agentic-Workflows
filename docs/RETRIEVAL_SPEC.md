# Retrieval Adapter and Qdrant Backend

> Status: Stable · Owner: Kabir · Last updated: 2026-10-04 · Related: DATA.md, LIBRARY_NOTES.md, ADR-009

## 1. Interface (`engine/retrieval/interfaces.py`)
```python
class RetrieverAdapter(ABC):
    def search(self, query: str, mode: RetrievalMode, k: int,
               filters: dict | None = None, with_vectors: bool = True) -> list[Candidate]: ...
    def health(self) -> bool: ...
```
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

## 2. Qdrant collection layout
| Item | Value (config key) |
|---|---|
| Collection | `qdrant.collection` (one per dataset, e.g. `multihop_v1`) |
| Dense named vector | `qdrant.dense_vector_name` (`dense`), size = `embedding.dimension`, distance Cosine |
| Sparse named vector | `qdrant.sparse_vector_name` (`sparse`), BM25-style with IDF modifier (verify in LIBRARY_NOTES) |
| HNSW | `qdrant.hnsw.m`, `qdrant.hnsw.ef_construct`; search-time `qdrant.search_ef` |
| Point ID | UUIDv5(namespace = `qdrant.id_namespace`, name = chunk_id). **Qdrant point IDs must be unsigned ints or UUIDs** — the string chunk_id is stored in the payload. |
| Payload | `chunk_id, doc_id, text, title, source, url, published_at, char_start, char_end, dataset` |
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
`retrieval.call` events (OBSERVABILITY.md).

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

## 6. Tests
- Unit: a `FakeRetriever` (TESTING.md) implements the interface over an in-memory corpus.
- Integration (needs local Qdrant, `qdrant.exe`): create a tiny collection from
  `tests/fixtures/mini_corpus.jsonl`; assert dense/sparse/hybrid each return the expected
  top-1 for three handcrafted queries (one semantic, one exact-entity, one mixed);
  `search` round-trips payloads and vectors; timeouts raise `BackendError`.

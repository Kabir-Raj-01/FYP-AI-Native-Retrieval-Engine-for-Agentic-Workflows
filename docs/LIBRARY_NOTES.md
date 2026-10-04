# Library Notes (pinned versions + verified usage)

> Status: Living document · Owner: All · Last updated: 2026-10-04

Coding assistants often write code for an older version of a library than the one
installed. This file records the **exact pinned version** and **usage verified by running
it**. Rules:
1. Fill a "Verified" block only after the code ran against the pinned version.
2. Link the official doc page you used.
3. When a confusing error is solved, record the fix under "Gotchas" in the same PR.
4. Do not paste snippets from memory (human or AI) into "Verified" blocks.

## Pinned versions
| Package | Version | Pinned on | Official docs |
|---|---|---|---|
| python | 3.11.x | | https://docs.python.org/3.11/ |
| qdrant-client | | | https://qdrant.tech/documentation/ |
| Qdrant server, Windows binary (`QDRANT_VERSION`) | | | https://github.com/qdrant/qdrant/releases |
| fastembed | | | https://qdrant.github.io/fastembed/ |
| sentence-transformers | | | https://www.sbert.net/ |
| langgraph | | | https://langchain-ai.github.io/langgraph/ |
| fastapi | | | https://fastapi.tiangolo.com/ |
| pydantic | 2.x | | https://docs.pydantic.dev/latest/ |
| httpx | | | https://www.python-httpx.org/ |
| anthropic | | | https://docs.claude.com |
| openai | | | https://platform.openai.com/docs |
| rapidfuzz | | | https://rapidfuzz.github.io/RapidFuzz/ |
| structlog | | | https://www.structlog.org/ |
| tenacity | | | https://tenacity.readthedocs.io/ |
| python-dotenv | | | https://saurabh-kumar.com/python-dotenv/ |
| pytest / hypothesis / ruff / mypy | | | |

## Qdrant — things to verify and record
- [ ] Creating a collection with a **named dense vector** and a **named sparse vector** on
      the same collection. Verified snippet:
- [ ] Sparse vectors for BM25 — whether an IDF modifier must be set on the sparse vector
      config when using fastembed's BM25 model. Verified snippet:
- [ ] Hybrid search via the Query API (`query_points` with `prefetch` for dense and sparse,
      fusion RRF). Verified snippet:
- [ ] Returning vectors with results (`with_vectors` for a specific named vector) and
      payload. Verified snippet:
- [ ] Point ids: UUID strings accepted; fetch by payload `chunk_id` with a keyword index
      (or `retrieve` by UUID). Verified snippet:
- Gotchas:

## fastembed (sparse BM25)
- [ ] Encoding documents vs queries for BM25 (separate methods?) and output format →
      Qdrant `SparseVector(indices, values)`. Verified snippet:
- Gotchas:

## sentence-transformers
- [ ] Embedder: `SentenceTransformer(model).encode(texts, normalize_embeddings=True)`;
      where the BGE query instruction goes. Verified snippet:
- [ ] CrossEncoder: does `predict` return logits or probabilities for the chosen reranker
      (depends on model and library version)? Relevance must end in [0,1]. Verified:
- Gotchas:

## LangGraph
- [ ] `StateGraph` with a Pydantic model as state; nodes returning partial updates;
      conditional edges; compile without a checkpointer (ADR-002). Verified snippet:
- [ ] Setting `recursion_limit` at invoke time. Verified snippet:
- Gotchas:

## FastAPI + Pydantic v2
- Use v2 APIs: `model_config = ConfigDict(...)`, `field_validator`, `model_validator`,
  `model_dump()`, `model_validate()` — not v1 `class Config`, `@validator`, `.dict()`.
- [ ] Custom exception handlers returning the API_CONTRACT error body. Verified snippet:
- [ ] Lifespan handler that builds the Engine once at startup. Verified snippet:
- Gotchas:

## LLM providers
- [ ] Anthropic: current model names, message format, usage fields, structured output /
      tool-use option. Check https://docs.claude.com — do not rely on remembered model
      strings. Verified snippet:
- [ ] OpenAI-compatible (Ollama/LM Studio): JSON mode / schema support of the chosen local
      model and server version; usage fields present? Verified snippet:
- Gotchas:

## rapidfuzz
- [ ] `fuzz.partial_ratio` scale 0–100 (thresholds in config use this scale). Verified:

# Library Notes (pinned versions + verified usage)

> Status: Living document · Owner: All · Last updated: 2026-10-05

Coding assistants often write code for an older version of a library than the one
installed. This file records the **exact pinned version** and **usage verified by running
it**. Rules:
1. Fill a "Verified" block only after the code ran against the pinned version.
2. Link the official doc page you used.
3. When a confusing error is solved, record the fix under "Gotchas" in the same PR.
4. Do not paste snippets from memory (human or AI) into "Verified" blocks.

## Pinned versions
"Installed on" = the version resolved by `pip install -e ".[dev]"` on Kabir's machine
(Windows 11, python.org 3.11.9); these become pins once `requirements.lock` is committed
(SETUP §6).

| Package | Version | Installed on | Official docs |
|---|---|---|---|
| python | 3.11.9 (python.org, Authenticode-signed) | 2026-10-04 | https://docs.python.org/3.11/ |
| qdrant-client | 1.19.1 | 2026-10-04 | https://qdrant.tech/documentation/ |
| Qdrant server, Windows binary (`QDRANT_VERSION`) | **v1.19.1** (`qdrant-x86_64-pc-windows-msvc.zip`, SHA-256 `9b6f69bd85f6abed4bc13f943099f55c6ffd55f5dd90388635320d8fbb569eb0`, matches the release's published digest) | 2026-10-05 | https://github.com/qdrant/qdrant/releases |
| fastembed | 0.8.1 | 2026-10-04 | https://qdrant.github.io/fastembed/ |
| sentence-transformers | 6.1.0 (torch 2.14.1) | 2026-10-04 | https://www.sbert.net/ |
| langgraph | 1.2.12 (langsmith 0.14.4, uuid-utils **0.17.0**, see gotchas) | 2026-10-05 | https://langchain-ai.github.io/langgraph/ |
| fastapi | 0.142.2 | 2026-10-04 | https://fastapi.tiangolo.com/ |
| pydantic | 2.13.5 (pydantic-core 2.46.5) | 2026-10-04 | https://docs.pydantic.dev/latest/ |
| httpx | 0.28.1 | 2026-10-04 | https://www.python-httpx.org/ |
| anthropic | 1.11.0 | 2026-10-04 | https://docs.claude.com |
| openai | 3.24.0 | 2026-10-04 | https://platform.openai.com/docs |
| rapidfuzz | 3.14.6 | 2026-10-04 | https://rapidfuzz.github.io/RapidFuzz/ |
| structlog | 26.1.0 | 2026-10-04 | https://www.structlog.org/ |
| tenacity | 9.1.4 | 2026-10-04 | https://tenacity.readthedocs.io/ |
| python-dotenv | 1.2.4 | 2026-10-04 | https://saurabh-kumar.com/python-dotenv/ |
| pyyaml / numpy | 6.0.3 / 2.4.6 | 2026-10-04 | |
| pytest / pytest-cov / hypothesis / ruff / mypy | 9.1.1 / 7.1.0 / 6.168.3 / 0.16.10 / 2.4.0 | 2026-10-04 | |
| pre-commit / detect-secrets | 4.6.2 / 1.5.0 | 2026-10-04 | |

## Windows environment — gotchas (verified 2026-10-04)
- **Smart App Control blocks some unsigned native DLLs.** With Smart App Control on, the
  uv-managed CPython 3.11.16 failed `import sqlite3` ("DLL load failed … An Application
  Control policy has blocked this file"); `sqlite3.dll`/`_sqlite3.pyd` are unsigned. That
  breaks pre-commit, `pytest --cov` (coverage.py stores data in SQLite) and every SQLite
  store. Fix: build `.venv` from the signed python.org installer (`py -3.11`; per-user via
  `winget install --id Python.Python.3.11 -e --scope user`). Do not disable Smart App
  Control (it cannot be re-enabled without reinstalling Windows).
- **`import langgraph.graph` failed on the same machine** with `uuid-utils` 0.17.1 (pulled
  in by langsmith/langchain-core, which accept `>=0.12.0,<1.0`): its `_uuid_utils` `.pyd` is
  blocked by Smart App Control. Tested 2026-10-05, each version in a fresh venv: 0.12.0,
  0.13.0, 0.14.0, 0.14.1, 0.15.0, 0.16.0, 0.16.1, 0.16.2, **0.17.0** and 1.0.0 load;
  only 0.17.1 is blocked. `requirements.lock` holds 0.17.0 (newest allowed that loads).
  Do not upgrade it to 0.17.1. Other wheels' native modules (pydantic-core, numpy, torch,
  onnxruntime, rapidfuzz, PyYAML) load fine.
- **`py` can resolve to the Python install manager's app alias**
  (`%LOCALAPPDATA%\Microsoft\WindowsApps\py.exe`), which may fail with "The file cannot be
  accessed by the system"; call the python.org interpreter by its path instead (SETUP §7).
- **langsmith registers a pytest plugin** (`langsmith_plugin`) that imports `uuid_utils` at
  pytest start-up; `pyproject.toml` disables it with `addopts = "-p no:langsmith_plugin"`.
- **ruff 0.16 formats Python code blocks inside Markdown**: `ruff format --check .` reformats
  the spec snippets. `pyproject.toml` sets `extend-exclude = ["*.md"]` (the pre-commit hook
  already restricted ruff to Python files).

## Qdrant — things to verify and record
- Server start (SETUP §3) verified 2026-10-05: `qdrant.exe` v1.19.1 is **not
  Authenticode-signed**, but Smart App Control lets it run on Kabir's machine; it answers
  `{"title":"qdrant - vector search engine","version":"1.19.1",...}` on 127.0.0.1:6333.
- [x] Creating a collection with a **named dense vector** and a **named sparse vector** on
      the same collection. Verified 2026-10-05 (qdrant-client 1.19.1, server 1.19.1):
      ```python
      c = QdrantClient(url="http://127.0.0.1:6333", timeout=10)
      c.create_collection(
          name,
          vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
          sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
      )
      c.get_collection(name).config.params.sparse_vectors["sparse"].modifier  # -> idf
      ```
- [ ] Sparse vectors for BM25 — whether an IDF modifier must be set on the sparse vector
      config when using fastembed's BM25 model. (Setting `Modifier.IDF` is accepted — above —
      but whether fastembed's BM25 output expects it is still unverified.) Verified snippet:
- [ ] Hybrid search via the Query API (`query_points` with `prefetch` for dense and sparse,
      fusion RRF). Verified snippet:
- [ ] Returning vectors with results (`with_vectors` for a specific named vector) and
      payload. Verified snippet:
- [ ] Point ids: UUID strings accepted; fetch by payload `chunk_id` with a keyword index
      (or `retrieve` by UUID). Verified snippet:
- Gotchas:
  - `qdrant.exe` writes a `.qdrant-initialized` marker into its **working directory**
    (the repo root when started per SETUP §3); it is gitignored.

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
- [x] `StateGraph` with a Pydantic model as state; nodes returning partial updates;
      conditional edges; compile without a checkpointer (ADR-002). Verified snippet
      (2026-10-05, 1.2.12; flat state — nested GraphState below):
      ```python
      class S(BaseModel):
          n: int = 0

      def inc(s: S) -> dict:                           # receives the model, returns a partial dict
          return {"n": s.n + 1}

      g = StateGraph(S)
      g.add_node("inc", inc)
      g.add_edge(START, "inc")
      g.add_conditional_edges("inc", lambda s: END if s.n >= 3 else "inc")
      app = g.compile()                                # no checkpointer
      app.invoke(S(), config={"recursion_limit": 10})  # -> {'n': 3}
      ```
- [x] **Nested `GraphState` (ADR-009 §2) verified 2026-10-05 on 1.2.12** with the real
      `GraphState`/`RetrievalState`/`IterationScratch` models, pinned by
      `tests/unit/graph/test_langgraph_graphstate.py` (read from the installed source:
      `langgraph/graph/state.py`, `_pick_mapper`/`_coerce_state`/`_get_updates`):
  - Each top-level field (`state`, `scratch`) is one LastValue channel; every node gets
    `GraphState(**channels)` with typed nested models and working computed fields. A
    plan → … → persist loop with a conditional edge and `{"scratch": IterationScratch()}`
    in `persist` behaves as ARCHITECTURE §4 describes (one-node invoke with a populated
    state: ~0.5 ms).
  - A returned field **replaces** the field: `{"scratch": {"scored": [...]}}` drops the
    other scratch fields. Return full values: `gs.scratch.model_copy(update={...})`,
    `gs.state.model_copy(update={...})`.
  - Return values are validated only when the **next** node's input is built
    (`ValidationError`, loc `state`); a bad value from the last node reaches `invoke()`'s
    output unvalidated → validate the output (`GraphState.model_validate(out)`).
  - Returned keys that are not GraphState fields are **silently dropped** (a typo such as
    `"scrach"` is not an error).
  - Objects pass by reference — no copy, no revalidation — starting with the caller's own
    input (`GraphState(state=s).state is s`, pydantic `revalidate_instances="never"`). An
    in-place mutation leaks to later nodes and to the caller; take `prev` for
    `check_invariants` as `state.model_copy(deep=True)` before `invoke()`.
- [x] Setting `recursion_limit` at invoke time: `invoke(..., config={"recursion_limit": n})`
      (smoke run above, 2026-10-05).
- Gotchas:
  - `invoke()` returns a **dict** of the state's fields, not an instance of the Pydantic
    state model — validate it back (`GraphState.model_validate(...)`) where a model is needed.

## FastAPI + Pydantic v2
- Use v2 APIs: `model_config = ConfigDict(...)`, `field_validator`, `model_validator`,
  `model_dump()`, `model_validate()` — not v1 `class Config`, `@validator`, `.dict()`.
- [x] **Verified 2026-10-04 on pydantic 2.13.5** by a minimal experiment (run, not recalled;
      source: behaviour of the installed package, not a doc page) and by
      `tests/unit/state/`:
  - A `@field_serializer("vector")` reading `info.context` runs for `model_dump()` and
    `model_dump_json()`; returning `None` keeps the key (`"vector": null`) — a field
    serializer cannot drop a key. The context passed to `model_dump(context=...)` reaches
    nested models.
  - `model_dump(exclude=...)` drops `@computed_field` keys, including through nested
    specs (`{"field": {"computed": True}}`, list indexes `{0: {...}}`).
  - With `extra="forbid"`, `model_validate()` of a payload containing a computed-field key
    fails with `extra_forbidden` (why `dump_with_vectors` strips them; ADR-009 §4a).
  - Assigning to a `@computed_field` property raises `AttributeError: property '…' of
    '…' object has no setter`, also with `validate_assignment=True`.
  - A model-instance default (`scratch: IterationScratch = IterationScratch()`) and `= []` /
    `= {}` defaults are copied per instance, not shared.
  - `AwareDatetime` rejects naive datetimes.
- Gotchas:
  - mypy 2.4 needs `@computed_field  # type: ignore[prop-decorator]` above `@property`
    (confirmed needed: `--strict` reports unused ignores and reports none).
  - A field named `json` would shadow `BaseModel.json`; use `json_: bool =
    Field(alias="json")`. Do **not** add `populate_by_name=True`: it also accepts the key
    `json_`, which then slips past `extra="forbid"`.

## python-dotenv
- [x] Verified 2026-10-04 on 1.2.4 (installed signature): `load_dotenv(override=False)`
      by default (existing variables win); `find_dotenv(usecwd=True)` searches upwards from
      the working directory. `engine/config.py` uses
      `load_dotenv(find_dotenv(usecwd=True), override=False)`; covered by
      `tests/unit/test_config.py::test_dotenv_is_loaded_but_existing_env_wins`.
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

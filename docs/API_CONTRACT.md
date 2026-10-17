# API Contract

> Status: Frozen (contract — change only via ADR) · Owner: Vishal · Last updated: 2026-10-17 (ADR-015)
> Implementation: `engine/api/` (routers + `schemas.py`), `client/retrieval_client.py`.
> `BudgetRequest` bounds amended by ADR-009 §8; request limits moved to config by ADR-010.

Base path `/v1`. JSON only. Models referenced are from STATE_SPEC.md; `vector` fields are
never serialised. Interactive docs at `/docs` (OpenAPI). The server binds to
`api.host` (default `127.0.0.1`) — see SECURITY.md before exposing it.

## Endpoints
| Method & path | Purpose | Success |
|---|---|---|
| `GET /v1/health` | Liveness + backend check | 200 `{status, qdrant: bool, version, config_hash}` |
| `POST /v1/tasks` | Create task, persist S_0 | 201 `CreateTaskResponse` |
| `POST /v1/tasks/{task_id}/step` | Run one controlled iteration | 200 `StepResponse` |
| `POST /v1/tasks/{task_id}/run` | Iterate until stop | 200 `EvidenceBundle` |
| `GET /v1/tasks/{task_id}/result` | Bundle of a stopped task | 200 `EvidenceBundle` |
| `GET /v1/tasks/{task_id}/state` | Full state (debug) | 200 `RetrievalState` |
| `GET /v1/tasks/{task_id}/trace` | All TraceSteps | 200 `list[TraceStep]` |
| `POST /v1/retrieve/direct` | Controller-OFF retrieval | 200 `DirectResponse` |

## Schemas (`engine/api/schemas.py`)
```python
class BudgetRequest(BaseModel):          # every field optional → budget_defaults.*
    max_steps: int | None = Field(default=None, ge=1)
    max_retrieval_calls: int | None = Field(default=None, ge=1)
    max_llm_calls: int | None = Field(default=None, ge=0)   # 0 = no-LLM ablation
    max_tokens: int | None = Field(default=None, ge=1)

class CreateTaskRequest(BaseModel):
    goal: str = Field(min_length=1)      # length bounds: api.goal_min_chars..api.goal_max_chars (router, ADR-010)
    budget: BudgetRequest = BudgetRequest()
    task_id: str | None = None           # optional caller-supplied id (eval uses dataset ids)

class CreateTaskResponse(BaseModel):
    task_id: str
    state: RetrievalState

class StepResponse(BaseModel):
    task_id: str
    step: int
    action: Action
    done: bool
    stop_reason: StopReason | None
    new_evidence: list[EvidenceItem]
    open_gaps: list[EvidenceGap]
    budget: Budget

class DirectRequest(BaseModel):
    query: str = Field(min_length=1)     # max length: api.query_max_chars (router, ADR-010)
    mode: RetrievalMode = RetrievalMode.DENSE
    k: int | None = Field(default=None, ge=1)   # None → retrieval.default_k; > retrieval.max_k → 422
    rerank: bool = False                  # relevance-only cross-encoder rerank

class DirectResponse(BaseModel):
    candidates: list[Candidate]           # without vectors
    relevances: list[float] | None        # present when rerank=true
    latency_ms: float
```

## Behaviour
- Optional bearer token (ADR-015, SRS NFR-S3): when the server's `ENGINE_API_TOKEN` is set,
  every `/v1/*` request must send `Authorization: Bearer <token>`; a missing or wrong header
  → 401 `UNAUTHORIZED` with `WWW-Authenticate: Bearer`, before the request reaches the
  engine. `/docs`, `/redoc` and `/openapi.json` stay open. Unset or empty (the default): no
  check.
- `POST /tasks` with an existing `task_id` → 409 `TASK_EXISTS`.
- Request limits are checked against config in the router (ADR-010): goal length within
  `api.goal_min_chars`–`api.goal_max_chars`, query length ≤ `api.query_max_chars`,
  `k ≤ retrieval.max_k`; a violation → 422 `VALIDATION_ERROR`. An omitted `k` uses
  `retrieval.default_k`.
- `/step` and `/run` on a task with `status != active` → 409 `TASK_NOT_ACTIVE`
  (use `/result`). `/result` on an active task → 409 `TASK_NOT_ACTIVE`.
- Concurrent `/step` calls on the same task → the second gets 409 `TASK_CONFLICT`.
- A failed step leaves the stored state unchanged (ARCHITECTURE §8).
- `/run` has a server-side timeout `api.run_timeout_s`; on timeout 504 `RUN_TIMEOUT`,
  and the state reflects the last completed step (the caller may continue with `/step`).

## Errors
All errors: `{"error": {"code": "<CODE>", "message": "...", "task_id": "...|null"}}`
| HTTP | Code |
|---|---|
| 401 | `UNAUTHORIZED` (only when `ENGINE_API_TOKEN` is set: missing or wrong bearer token) |
| 404 | `TASK_NOT_FOUND` |
| 409 | `TASK_EXISTS`, `TASK_NOT_ACTIVE`, `TASK_CONFLICT` |
| 422 | `VALIDATION_ERROR` (request body) |
| 502 | `LLM_ERROR` |
| 503 | `BACKEND_ERROR` (Qdrant unreachable/timeout) |
| 504 | `RUN_TIMEOUT` |
| 429 | `RUN_BUDGET_EXCEEDED` (per-run LLM cap hit) |
| 500 | `INTERNAL_ERROR` |

## Example
```powershell
$task = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/tasks -ContentType 'application/json' `
  -Body '{"goal": "Which company that acquired a chip startup in 2023 was founded earlier?"}'
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/tasks/$($task.task_id)/run"
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/retrieve/direct -ContentType 'application/json' `
  -Body '{"query": "chip startup acquisition 2023", "mode": "hybrid", "k": 10}'
```

## Python client (`client/retrieval_client.py`)
Imports request/response models from `engine.state.models` / `engine.api.schemas`
(never redeclares them) and re-exports them as `client.models`, so host apps and the demo
agent import only `client` and `client.models`. Uses `httpx` with timeout `client_timeout_s`. Maps error codes to
exceptions (`TaskNotFound`, `TaskNotActive`, `BackendUnavailable`, `Unauthorized`, ...).
`RetrievalClient(base_url, token=...)` sends the bearer token when the server requires one.
```python
client = RetrievalClient("http://127.0.0.1:8000")
task = client.create_task(goal="...", budget=BudgetRequest(max_retrieval_calls=4))
bundle = client.run(task.task_id)             # or: while not (r := client.step(id)).done: ...
direct = client.retrieve_direct("...", mode=RetrievalMode.HYBRID, k=10)
```

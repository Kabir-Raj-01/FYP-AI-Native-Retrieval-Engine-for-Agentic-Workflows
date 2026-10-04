# LLM Client

> Status: Stable · Owner: Vishal · Last updated: 2026-10-04 · Related: SECURITY.md, ADR-004 (model choices, pending), ADR-009

## 1. Purpose
One place for every LLM call in the project: the planner, the optional evidence
verifier, and the eval harness's answer generator. It provides structured output,
response caching, token/cost accounting and hard caps. No other module imports a
provider SDK.

## 2. Roles
| Role | Used by | Config section | Notes |
|---|---|---|---|
| `planner` | engine/planner | `llm.planner` | structured output, temperature 0 |
| `verifier` | engine/evidence (optional) | `llm.verifier` | structured `{supports: bool}`; disabled by default |
| `generator` | eval/generator.py (outside the engine) | `llm.generator` | text output, temperature 0; identical for all systems |
The generator role exists in the same client so costs and caching are uniform, but the
engine itself never calls it.

## 3. Interface (`engine/llm/interfaces.py`)
`LLMResult` and `Message` live here, not in `engine/state/models.py` (ADR-009 §10) —
they are LLM-transport types, used only by `engine/llm` and `engine/planner`.
```python
class LLMResult(BaseModel, Generic[T]):
    value: T
    input_tokens: int
    output_tokens: int
    cost_usd: float          # estimated from llm.prices; 0 when served from cache
    cached: bool
    latency_ms: float
    model: str

class LLMClient(ABC):
    def structured(self, role: str, messages: list[Message], schema: type[T]) -> LLMResult[T]: ...
    def text(self, role: str, messages: list[Message]) -> LLMResult[str]: ...
```
`Message = {"role": "system" | "user" | "assistant", "content": str}`.
The planner's `schema` for `role="planner"` is `PlanLLMOutput` (STATE_SPEC §3) — a
narrower model than `PlanResult`, with no `mode` field and no length/count constraints
baked into the schema (ADR-009 §10); those bounds are enforced by
`engine/planner/validator.py` from `planner.*` config instead.

## 4. Providers (`engine/llm/providers/`)
| `provider` value | Covers |
|---|---|
| `anthropic` | Claude models via the Anthropic API |
| `openai_compatible` | OpenAI API and any OpenAI-compatible local server (Ollama, LM Studio) via `base_url` |
| `fake` | tests only (TESTING.md) |
Model name strings, base URLs and API-key env var names come from config/`.env`. Check
current model names in the provider's documentation when filling in ADR-004 — do not
copy remembered model strings.

## 5. Structured output
1. Build a JSON schema from the Pydantic model; include it in the system message with the
   instruction to return only JSON. If the provider supports native schema-constrained
   output or tool/function calling, the provider class may use it (verify in LIBRARY_NOTES).
2. Parse: strip code fences / leading prose, `json.loads`, then `schema.model_validate`.
3. On parse/validation failure, raise `LLMOutputError` (the planner counts this as a
   rejected attempt with reason `SCHEMA`). No silent repair beyond fence stripping.

## 6. Transport reliability
Retries only for transport errors (timeouts, connection errors, HTTP 429/5xx): up to
`llm.<role>.max_transport_retries` with exponential backoff + jitter (tenacity).
Timeout `llm.<role>.timeout_s`. After retries → `LLMError`.

## 7. Response cache
- SQLite at `paths.llm_cache_db`, enabled by `llm.cache_enabled`.
- Key = sha256 of canonical JSON of (provider, model, role, temperature, max tokens,
  messages, schema JSON, `llm.cache_namespace`).
- **Budget accounting counts logical calls, cached or not.** A cache hit still increments
  `budget.llm_calls_used` and adds the recorded token usage to `budget.tokens_used`, so
  experiment behaviour does not depend on cache state. `cost_usd` actually spent is 0 for
  hits; the estimated cost is recorded separately as `cost_usd_equivalent`.
- `llm.cache_namespace` lets repeated-run experiments (EVAL_PROTOCOL §6) bypass earlier
  cached responses by using `repeat-<n>`.

## 8. Caps and accounting
- Per task: enforced by the caller through `Budget` (the planner checks
  `llm_calls_used < max_llm_calls` before every attempt).
- Per run: `llm.max_calls_per_run` and `llm.max_cost_usd_per_run` enforced inside the
  client; exceeding raises `LLMBudgetExceeded`, which aborts the run cleanly (manifest
  records the abort). This protects the team's API credit.
- Token counts come from provider usage fields; if a local server reports none, estimate
  as `ceil(chars / 4)` and flag `estimated_tokens=true` in the event.
- Every call emits `llm.call` (role, model, tokens, cost, cached, latency, attempt).

## 9. Message construction rules
System message: instructions and schema only. User message: task data; any retrieved or
corpus-derived text wrapped as described in SECURITY.md §3. Never put retrieved text in
the system message.

## 10. Tests
Fake provider returning scripted strings; fence stripping; invalid JSON → `LLMOutputError`;
transport retry then success; retry exhaustion → `LLMError`; cache hit returns identical
value, increments logical-call counters, zero actual cost; per-run cap raises.

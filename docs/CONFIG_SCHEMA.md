# Configuration Schema

> Status: Stable · Owner: All (each section's owner listed) · Last updated: 2026-10-04
> Defaults: `configs/base.yaml`. Loader/validator: `engine/config.py` (`EngineConfig`, Pydantic, `extra="forbid"`).
> Budget bounds and `planner.rationale_max_chars` added by ADR-009 (§8, §10).

## 1. Loading and precedence
1. `configs/base.yaml` (all keys, defaults)
2. `configs/experiments/<name>.yaml` (deep-merged overrides + the `experiment` section)
3. Environment variables — **secrets and machine-specific values only** (API keys,
   `QDRANT_URL`, `ENGINE_CONFIG`). Never thresholds or weights. `engine/config.py`
   calls `load_dotenv()` (python-dotenv) first, so values in `.env` are loaded;
   variables already set in the environment take precedence over `.env`.
4. `--set key=value` CLI overrides — for local debugging only; `eval/run.py` refuses to
   mark a run as reportable if any `--set` was used.

The fully resolved config is saved as `config.resolved.yaml` in every run folder, and its
hash goes into the manifest. The server loads the file named by `ENGINE_CONFIG`
(default `configs/base.yaml`).

Validation at load: unknown keys rejected; `controller.weights` and
`evidence.gain_weights` and `controller.gain_subweights` must be non-negative and sum
to 1 (±1e-6); probabilities/thresholds in [0,1]; `retrieval.default_k ≤ retrieval.max_k`;
`ambiguous_band.low < high`; `budget_defaults.max_steps ≥ 1`,
`budget_defaults.max_retrieval_calls ≥ 1`, `budget_defaults.max_llm_calls ≥ 0`,
`budget_defaults.max_tokens ≥ 1` (ADR-009 §8 — `Budget` mirrors these bounds);
rapidfuzz thresholds (`int 0–100`) in [0,100]. Any LLM role that will be used must have a
non-null `model`: because base.yaml ships with every model `null` until ADR-004, this is
checked before first use, not at load — the code that builds a role's client calls
`EngineConfig.require_llm_model(role)`, which raises `ConfigError`.

`EngineConfig` declares no defaults of its own: every key must come from base.yaml (an
experiment file only overrides), so a key missing from base.yaml is a load error. Errors
are raised as `ConfigError`. `tests/unit/test_config_schema_sync.py` fails if the key
tables below and base.yaml drift apart.

**Frozen** = may change only via an ADR (see the ADR named). Changing an index-defining
key (embedding/sparse/chunking) requires a new `qdrant.collection` name.

## 2. Keys
Only the dotted names in this section are config keys. Dotted names such as `cache.hit` or
`llm.call` in other docs are log event names (OBSERVABILITY §3).

### project / paths (Owner: Kabir)
| Key | Type | Default | Used by |
|---|---|---|---|
| project.name | str | ai-native-retrieval-engine | manifest |
| project.seed | int | 42 | stats bootstrap, tie-free sampling |
| paths.data_dir | path | data | ingestion, datasets |
| paths.runs_dir | path | eval/runs | eval |
| paths.state_db | path | var/state.sqlite3 | TaskStore, Cache |
| paths.llm_cache_db | path | var/llm_cache.sqlite3 | LLMClient |

### embedding / sparse / chunking — Frozen after ADR-001 (Owner: Kabir)
| Key | Type | Default | Notes |
|---|---|---|---|
| embedding.model_name | str | BAAI/bge-base-en-v1.5 | same model for index + all similarity |
| embedding.dimension | int | 768 | must match model |
| embedding.query_prefix | str | BGE retrieval instruction | per model card |
| embedding.passage_prefix | str | "" | per model card |
| embedding.batch_size | int | 64 | |
| embedding.device | cpu\|cuda | cpu | |
| embedding.lru_size | int | 4096 | in-process query-embedding cache |
| sparse.method | bm25 | bm25 | |
| sparse.model_name | str | Qdrant/bm25 | verify in LIBRARY_NOTES |
| chunking.chunk_size_tokens | int | 384 | |
| chunking.chunk_overlap_tokens | int | 48 | |
| chunking.tokenizer | str | BAAI/bge-base-en-v1.5 | token counting for chunking |
| chunking.sentence_aware | bool | true | |

### qdrant / retrieval (Owner: Kabir)
| Key | Type | Default | Notes |
|---|---|---|---|
| qdrant.url | str | http://127.0.0.1:6333 | env `QDRANT_URL` may override |
| qdrant.collection | str | multihop_v1 | Frozen after ADR-001 |
| qdrant.dense_vector_name | str | dense | |
| qdrant.sparse_vector_name | str | sparse | |
| qdrant.id_namespace | uuid | fixed UUID | UUIDv5 namespace for point ids; never change |
| qdrant.hnsw.m / ef_construct | int | 16 / 128 | index build |
| qdrant.search_ef | int | 128 | query time |
| qdrant.timeout_s | int | 10 | → BackendError |
| retrieval.default_k | int | 20 | DENSE/SPARSE_HYBRID and baselines |
| retrieval.expand_step_k | int | 20 | RERANK_EXPAND depth increment |
| retrieval.max_k | int | 60 | depth ceiling |
| retrieval.hybrid_prefetch_k | int | 60 | per-branch prefetch |
| retrieval.hybrid_fusion | rrf | rrf | |

### data — split keys Frozen after ADR-003 (Owner: Kabir)
| Key | Type | Default |
|---|---|---|
| data.dev_fraction | float | 0.7 |
| data.split_seed | int | 42 |
| data.dev_subset_size | int | 300 |
| data.gold_fuzzy_threshold | int 0–100 | 90 |
| data.gold_min_coverage | float [0,1] | 0.95 |
| data.hotpot_sample_size | int | 1000 |
| data.download_timeout_s | int | 60 |
| data.sources.multihop.hf_repo | str | yixuantt/MultiHopRAG |
| data.sources.multihop.revision | str (40-hex commit) | 71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82 |
| data.sources.multihop.files.{corpus.json,MultiHopRAG.json} | str (64-hex SHA-256) | SHA-256 values in ADR-003 |

`data.sources` maps a dataset name to its raw files at a pinned Hugging Face revision;
`data/download.py` downloads `files` into `<paths.data_dir>/raw/<dataset>/` and refuses
any file whose SHA-256 differs. `revision` and `files` are Frozen after ADR-003; a new
source or revision needs an ADR-003 update. Validated: `revision` is 40 lower-case hex
characters, each SHA-256 64.

### llm — models/prompts Frozen after ADR-004 (Owner: Vishal)
| Key | Type | Default | Notes |
|---|---|---|---|
| llm.cache_enabled | bool | true | |
| llm.cache_namespace | str | default | `repeat-<n>` for repeats |
| llm.max_calls_per_run | int | 20000 | hard stop |
| llm.max_cost_usd_per_run | float | 10.0 | hard stop |
| llm.prices | map | {} | `{model: {input_per_mtok, output_per_mtok}}` |
| llm.{planner,verifier,generator}.provider | anthropic\|openai_compatible\|fake | openai_compatible | |
| llm.<role>.model | str\|null | null | required before use |
| llm.<role>.base_url | str | http://127.0.0.1:11434/v1 | ignored by `anthropic` |
| llm.<role>.api_key_env | str | LLM_API_KEY | env var **name** |
| llm.<role>.temperature | float | 0.0 | |
| llm.<role>.max_output_tokens | int | 600 / 50 / 200 | planner / verifier / generator |
| llm.<role>.timeout_s | int | 60 / 30 / 60 | |
| llm.<role>.max_transport_retries | int | 3 | |
| llm.verifier.enabled | bool | false | |

### planner (Owner: Vishal)
| Key | Type | Default |
|---|---|---|
| planner.max_subgoals | int | 3 |
| planner.max_gaps | int | 6 |
| planner.max_attempts | int | 2 |
| planner.evidence_summary_chars | int | 300 |
| planner.max_evidence_summaries | int | 8 |
| planner.entity_match_threshold | int 0–100 | 85 |
| planner.rationale_max_chars | int | 300 |
Truncates `PlanResult.rationale` (never rejects it) — ADR-009 §10.

### cache (Owner: Kabir)
| Key | Type | Default | Frozen |
|---|---|---|---|
| cache.similarity_threshold | float | 0.92 | after calibration ADR |
| cache.max_entries_per_task | int | 64 | |

### controller / monitor (Owner: Samay)
| Key | Type | Default | Frozen |
|---|---|---|---|
| controller.weights.{alpha,beta,gamma,delta} | float, sum 1 | 0.25 each | after calibration ADR |
| controller.gain_subweights.{gap_alignment,recent_gain,type_match} | float, sum 1 | 0.4/0.3/0.3 | |
| controller.type_match.{DENSE,SPARSE_HYBRID}.{semantic,mixed,lexical} | float | see base.yaml | |
| controller.ema_decay | float | 0.5 | |
| controller.prior_action_gain | float | 0.5 | |
| controller.min_utility | float | 0.05 | after calibration ADR |
| controller.cost_units.<Action> | int | REUSE 0, DENSE 1, SPARSE_HYBRID 1, RERANK_EXPAND 2, TRANSFORM 2 | |
| controller.default_latency_ms.<Action> | int | 5/60/90/250/1500 | fallback if no profile |
| controller.latency_profile_path | path | configs/latency_profile.yaml | generated |
| controller.ambiguous_band.{low,high} | float | 0.3 / 0.6 | |
| controller.calibration_lambda | float | 0.1 | calibration objective |
| monitor.stagnation_threshold | float | 0.05 | after calibration ADR |
| monitor.stagnation_patience | int | 2 | |
| monitor.max_transforms_per_streak | int | 1 | |

### rerank / evidence (Owner: Samay)
| Key | Type | Default | Frozen |
|---|---|---|---|
| rerank.cross_encoder_model | str | BAAI/bge-reranker-base | after ADR-005 |
| rerank.weights.{relevance,gap_coverage,novelty,redundancy} | float ≥ 0 | 0.6/0.2/0.2/0.3 | after calibration ADR |
| rerank.top_n_output | int | 10 | bundle size for all systems |
| rerank.batch_size | int | 32 | |
| rerank.max_input_tokens | int | 512 | |
| rerank.device | cpu\|cuda | cpu | |
| evidence.relevance_threshold | float | 0.5 | after calibration ADR |
| evidence.novelty_threshold | float | 0.15 | after calibration ADR |
| evidence.gap_coverage_threshold | float | 0.6 | after calibration ADR |
| evidence.max_accept_per_step | int | 5 | |
| evidence.max_items | int | 20 | |
| evidence.gain_weights.{gap_resolution,quality} | float, sum 1 | 0.5/0.5 | |
| evidence.verify_top_m | int | 2 | |
| evidence.verify_margin | float | 0.1 | |
| evidence.min_items_for_sufficiency | int | 3 | ablation only |

### baselines / budgets / ablation (Owner: Vishal / Samay)
| Key | Type | Default | Notes |
|---|---|---|---|
| fixed_iteration.mode | dense\|sparse\|hybrid | hybrid | |
| budget_defaults.max_steps | int | 8 | per task |
| budget_defaults.max_retrieval_calls | int | 4 | overridden per sub-run by `experiment.retrieval_budgets` |
| budget_defaults.max_llm_calls | int | 6 | |
| budget_defaults.max_tokens | int | 20000 | |
| ablation.{planner,cache,state_aware_rerank,gap_tracking,redundancy_signal,budget_awareness,adaptive_actions} | bool | true | effects in ACTION_POLICY, RERANK_SPEC, EVIDENCE_MANAGER_SPEC |

### api / eval / logging (Owner: Vishal)
| Key | Type | Default |
|---|---|---|
| api.host | str | 127.0.0.1 |
| api.port | int | 8000 |
| api.run_timeout_s | int | 120 |
| api.client_timeout_s | int | 150 |
| api.goal_min_chars | int | 3 |
| api.goal_max_chars | int | 2000 |
| api.query_max_chars | int | 2000 |
| eval.max_workers | int | 4 |
| eval.recall_ks | list[int] | [1,3,5,10] |
| eval.bootstrap_resamples | int | 10000 |
| eval.generator_max_context_tokens | int | 3000 |
| eval.redundancy_cos | float | 0.95 |
| eval.repeats | int | 1 (3 for final test) |
| eval.natural_max_retrieval_calls | int | 8 |
| logging.level | str | INFO |
| logging.json | bool | true |

## 3. `experiment` section (experiment files only)
| Key | Type | Meaning |
|---|---|---|
| experiment.name | str | must equal the file name |
| experiment.system | dense_oneshot\|hybrid_oneshot\|fixed_iteration\|controller_on | |
| experiment.dataset | multihop\|hotpotqa\|curated | |
| experiment.split | dev\|test | `test` requires the frozen tag (run.py checks) |
| experiment.subset | bool | dev only: use `data.dev_subset_size` |
| experiment.budget_mode | equal\|natural | natural → `eval.natural_max_retrieval_calls` |
| experiment.retrieval_budgets | list[int] | one sub-run per value (equal mode); empty in natural mode |
| experiment.notes | str | free text, copied into the manifest |

## 4. Adding or changing a key
Same PR must: add it to `configs/base.yaml` with a comment, to `EngineConfig`, to this
table (owner, default, frozen status), and to the governing spec. Removing a key requires
checking all experiment files.

"""EngineConfig and its loader (docs/CONFIG_SCHEMA.md).

``EngineConfig`` mirrors ``configs/base.yaml`` exactly. Every field is required and has no
default in code: defaults live only in ``configs/base.yaml`` (docs/README.md maintenance
rule 4; CLAUDE.md rule 4), so a key missing from base.yaml fails validation instead of
silently falling back to a value written here. ``extra="forbid"`` at every level rejects
unknown keys.

Load precedence (CONFIG_SCHEMA §1):
1. ``configs/base.yaml``;
2. an experiment file (``configs/experiments/<name>.yaml``), deep-merged over it;
3. environment variables for machine-specific values only (``QDRANT_URL``); ``.env`` is
   loaded first with python-dotenv and never overrides a variable already set;
4. ``--set key=value`` overrides (debugging only; ``eval/run.py`` marks such runs
   non-reportable).

Keys marked Frozen in CONFIG_SCHEMA.md are documented there and in base.yaml; they are not
enforced at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from uuid import UUID

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from engine.errors import ConfigError

__all__ = [
    "CONFIG_PATH_ENV",
    "DEFAULT_CONFIG_PATH",
    "ENV_OVERRIDES",
    "EngineConfig",
    "LLMRole",
    "config_hash",
    "deep_merge",
    "load_config",
]

# --- bootstrap locations (the only literal paths: needed to find the config itself) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
CONFIG_PATH_ENV = "ENGINE_CONFIG"

# Environment variable -> config key it overrides. Secrets (API keys) are never copied into
# the config: roles name the variable (llm.<role>.api_key_env) and the LLM client reads it.
ENV_OVERRIDES: Mapping[str, tuple[str, ...]] = {"QDRANT_URL": ("qdrant", "url")}

# --- validation bounds stated by CONFIG_SCHEMA §1/§2 (schema rules, not tunables) ---
WEIGHT_SUM_TOLERANCE = 1e-6  # "sum to 1 (±1e-6)"
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]  # probabilities / thresholds in [0,1]
NonNegFloat = Annotated[float, Field(ge=0.0)]  # weights
Percent = Annotated[int, Field(ge=0, le=100)]  # rapidfuzz "int 0-100" thresholds
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]  # lower-case hex
GitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]  # commit sha (HF repos)
PositiveFloat = Annotated[float, Field(gt=0.0)]

LLMRole = Literal["planner", "verifier", "generator"]
Device = Literal["cpu", "cuda"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_weight_sum(model: BaseModel, label: str) -> None:
    values = [float(v) for v in model.model_dump().values()]
    total = math.fsum(values)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"{label} must sum to 1 (got {total})")


# ----------------------------------------------------------------------------- sections
class ProjectConfig(_Section):
    name: str
    seed: int


class PathsConfig(_Section):
    data_dir: Path
    runs_dir: Path
    state_db: Path
    llm_cache_db: Path
    rerank_cache_db: Path  # cross-encoder score cache (rerank.score_cache_enabled)
    model_cache_dir: Path  # fastembed model files (its default is %TEMP%)


class EmbeddingConfig(_Section):  # Frozen after ADR-001
    model_name: str
    revision: GitRevision  # Hugging Face commit of model_name (pinned weights)
    dimension: int
    query_prefix: str
    passage_prefix: str
    batch_size: int
    device: Device
    lru_size: int


class SparseConfig(_Section):  # Frozen after ADR-001
    method: Literal["bm25"]
    model_name: str
    revision: GitRevision  # Hugging Face commit of model_name
    k1: NonNegFloat  # BM25 term-frequency saturation
    b: UnitFloat  # BM25 length normalisation
    avg_len: PositiveFloat  # BM25 average document length (words)


class ChunkingConfig(_Section):  # Frozen after ADR-001
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    tokenizer: str
    tokenizer_revision: GitRevision  # Hugging Face commit of tokenizer
    sentence_aware: bool


class HnswConfig(_Section):
    m: int
    ef_construct: int


class QdrantConfig(_Section):
    url: str
    collection: str  # Frozen after ADR-001
    dense_vector_name: str
    sparse_vector_name: str
    id_namespace: UUID  # never change
    hnsw: HnswConfig
    search_ef: int
    timeout_s: int
    upsert_batch_size: int  # points per upsert during indexing


class RetrievalConfig(_Section):
    default_k: int
    expand_step_k: int
    max_k: int
    hybrid_prefetch_k: int
    hybrid_fusion: Literal["rrf"]

    @model_validator(mode="after")
    def _default_k_within_max_k(self) -> Self:
        if self.default_k > self.max_k:
            raise ValueError(
                f"retrieval.default_k ({self.default_k}) must be <= retrieval.max_k ({self.max_k})"
            )
        return self


class DataSource(_Section):
    """A benchmark's raw files at a pinned Hugging Face dataset revision (ADR-003)."""

    hf_repo: str
    revision: GitRevision  # Frozen after ADR-003
    files: dict[str, Sha256Hex]  # file name -> expected SHA-256 (lower-case hex)


class DataConfig(_Section):  # split keys Frozen after ADR-003
    dev_fraction: UnitFloat
    split_seed: int
    dev_subset_size: int
    gold_fuzzy_threshold: Percent
    gold_min_coverage: UnitFloat  # DATA §4 acceptance bar for scripts/check_gold_mapping.py
    hotpot_sample_size: int
    download_timeout_s: int
    sources: dict[str, DataSource]  # dataset name -> source (data/download.py)


class LLMPrice(_Section):
    input_per_mtok: float
    output_per_mtok: float


class LLMRoleConfig(_Section):  # model Frozen after ADR-004 (planner, generator)
    provider: Literal["anthropic", "openai_compatible", "fake"]
    model: str | None  # null = role unused (verifier); required before use (require_llm_model)
    base_url: str
    api_key_env: str  # env var NAME, never the secret
    temperature: float
    max_output_tokens: int
    timeout_s: int
    max_transport_retries: int


class VerifierConfig(LLMRoleConfig):
    enabled: bool


class LLMConfig(_Section):
    cache_enabled: bool
    cache_namespace: str
    max_calls_per_run: int
    max_cost_usd_per_run: float
    prices: dict[str, LLMPrice]
    planner: LLMRoleConfig
    verifier: VerifierConfig
    generator: LLMRoleConfig


class PlannerConfig(_Section):
    max_subgoals: int
    max_gaps: int
    max_attempts: int
    evidence_summary_chars: int
    max_evidence_summaries: int
    entity_match_threshold: Percent
    rationale_max_chars: int
    heuristic_max_tokens: int  # PLANNER_SPEC §6 heuristic(): "short" = at most this many tokens


class CacheConfig(_Section):
    similarity_threshold: UnitFloat  # Frozen after calibration ADR
    max_entries_per_task: int


class ControllerWeights(_Section):  # Frozen after calibration ADR
    alpha: NonNegFloat
    beta: NonNegFloat
    gamma: NonNegFloat
    delta: NonNegFloat

    @model_validator(mode="after")
    def _sum_to_one(self) -> Self:
        _check_weight_sum(self, "controller.weights")
        return self


class GainSubweights(_Section):
    gap_alignment: NonNegFloat
    recent_gain: NonNegFloat
    type_match: NonNegFloat

    @model_validator(mode="after")
    def _sum_to_one(self) -> Self:
        _check_weight_sum(self, "controller.gain_subweights")
        return self


class TypeMatchRow(_Section):
    semantic: UnitFloat
    mixed: UnitFloat
    lexical: UnitFloat


class TypeMatchConfig(_Section):
    DENSE: TypeMatchRow
    SPARSE_HYBRID: TypeMatchRow


class ActionValues(_Section):
    """Per-action integers keyed by the non-STOP action names (cost units, latencies)."""

    REUSE: int
    DENSE: int
    SPARSE_HYBRID: int
    RERANK_EXPAND: int
    TRANSFORM: int


class AmbiguousBand(_Section):
    low: UnitFloat
    high: UnitFloat

    @model_validator(mode="after")
    def _low_below_high(self) -> Self:
        if not self.low < self.high:
            raise ValueError(
                f"controller.ambiguous_band.low ({self.low}) must be < high ({self.high})"
            )
        return self


class ControllerConfig(_Section):
    weights: ControllerWeights
    gain_subweights: GainSubweights
    type_match: TypeMatchConfig
    ema_decay: UnitFloat
    prior_action_gain: UnitFloat
    min_utility: UnitFloat  # Frozen after calibration ADR
    cost_units: ActionValues
    default_latency_ms: ActionValues
    latency_profile_path: Path
    ambiguous_band: AmbiguousBand
    calibration_lambda: UnitFloat


class MonitorConfig(_Section):
    stagnation_threshold: UnitFloat  # Frozen after calibration ADR
    stagnation_patience: int
    max_transforms_per_streak: int


class RerankWeights(_Section):  # Frozen after calibration ADR; non-negative, no sum rule
    relevance: NonNegFloat
    gap_coverage: NonNegFloat
    novelty: NonNegFloat
    redundancy: NonNegFloat


class RerankConfig(_Section):
    cross_encoder_model: str  # Frozen after ADR-005
    revision: GitRevision  # Hugging Face commit of cross_encoder_model (pinned weights)
    weights: RerankWeights
    top_n_output: int
    batch_size: int
    max_input_tokens: int
    device: Device
    score_cache_enabled: bool  # memoise scores per (model, revision, query, text)


class EvidenceGainWeights(_Section):
    gap_resolution: NonNegFloat
    quality: NonNegFloat

    @model_validator(mode="after")
    def _sum_to_one(self) -> Self:
        _check_weight_sum(self, "evidence.gain_weights")
        return self


class EvidenceConfig(_Section):
    relevance_threshold: UnitFloat  # Frozen after calibration ADR
    novelty_threshold: UnitFloat  # Frozen after calibration ADR
    gap_coverage_threshold: UnitFloat  # Frozen after calibration ADR
    max_accept_per_step: int
    max_items: int
    gain_weights: EvidenceGainWeights
    verify_top_m: int
    verify_margin: UnitFloat
    min_items_for_sufficiency: int


class FixedIterationConfig(_Section):
    mode: Literal["dense", "sparse", "hybrid"]


class BudgetDefaults(_Section):
    # Bounds mirror engine.state.models.Budget (ADR-009 §8).
    max_steps: int = Field(ge=1)
    max_retrieval_calls: int = Field(ge=1)
    max_llm_calls: int = Field(ge=0)  # 0 = no-LLM ablation
    max_tokens: int = Field(ge=1)


class AblationConfig(_Section):
    planner: bool
    cache: bool
    state_aware_rerank: bool
    gap_tracking: bool
    redundancy_signal: bool
    budget_awareness: bool
    adaptive_actions: bool


class ApiConfig(_Section):
    host: str
    port: int
    run_timeout_s: int
    client_timeout_s: int
    goal_min_chars: int
    goal_max_chars: int
    query_max_chars: int


class EvalConfig(_Section):
    max_workers: int
    recall_ks: list[int]
    bootstrap_resamples: int
    generator_max_context_tokens: int
    redundancy_cos: UnitFloat
    repeats: int
    natural_max_retrieval_calls: int


class LoggingConfig(_Section):
    # `json` would shadow the (deprecated) BaseModel.json method, so the attribute is
    # `json_` and the YAML key stays `json`. No populate_by_name: a `json_` key in YAML
    # must be rejected like any other unknown key.
    level: str
    json_: bool = Field(alias="json")


class ExperimentConfig(_Section):
    """The `experiment` section; present only in experiment files (CONFIG_SCHEMA §3)."""

    name: str  # must equal the file name (checked by load_config)
    system: Literal["dense_oneshot", "hybrid_oneshot", "fixed_iteration", "controller_on"]
    dataset: Literal["multihop", "hotpotqa", "curated"]
    split: Literal["dev", "test"]
    subset: bool
    budget_mode: Literal["equal", "natural"]
    retrieval_budgets: list[int]
    notes: str


class EngineConfig(_Section):
    project: ProjectConfig
    paths: PathsConfig
    embedding: EmbeddingConfig
    sparse: SparseConfig
    chunking: ChunkingConfig
    qdrant: QdrantConfig
    retrieval: RetrievalConfig
    data: DataConfig
    llm: LLMConfig
    planner: PlannerConfig
    cache: CacheConfig
    controller: ControllerConfig
    monitor: MonitorConfig
    rerank: RerankConfig
    evidence: EvidenceConfig
    fixed_iteration: FixedIterationConfig
    budget_defaults: BudgetDefaults
    ablation: AblationConfig
    api: ApiConfig
    eval: EvalConfig
    logging: LoggingConfig
    experiment: ExperimentConfig | None = None  # experiment files only

    def require_llm_model(self, role: LLMRole) -> str:
        """Return ``llm.<role>.model``, or raise ConfigError if it is still null.

        CONFIG_SCHEMA §1 requires every LLM role that will be used to name a model. Whether
        a role is used is known only to the code that builds its client (factory, eval
        harness), so this check runs there, before first use, not at load: base.yaml may ship
        a role's model as null (the verifier, not built yet).
        """
        model = getattr(self.llm, role).model
        if model is None:
            raise ConfigError(f"llm.{role}.model is null; set it (ADR-004) before using the role")
        return str(model)


def config_hash(cfg: EngineConfig) -> str:
    """SHA-256 of the canonical JSON of the resolved config (manifests, ``/v1/health``)."""
    canonical = json.dumps(cfg.model_dump(mode="json", by_alias=True), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- loader
def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict: mappings merge recursively; every other overlay value replaces."""
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")
    return data


def _set_dotted(data: dict[str, Any], keys: Sequence[str], value: Any) -> dict[str, Any]:
    overlay: dict[str, Any] = {keys[-1]: value}
    for key in reversed(keys[:-1]):
        overlay = {key: overlay}
    return deep_merge(data, overlay)


def _apply_set_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"override must look like key.path=value, got {item!r}")
        data = _set_dotted(data, key.strip().split("."), yaml.safe_load(raw))
    return data


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    base_path: Path = DEFAULT_CONFIG_PATH,
) -> EngineConfig:
    """Load, merge and validate the configuration.

    Args:
        path: base.yaml or an experiment file. ``None`` uses ``$ENGINE_CONFIG``, else
            base.yaml. Any file other than ``base_path`` is deep-merged over base.yaml.
        overrides: ``key.path=value`` strings (CONFIG_SCHEMA §1.4), values parsed as YAML.
        environ: environment to read. ``None`` loads ``.env`` (found from the working
            directory upwards; existing variables are not overridden) and uses
            ``os.environ``. Pass a mapping to bypass ``.env`` entirely (tests).
        base_path: location of base.yaml.

    Raises:
        ConfigError: unreadable or invalid file, experiment name mismatch, or any
            validation failure (unknown key, out-of-range value, broken cross-field rule).
    """
    if environ is None:
        load_dotenv(find_dotenv(usecwd=True), override=False)
        environ = os.environ

    if path is None:
        env_path = environ.get(CONFIG_PATH_ENV, "")
        path = env_path if env_path else base_path
    config_path = Path(path)

    data = _read_yaml(base_path)
    if config_path.resolve() != base_path.resolve():
        overlay = _read_yaml(config_path)
        experiment = overlay.get("experiment")
        if isinstance(experiment, Mapping) and experiment.get("name") != config_path.stem:
            raise ConfigError(
                f"experiment.name {experiment.get('name')!r} must equal the file name "
                f"{config_path.stem!r}"
            )
        data = deep_merge(data, overlay)

    for variable, keys in ENV_OVERRIDES.items():
        value = environ.get(variable, "")
        if value:  # an empty variable (e.g. copied from .env.example) counts as unset
            data = _set_dotted(data, keys, value)

    data = _apply_set_overrides(data, overrides)

    try:
        return EngineConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration ({config_path}):\n{exc}") from exc

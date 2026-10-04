"""engine/config.py: EngineConfig validation rules and the loader (CONFIG_SCHEMA.md)."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError

from engine.config import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    EngineConfig,
    deep_merge,
    load_config,
)
from engine.errors import ConfigError

EXPERIMENTS_DIR = REPO_ROOT / "configs" / "experiments"
EXPERIMENT_FILES = sorted(EXPERIMENTS_DIR.glob("*.yaml"))


def base_data() -> dict[str, Any]:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def with_value(dotted: str, value: Any) -> dict[str, Any]:
    data = copy.deepcopy(base_data())
    node = data
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node[key]
    node[leaf] = value
    return data


def without_key(dotted: str) -> dict[str, Any]:
    data = copy.deepcopy(base_data())
    node = data
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node[key]
    del node[leaf]
    return data


def mapping_paths(node: Any, prefix: str = "") -> Iterator[str]:
    """Dotted path of every mapping in base.yaml ('' = top level)."""
    if isinstance(node, dict):
        yield prefix
        for key, value in node.items():
            yield from mapping_paths(value, f"{prefix}.{key}" if prefix else key)


def leaves(node: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict) and node:
        for key, value in node.items():
            yield from leaves(value, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, node


def write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ------------------------------------------------------------------ base and experiments
def test_base_yaml_loads_without_errors() -> None:
    cfg = load_config(environ={})
    assert cfg.experiment is None
    assert cfg.retrieval.default_k <= cfg.retrieval.max_k


def test_engine_config_mirrors_base_yaml_exactly() -> None:
    cfg = load_config(environ={})
    dumped = dict(leaves(cfg.model_dump(by_alias=True, exclude={"experiment"})))
    expected = dict(leaves(base_data()))
    assert dumped.keys() == expected.keys()
    for key, yaml_value in expected.items():
        value = dumped[key]
        if isinstance(value, Path):
            assert value == Path(yaml_value), key
        elif isinstance(value, UUID):
            assert value == UUID(yaml_value), key
        else:
            assert value == yaml_value, key


def test_there_are_24_experiment_files() -> None:
    assert len(EXPERIMENT_FILES) == 24


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=lambda p: p.stem)
def test_every_experiment_file_loads_and_matches_its_file_name(path: Path) -> None:
    cfg = load_config(path, environ={})
    assert cfg.experiment is not None
    assert cfg.experiment.name == path.stem


def test_experiment_overrides_are_deep_merged_over_base() -> None:
    cfg = load_config(EXPERIMENTS_DIR / "ablation_no_cache_test.yaml", environ={})
    base = load_config(environ={})
    assert cfg.ablation.cache is False
    assert cfg.ablation.planner is base.ablation.planner  # sibling key kept from base
    assert cfg.eval.repeats == 3
    assert cfg.eval.recall_ks == base.eval.recall_ks


def test_experiment_name_must_equal_file_name(tmp_path: Path) -> None:
    overlay = yaml.safe_load((EXPERIMENTS_DIR / "controller_on_dev.yaml").read_text("utf-8"))
    path = write_yaml(tmp_path / "renamed.yaml", overlay)
    with pytest.raises(ConfigError, match="must equal the file name"):
        load_config(path, environ={})


def test_overlay_without_experiment_section_is_allowed(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "local.yaml", {"retrieval": {"default_k": 30}})
    assert load_config(path, environ={}).retrieval.default_k == 30


# ------------------------------------------------------------------------ unknown keys
@pytest.mark.parametrize(
    "parent",
    [p for p in mapping_paths(base_data()) if p != "llm.prices"],
    ids=lambda p: p or "<top>",
)
def test_unknown_key_rejected_at_every_level(parent: str) -> None:
    dotted = f"{parent}.unexpected_key" if parent else "unexpected_key"
    with pytest.raises(ValidationError, match="unexpected_key"):
        EngineConfig.model_validate(with_value(dotted, 1))


def test_unknown_key_in_price_entry_rejected() -> None:
    price = {"input_per_mtok": 1.0, "output_per_mtok": 2.0, "unexpected_key": 3}
    with pytest.raises(ValidationError, match="unexpected_key"):
        EngineConfig.model_validate(with_value("llm.prices", {"some-model": price}))


def test_valid_price_entry_accepted() -> None:
    price = {"input_per_mtok": 1.0, "output_per_mtok": 2.0}
    cfg = EngineConfig.model_validate(with_value("llm.prices", {"some-model": price}))
    assert cfg.llm.prices["some-model"].output_per_mtok == 2.0


def test_unknown_key_in_experiment_section_rejected(tmp_path: Path) -> None:
    overlay = yaml.safe_load((EXPERIMENTS_DIR / "controller_on_dev.yaml").read_text("utf-8"))
    overlay["experiment"]["unexpected_key"] = 1
    path = write_yaml(tmp_path / "controller_on_dev.yaml", overlay)
    with pytest.raises(ConfigError, match="unexpected_key"):
        load_config(path, environ={})


def test_unknown_key_rejected_through_loader(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "local.yaml", {"qdrant": {"hnsw": {"unexpected_key": 1}}})
    with pytest.raises(ConfigError, match="unexpected_key"):
        load_config(path, environ={})


def test_logging_json_key_cannot_be_spelt_as_the_attribute_name() -> None:
    data = base_data()
    data["logging"] = {"level": "INFO", "json_": True}
    with pytest.raises(ValidationError):
        EngineConfig.model_validate(data)


def test_missing_key_rejected_because_code_has_no_defaults() -> None:
    with pytest.raises(ValidationError, match="default_k"):
        EngineConfig.model_validate(without_key("retrieval.default_k"))


# ---------------------------------------------------------------------- weight sum rules
WEIGHT_GROUPS = {
    "controller.weights": ["alpha", "beta", "gamma", "delta"],
    "controller.gain_subweights": ["gap_alignment", "recent_gain", "type_match"],
    "evidence.gain_weights": ["gap_resolution", "quality"],
}


def _weights(group: str, values: list[float]) -> dict[str, float]:
    return dict(zip(WEIGHT_GROUPS[group], values, strict=True))


@pytest.mark.parametrize("group", list(WEIGHT_GROUPS))
def test_weights_summing_to_one_accepted(group: str) -> None:
    n = len(WEIGHT_GROUPS[group])
    values = [1.0] + [0.0] * (n - 1)  # zero weights are allowed (non-negative)
    EngineConfig.model_validate(with_value(group, _weights(group, values)))


@pytest.mark.parametrize("group", list(WEIGHT_GROUPS))
def test_weights_not_summing_to_one_rejected(group: str) -> None:
    n = len(WEIGHT_GROUPS[group])
    with pytest.raises(ValidationError, match="must sum to 1"):
        EngineConfig.model_validate(with_value(group, _weights(group, [0.6] * n)))


@pytest.mark.parametrize("group", list(WEIGHT_GROUPS))
def test_negative_weight_rejected(group: str) -> None:
    n = len(WEIGHT_GROUPS[group])
    values = [1.5, -0.5] + [0.0] * (n - 2)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        EngineConfig.model_validate(with_value(group, _weights(group, values)))


def test_weight_sum_tolerance_is_1e6() -> None:
    inside = {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25 + 5e-7}
    outside = {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25 + 5e-6}
    EngineConfig.model_validate(with_value("controller.weights", inside))
    with pytest.raises(ValidationError, match="must sum to 1"):
        EngineConfig.model_validate(with_value("controller.weights", outside))


def test_rerank_weights_need_not_sum_to_one_but_must_be_non_negative() -> None:
    cfg = load_config(environ={})
    total = sum(cfg.rerank.weights.model_dump().values())
    assert total != pytest.approx(1.0)  # base.yaml: 0.6/0.2/0.2/0.3 is valid
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        EngineConfig.model_validate(with_value("rerank.weights.redundancy", -0.1))


# ------------------------------------------------------------------------- range rules
UNIT_INTERVAL_KEYS = [
    "cache.similarity_threshold",
    "controller.ema_decay",
    "controller.prior_action_gain",
    "controller.min_utility",
    "controller.calibration_lambda",
    "controller.type_match.DENSE.semantic",
    "controller.type_match.SPARSE_HYBRID.lexical",
    "monitor.stagnation_threshold",
    "evidence.relevance_threshold",
    "evidence.novelty_threshold",
    "evidence.gap_coverage_threshold",
    "evidence.verify_margin",
    "data.dev_fraction",
    "eval.redundancy_cos",
]


@pytest.mark.parametrize("key", UNIT_INTERVAL_KEYS)
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_unit_interval_bounds_accepted(key: str, value: float) -> None:
    EngineConfig.model_validate(with_value(key, value))


@pytest.mark.parametrize("key", UNIT_INTERVAL_KEYS)
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_unit_interval_violation_rejected(key: str, value: float) -> None:
    with pytest.raises(ValidationError):
        EngineConfig.model_validate(with_value(key, value))


@pytest.mark.parametrize("key", ["data.gold_fuzzy_threshold", "planner.entity_match_threshold"])
def test_fuzzy_threshold_scale_is_0_to_100(key: str) -> None:
    EngineConfig.model_validate(with_value(key, 0))
    EngineConfig.model_validate(with_value(key, 100))
    with pytest.raises(ValidationError):
        EngineConfig.model_validate(with_value(key, 101))
    with pytest.raises(ValidationError):
        EngineConfig.model_validate(with_value(key, -1))


def test_default_k_equal_to_max_k_accepted() -> None:
    EngineConfig.model_validate(
        with_value("retrieval.default_k", base_data()["retrieval"]["max_k"])
    )


def test_default_k_above_max_k_rejected() -> None:
    too_big = base_data()["retrieval"]["max_k"] + 1
    with pytest.raises(ValidationError, match="default_k"):
        EngineConfig.model_validate(with_value("retrieval.default_k", too_big))


def test_ambiguous_band_low_below_high_accepted() -> None:
    EngineConfig.model_validate(with_value("controller.ambiguous_band", {"low": 0.1, "high": 0.2}))


@pytest.mark.parametrize("band", [{"low": 0.6, "high": 0.6}, {"low": 0.7, "high": 0.3}])
def test_ambiguous_band_not_ordered_rejected(band: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="ambiguous_band"):
        EngineConfig.model_validate(with_value("controller.ambiguous_band", band))


@pytest.mark.parametrize(
    ("key", "lowest_valid"),
    [
        ("budget_defaults.max_steps", 1),
        ("budget_defaults.max_retrieval_calls", 1),
        ("budget_defaults.max_llm_calls", 0),
        ("budget_defaults.max_tokens", 1),
    ],
)
def test_budget_default_bounds(key: str, lowest_valid: int) -> None:
    EngineConfig.model_validate(with_value(key, lowest_valid))
    with pytest.raises(ValidationError):
        EngineConfig.model_validate(with_value(key, lowest_valid - 1))


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("embedding.device", "tpu"),
        ("rerank.device", "tpu"),
        ("sparse.method", "splade"),
        ("retrieval.hybrid_fusion", "dbsf"),
        ("fixed_iteration.mode", "DENSE"),
        ("llm.planner.provider", "unknown"),
        ("qdrant.id_namespace", "not-a-uuid"),
    ],
)
def test_enumerated_values_rejected_when_unknown(key: str, bad: str) -> None:
    with pytest.raises(ValidationError):
        EngineConfig.model_validate(with_value(key, bad))


# ------------------------------------------------------------------------ data sources
GOOD_SHA = "0" * 64
GOOD_REV = "a" * 40


def _source(**overrides: Any) -> dict[str, Any]:
    source: dict[str, Any] = {
        "hf_repo": "org/name",
        "revision": GOOD_REV,
        "files": {"f.json": GOOD_SHA},
    }
    source.update(overrides)
    return source


def test_valid_data_source_accepted() -> None:
    cfg = EngineConfig.model_validate(with_value("data.sources", {"x": _source()}))
    assert cfg.data.sources["x"].files == {"f.json": GOOD_SHA}


@pytest.mark.parametrize("revision", ["main", "A" * 40, "a" * 39, "a" * 41])
def test_revision_must_be_a_40_hex_commit(revision: str) -> None:
    with pytest.raises(ValidationError, match="revision"):
        EngineConfig.model_validate(with_value("data.sources", {"x": _source(revision=revision)}))


@pytest.mark.parametrize("sha", ["A" * 64, "0" * 63, "g" * 64, ""])
def test_file_checksum_must_be_lowercase_sha256_hex(sha: str) -> None:
    source = _source(files={"f.json": sha})
    with pytest.raises(ValidationError, match="files"):
        EngineConfig.model_validate(with_value("data.sources", {"x": source}))


def test_base_yaml_pins_the_multihop_source_from_adr_003() -> None:
    source = load_config(environ={}).data.sources["multihop"]
    assert source.hf_repo == "yixuantt/MultiHopRAG"
    assert set(source.files) == {"corpus.json", "MultiHopRAG.json"}


# --------------------------------------------------------------------- LLM role models
@pytest.mark.parametrize("role", ["planner", "verifier", "generator"])
def test_role_with_null_model_cannot_be_used(role: str) -> None:
    cfg = load_config(environ={})  # base.yaml: every model is null until ADR-004
    with pytest.raises(ConfigError, match=f"llm.{role}.model"):
        cfg.require_llm_model(role)  # type: ignore[arg-type]


@pytest.mark.parametrize("role", ["planner", "verifier", "generator"])
def test_role_with_model_can_be_used(role: str) -> None:
    cfg = EngineConfig.model_validate(with_value(f"llm.{role}.model", "some-model"))
    assert cfg.require_llm_model(role) == "some-model"  # type: ignore[arg-type]


# ------------------------------------------------------------------ environment and .env
def test_qdrant_url_env_overrides_config() -> None:
    cfg = load_config(environ={"QDRANT_URL": "http://127.0.0.1:7000"})
    assert cfg.qdrant.url == "http://127.0.0.1:7000"


def test_empty_env_var_counts_as_unset() -> None:
    cfg = load_config(environ={"QDRANT_URL": ""})
    assert cfg.qdrant.url == base_data()["qdrant"]["url"]


def test_env_wins_over_experiment_file(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "local.yaml", {"qdrant": {"url": "http://127.0.0.1:6000"}})
    cfg = load_config(path, environ={"QDRANT_URL": "http://127.0.0.1:7000"})
    assert cfg.qdrant.url == "http://127.0.0.1:7000"


def test_engine_config_env_selects_the_file() -> None:
    path = EXPERIMENTS_DIR / "dense_oneshot_dev.yaml"
    cfg = load_config(environ={"ENGINE_CONFIG": str(path)})
    assert cfg.experiment is not None
    assert cfg.experiment.name == "dense_oneshot_dev"


def test_dotenv_is_loaded_but_existing_env_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("QDRANT_URL=http://127.0.0.1:7001\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("ENGINE_CONFIG", raising=False)
    assert load_config().qdrant.url == "http://127.0.0.1:7001"

    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:7002")
    assert load_config().qdrant.url == "http://127.0.0.1:7002"


# ------------------------------------------------------------------------ --set overrides
def test_set_override_applies_yaml_typed_value() -> None:
    cfg = load_config(environ={}, overrides=["retrieval.default_k=30", "ablation.cache=false"])
    assert cfg.retrieval.default_k == 30
    assert cfg.ablation.cache is False


def test_set_override_on_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError, match="unexpected_key"):
        load_config(environ={}, overrides=["retrieval.unexpected_key=1"])


@pytest.mark.parametrize("item", ["retrieval.default_k", "=3"])
def test_malformed_set_override_rejected(item: str) -> None:
    with pytest.raises(ConfigError, match="override"):
        load_config(environ={}, overrides=[item])


# ------------------------------------------------------------------------ file handling
def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "absent.yaml", environ={})


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("retrieval: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path, environ={})


def test_non_mapping_file_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path, environ={})


def test_empty_overlay_file_is_a_no_op(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path, environ={}) == load_config(environ={})


def test_deep_merge_replaces_lists_and_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 1}
    overlay = {"a": {"c": [3]}}
    merged = deep_merge(base, overlay)
    assert merged == {"a": {"b": 1, "c": [3]}, "d": 1}
    assert base == {"a": {"b": 1, "c": [1, 2]}, "d": 1}
    assert overlay == {"a": {"c": [3]}}

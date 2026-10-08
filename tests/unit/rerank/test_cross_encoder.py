"""engine/rerank/cross_encoder.py with a fake model (RERANK_SPEC §2, §4; LIBRARY_NOTES)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from engine.config import EngineConfig
from engine.errors import ConfigError
from engine.rerank import cross_encoder
from engine.rerank.cross_encoder import CrossEncoderScorer


class Identity:  # names match torch.nn activation classes, which is what the scorer reads
    pass


class Sigmoid:
    pass


class FakeModel:
    def __init__(self, outputs: list[float], activation: object) -> None:
        self.outputs = outputs
        self.activation_fn = activation
        self.calls: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []

    def predict(self, inputs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.calls.append((inputs, kwargs))
        return self.outputs[: len(inputs)]


def test_logits_are_mapped_through_a_sigmoid(cfg: EngineConfig) -> None:
    model = FakeModel([0.0, 10.0, -10.0], Identity())
    scores = CrossEncoderScorer(cfg.rerank, model)("q", ["a", "b", "c"])
    assert scores[0] == pytest.approx(0.5)
    assert 0.999 < scores[1] <= 1.0 and 0.0 <= scores[2] < 0.001


def test_probabilities_are_kept_as_they_are(cfg: EngineConfig) -> None:
    model = FakeModel([0.0021, 0.7], Sigmoid())
    assert CrossEncoderScorer(cfg.rerank, model)("q", ["a", "b"]) == [0.0021, 0.7]


def test_pairs_batch_size_and_no_progress_bar(cfg: EngineConfig) -> None:
    model = FakeModel([0.1, 0.2], Sigmoid())
    CrossEncoderScorer(cfg.rerank, model)("the query", ["a", "b"])
    ((pairs, kwargs),) = model.calls
    assert pairs == [("the query", "a"), ("the query", "b")]
    assert kwargs == {"batch_size": cfg.rerank.batch_size, "show_progress_bar": False}


def test_unknown_activation_is_a_config_error(cfg: EngineConfig) -> None:
    class Softmax:
        pass

    with pytest.raises(ConfigError, match="unsupported activation Softmax"):
        CrossEncoderScorer(cfg.rerank, FakeModel([0.1], Softmax()))("q", ["a"])


def test_model_loads_lazily_once_with_the_pinned_revision(
    cfg: EngineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[Any] = []

    def fake_load(c: Any) -> FakeModel:
        loads.append(c)
        return FakeModel([0.5, 0.5], Sigmoid())

    monkeypatch.setattr(cross_encoder, "_load_model", fake_load)
    scorer = CrossEncoderScorer(cfg.rerank)
    assert loads == []  # nothing loaded at construction
    assert scorer("q", []) == [] and loads == []  # empty input needs no model
    scorer("q", ["a"])
    scorer("q", ["b"])
    assert loads == [cfg.rerank]
    assert len(cfg.rerank.revision) == 40


class EchoModel:
    """Score = len(text) / 10, so every pair has a known, distinct score."""

    activation_fn = Sigmoid()

    def __init__(self) -> None:
        self.scored: list[tuple[str, str]] = []

    def predict(self, inputs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.scored += inputs
        return [len(t) / 10 for _, t in inputs]


def test_score_cache_scores_each_pair_once_and_persists(cfg: EngineConfig, tmp_path: Path) -> None:
    db = tmp_path / "rerank.sqlite3"
    model = EchoModel()
    scorer = CrossEncoderScorer(cfg.rerank, model, cache_path=db)
    assert scorer("q", ["a", "bbb"]) == [0.1, 0.3]
    assert scorer("q", ["bbb", "cc", "a"]) == [0.3, 0.2, 0.1]  # input order kept
    assert model.scored == [("q", "a"), ("q", "bbb"), ("q", "cc")]  # only the new pair
    assert scorer("other query", ["a"]) == [0.1]
    assert model.scored[-1] == ("other query", "a")  # the query is part of the key

    again = EchoModel()  # a new process: scores come from the file
    assert CrossEncoderScorer(cfg.rerank, again, cache_path=db)("q", ["cc", "a"]) == [0.2, 0.1]
    assert again.scored == []


def test_score_cache_key_includes_the_model_revision(cfg: EngineConfig, tmp_path: Path) -> None:
    db = tmp_path / "rerank.sqlite3"
    CrossEncoderScorer(cfg.rerank, EchoModel(), cache_path=db)("q", ["a"])
    other = cfg.rerank.model_copy(update={"revision": "f" * 40})
    model = EchoModel()
    CrossEncoderScorer(other, model, cache_path=db)("q", ["a"])
    assert model.scored == [("q", "a")]  # different weights: scored again


def test_without_a_cache_path_nothing_is_stored(cfg: EngineConfig, tmp_path: Path) -> None:
    model = EchoModel()
    scorer = CrossEncoderScorer(cfg.rerank, model)
    scorer("q", ["a"])
    scorer("q", ["a"])
    assert model.scored == [("q", "a"), ("q", "a")]
    assert list(tmp_path.iterdir()) == []

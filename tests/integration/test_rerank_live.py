"""The real cross-encoder at the pinned revision (RERANK_SPEC §2; ADR-005; LIBRARY_NOTES).

Needs the model in the Hugging Face cache (downloaded on first run, ~1.1 GB for
bge-reranker-base). Two pairs only, so it stays fast on CPU.
"""

from __future__ import annotations

import pytest

from engine.config import load_config
from engine.rerank.cross_encoder import CrossEncoderScorer

pytestmark = pytest.mark.integration


def test_relevant_passage_scores_higher_and_scores_are_probabilities() -> None:
    cfg = load_config()
    scorer = CrossEncoderScorer(cfg.rerank)
    relevant, unrelated = scorer(
        "Which company acquired the chip startup?",
        [
            "The chip startup was acquired by Nvidia for 700 million dollars in March.",
            "The football club won the derby on Sunday after a late penalty.",
        ],
    )
    assert 0.0 <= unrelated < relevant <= 1.0
    assert type(scorer._model.activation_fn).__name__ in {"Sigmoid", "Identity"}  # type: ignore[union-attr]

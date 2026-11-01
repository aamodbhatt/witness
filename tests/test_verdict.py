from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from witness.verify.calibration import (
    expected_calibration_error,
    fit_temperature,
    softmax,
)
from witness.verify.verdict import decide, summarize
from witness.verify.verifier import label_permutation


def test_confident_support_and_refute() -> None:
    assert decide([0.9, 0.05, 0.05], 0.6).verdict == "supported"
    assert decide([0.05, 0.9, 0.05], 0.6).verdict == "refuted"


def test_low_confidence_calls_become_unverifiable() -> None:
    d = decide([0.5, 0.3, 0.2], 0.6)
    assert d.verdict == "unverifiable"
    assert d.fever_label == "SUPPORTS"  # evaluation still sees the raw argmax
    assert d.confidence == pytest.approx(0.5)


def test_nei_is_unverifiable() -> None:
    d = decide([0.1, 0.1, 0.8], 0.6)
    assert d.verdict == "unverifiable"
    assert d.confidence == pytest.approx(0.8)


def test_invalid_distributions_rejected() -> None:
    with pytest.raises(ValueError):
        decide([0.5, 0.5], 0.6)
    with pytest.raises(ValueError):
        decide([0.9, 0.9, 0.1], 0.6)


@given(st.lists(st.floats(0.0, 1.0), min_size=3, max_size=3), st.floats(0.0, 1.0))
def test_decide_is_total_and_bounded(raw: list[float], threshold: float) -> None:
    total = sum(raw)
    if total <= 0:
        return
    d = decide([x / total for x in raw], threshold)
    assert d.verdict in ("supported", "refuted", "unverifiable")
    assert 0.0 <= d.confidence <= 1.0 + 1e-9


def test_summary_counts() -> None:
    s = summarize(["supported", "refuted", "supported", "unverifiable"])
    assert (s.supported, s.refuted, s.unverifiable, s.total) == (2, 1, 1, 4)


def test_temperature_scaling_recovers_true_temperature() -> None:
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 3)).astype(np.float32) * 2.0
    probs = softmax(true_logits)
    labels = np.asarray([rng.choice(3, p=p) for p in probs], dtype=np.int64)
    overconfident = true_logits * 3.0  # a model that is 3x too sharp
    t = fit_temperature(overconfident, labels)
    assert t == pytest.approx(3.0, rel=0.1)
    before = expected_calibration_error(softmax(overconfident), labels)
    after = expected_calibration_error(softmax(overconfident, t), labels)
    assert after < before


def test_label_permutation_handles_nli_heads() -> None:
    # cross-encoder/nli-* order: contradiction, entailment, neutral
    perm = label_permutation({0: "contradiction", 1: "entailment", 2: "neutral"})
    assert perm == [1, 0, 2]
    assert label_permutation({0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}) == [0, 1, 2]

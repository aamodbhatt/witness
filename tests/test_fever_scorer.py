"""The scorer must agree with the official fever-scorer, including its README examples."""

from __future__ import annotations

import pytest

from witness.evaluation.fever_scorer import fever_score, is_strictly_correct

# The two instances from the official fever-scorer README, verbatim in structure.
INSTANCE_1 = {
    "label": "REFUTES",
    "predicted_label": "REFUTES",
    "predicted_evidence": [["page1", 1]],
    "evidence": [[[None, None, "page1", 1], [None, None, "page2", 2]]],
}
INSTANCE_2 = {
    "label": "REFUTES",
    "predicted_label": "REFUTES",
    "predicted_evidence": [["page1", 1], ["page2", 2], ["page3", 3]],
    "evidence": [[[None, None, "page1", 1], [None, None, "page2", 2]]],
}


def test_official_readme_example() -> None:
    scores = fever_score([INSTANCE_1, INSTANCE_2])
    assert scores.fever_score == pytest.approx(0.5)
    assert scores.label_accuracy == pytest.approx(1.0)
    assert scores.evidence_precision == pytest.approx(0.8333, abs=1e-4)
    assert scores.evidence_recall == pytest.approx(0.5)
    assert scores.evidence_f1 == pytest.approx(0.625)


def test_nei_needs_only_the_label() -> None:
    nei = {
        "label": "NOT ENOUGH INFO",
        "predicted_label": "NOT ENOUGH INFO",
        "predicted_evidence": [["anything", 0]],
        "evidence": [[[1, None, None, None]]],
    }
    assert is_strictly_correct(nei)
    scores = fever_score([nei])
    assert scores.fever_score == 1.0
    assert scores.label_accuracy == 1.0


def test_wrong_label_is_never_strictly_correct() -> None:
    wrong = dict(INSTANCE_2, predicted_label="SUPPORTS")
    assert not is_strictly_correct(wrong, max_evidence=5)
    assert fever_score([wrong]).fever_score == 0.0


def test_only_first_five_predicted_sentences_count() -> None:
    late = dict(
        INSTANCE_2,
        predicted_evidence=[["x", i] for i in range(5)] + [["page1", 1], ["page2", 2]],
    )
    assert not is_strictly_correct(late, max_evidence=5)
    assert is_strictly_correct(late, max_evidence=None)


def test_any_complete_group_suffices() -> None:
    two_groups = {
        "label": "SUPPORTS",
        "predicted_label": "SUPPORTS",
        "predicted_evidence": [["b", 0]],
        "evidence": [[[1, 1, "a", 0], [1, 1, "c", 3]], [[2, 2, "b", 0]]],
    }
    assert fever_score([two_groups]).fever_score == 1.0


def test_empty_predictions_rejected() -> None:
    with pytest.raises(ValueError):
        fever_score([])

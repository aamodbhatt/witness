"""FEVER scoring, behaviourally identical to the official ``fever-scorer`` (Thorne et al.).

* **Label accuracy**: predicted label equals the gold label.
* **FEVER score** (strict): the label is right *and*, for SUPPORTS/REFUTES, at least one
  complete gold evidence group is contained in the first ``max_evidence`` (5) predicted
  sentences. NOT ENOUGH INFO claims only need the right label.
* Evidence precision/recall/F1 are macro-averaged over verifiable claims.

Instances use the official JSON shape: ``label``, ``predicted_label``,
``predicted_evidence`` (``[[page, line], ...]``) and ``evidence`` (groups of
``[annotation_id, evidence_id, page, line]``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

NEI_LABEL = "NOT ENOUGH INFO"
Instance = Mapping[str, Any]


def _gold_groups(instance: Instance) -> list[list[list[Any]]]:
    return [[[e[2], e[3]] for e in group] for group in instance["evidence"]]


def _predicted(instance: Instance, max_evidence: int | None) -> list[list[Any]]:
    predicted = [list(p) for p in instance["predicted_evidence"]]
    return predicted if max_evidence is None else predicted[:max_evidence]


def is_correct_label(instance: Instance) -> bool:
    return str(instance["label"]).upper() == str(instance["predicted_label"]).upper()


def is_strictly_correct(instance: Instance, max_evidence: int | None = None) -> bool:
    if not is_correct_label(instance):
        return False
    if str(instance["label"]).upper() == NEI_LABEL:
        return True
    predicted = _predicted(instance, max_evidence)
    return any(all(sentence in predicted for sentence in group) for group in _gold_groups(instance))


def evidence_macro_precision(
    instance: Instance, max_evidence: int | None = None
) -> tuple[float, float]:
    if str(instance["label"]).upper() == NEI_LABEL:
        return 0.0, 0.0
    gold = [[e[2], e[3]] for group in instance["evidence"] for e in group if e[3] is not None]
    predicted = _predicted(instance, max_evidence)
    hits = sum(1.0 for p in predicted if p in gold)
    return (hits / len(predicted) if predicted else 1.0), 1.0


def evidence_macro_recall(
    instance: Instance, max_evidence: int | None = None
) -> tuple[float, float]:
    if str(instance["label"]).upper() == NEI_LABEL:
        return 0.0, 0.0
    groups = _gold_groups(instance)
    if not groups:
        return 1.0, 1.0
    predicted = _predicted(instance, max_evidence)
    for group in groups:
        if all(sentence in predicted for sentence in group):
            return 1.0, 1.0
    return 0.0, 1.0


@dataclass(frozen=True, slots=True)
class FeverScores:
    fever_score: float
    label_accuracy: float
    evidence_precision: float
    evidence_recall: float
    evidence_f1: float


def fever_score(predictions: Sequence[Instance], max_evidence: int | None = 5) -> FeverScores:
    if not predictions:
        raise ValueError("fever_score needs at least one prediction")
    correct = strict = 0.0
    precision = precision_hits = recall = recall_hits = 0.0
    for instance in predictions:
        if is_correct_label(instance):
            correct += 1.0
            if is_strictly_correct(instance, max_evidence):
                strict += 1.0
        p, ph = evidence_macro_precision(instance, max_evidence)
        r, rh = evidence_macro_recall(instance, max_evidence)
        precision, precision_hits = precision + p, precision_hits + ph
        recall, recall_hits = recall + r, recall_hits + rh
    total = float(len(predictions))
    pr = precision / precision_hits if precision_hits > 0 else 1.0
    rec = recall / recall_hits if recall_hits > 0 else 0.0
    f1 = 2.0 * pr * rec / (pr + rec) if pr + rec > 0 else 0.0
    return FeverScores(strict / total, correct / total, pr, rec, f1)

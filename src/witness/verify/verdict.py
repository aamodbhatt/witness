"""From calibrated probabilities to what the user sees: green, red or grey.

The verifier always produces three probabilities. The UI must not paint a claim green or
red on a coin flip, so a support/refute call whose calibrated confidence is below
``abstain_below`` is shown as *unverifiable*, the same grey as NOT ENOUGH INFO.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from witness.verify.verifier import NEI, REFUTES, SUPPORTS

Verdict = Literal["supported", "refuted", "unverifiable"]


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    confidence: float  # calibrated probability of the displayed verdict's label
    fever_label: str  # the raw argmax label, for evaluation
    probabilities: tuple[float, float, float]  # S, R, NEI


def decide(probabilities: Sequence[float], abstain_below: float) -> Decision:
    if len(probabilities) != 3:
        raise ValueError(f"expected 3 probabilities (S, R, NEI), got {len(probabilities)}")
    probs = (float(probabilities[0]), float(probabilities[1]), float(probabilities[2]))
    if abs(sum(probs) - 1.0) > 1e-3 or min(probs) < 0:
        raise ValueError(f"probabilities must be a distribution, got {probs}")
    best = max(range(3), key=lambda i: (probs[i], -i))
    fever = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")[best]
    if best == SUPPORTS and probs[SUPPORTS] >= abstain_below:
        return Decision("supported", probs[SUPPORTS], fever, probs)
    if best == REFUTES and probs[REFUTES] >= abstain_below:
        return Decision("refuted", probs[REFUTES], fever, probs)
    # Grey: either the model says NEI, or it leans one way without enough confidence.
    confidence = probs[NEI] if best == NEI else 1.0 - probs[best]
    return Decision("unverifiable", confidence, fever, probs)


@dataclass(frozen=True, slots=True)
class PageSummary:
    supported: int
    refuted: int
    unverifiable: int

    @property
    def total(self) -> int:
        return self.supported + self.refuted + self.unverifiable


def summarize(verdicts: Iterable[Verdict]) -> PageSummary:
    counts = Counter(verdicts)
    return PageSummary(counts["supported"], counts["refuted"], counts["unverifiable"])

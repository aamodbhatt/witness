"""FEVER claim files and the leakage-safe splits used for training, calibration and eval.

* ``train``: FEVER ``train.jsonl`` minus the calibration bucket. The verifier trains here.
* ``calib``: a deterministic 2% bucket of ``train.jsonl`` (by claim id hash). The
  temperature and nothing else is fit here.
* ``dev``: FEVER ``shared_task_dev.jsonl``, the split the literature reports as "dev".
  It is only ever scored, never trained or tuned on.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from witness.errors import WitnessError

EvidenceGroup = tuple[tuple[str, int], ...]
CALIB_BUCKETS = 50


@dataclass(frozen=True, slots=True)
class FeverClaim:
    id: int
    claim: str
    label: str
    evidence: tuple[EvidenceGroup, ...]  # empty for NOT ENOUGH INFO
    raw_evidence: tuple[tuple[tuple[object, ...], ...], ...]  # official-format groups

    @property
    def gold_pages(self) -> set[str]:
        return {title for group in self.evidence for title, _ in group}


def load_claims(path: Path) -> list[FeverClaim]:
    if not path.is_file():
        raise WitnessError(f"FEVER claims file not found: {path}; download it first")
    claims: list[FeverClaim] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            label = str(record["label"])
            raw_groups = tuple(tuple(tuple(e) for e in g) for g in record["evidence"])
            groups: tuple[EvidenceGroup, ...] = ()
            if label != "NOT ENOUGH INFO":
                groups = tuple(
                    tuple((str(e[2]), int(e[3])) for e in group) for group in record["evidence"]
                )
            claims.append(FeverClaim(int(record["id"]), str(record["claim"]), label, groups,
                                     raw_groups))  # fmt: skip
    return claims


def is_calibration(claim_id: int) -> bool:
    digest = hashlib.sha1(str(claim_id).encode(), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "big") % CALIB_BUCKETS == 0


def split_train(claims: Sequence[FeverClaim]) -> tuple[list[FeverClaim], list[FeverClaim]]:
    """(train, calib) partition of FEVER train claims."""
    train = [c for c in claims if not is_calibration(c.id)]
    calib = [c for c in claims if is_calibration(c.id)]
    return train, calib


def seeded_sample(claims: Sequence[FeverClaim], n: int | None, seed: int) -> list[FeverClaim]:
    """A reproducible random subset (all claims when ``n`` is None or too large)."""
    if n is None or n >= len(claims):
        return list(claims)
    rng = random.Random(seed)
    return sorted(rng.sample(list(claims), n), key=lambda c: c.id)

"""Three-way claim verification: (evidence premise, claim hypothesis) -> S / R / NEI logits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from witness.errors import BackendError

LABELS: tuple[str, str, str] = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
SUPPORTS, REFUTES, NEI = 0, 1, 2

# The fine-tuning initialisation, usable as-is when no trained verifier exists
# (configs/laptop.toml); 71.70 label accuracy on FEVER dev with the full index.
OFF_THE_SHELF_VERIFIER = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# NLI heads name their classes differently; all of them map onto FEVER's three labels.
_ALIASES = {
    "supports": SUPPORTS,
    "entailment": SUPPORTS,
    "refutes": REFUTES,
    "contradiction": REFUTES,
    "not enough info": NEI,
    "neutral": NEI,
}


class Verifier(Protocol):
    """Returns raw logits of shape (n, 3) in :data:`LABELS` order."""

    def logits(self, claims: Sequence[str], premises: Sequence[str]) -> NDArray[np.float32]: ...


def label_permutation(id2label: dict[int, str]) -> list[int]:
    """Column indices that reorder a model's logits into :data:`LABELS` order."""
    position: dict[int, int] = {}
    for idx, name in id2label.items():
        key = name.strip().lower().replace("_", " ")
        if key not in _ALIASES:
            raise BackendError(
                f"verifier label {name!r} is not one of {sorted(_ALIASES)}; "
                "use an NLI or FEVER classification head"
            )
        position[_ALIASES[key]] = int(idx)
    if sorted(position) != [SUPPORTS, REFUTES, NEI]:
        raise BackendError(f"verifier labels {id2label} do not cover support/refute/neutral")
    return [position[SUPPORTS], position[REFUTES], position[NEI]]


def format_premise(evidence: Sequence[tuple[str, str]]) -> str:
    """Concatenate (title, sentence) evidence into one premise string."""
    return " ".join(f"{title}: {text}" for title, text in evidence)


def load_failure_message(model: str, exc: OSError) -> str:
    """Say why ``model`` did not load and the two ways to get a working verifier."""
    lines = str(exc).strip().splitlines()
    reason = lines[0] if lines else type(exc).__name__
    return (
        f"cannot load the verifier {model!r}: {reason}. A fresh clone has no trained "
        "verifier: either create one with `witness train-verifier` (needs a CUDA GPU; see "
        "the README), or set [verifier] model to an NLI model on the Hugging Face Hub, as "
        f"configs/laptop.toml does with {OFF_THE_SHELF_VERIFIER}"
    )


class NLIVerifier:
    """A sequence-classification model over (premise, hypothesis) pairs."""

    def __init__(self, model: str, device: str, *, max_length: int, batch_size: int) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model)
            hf_model = AutoModelForSequenceClassification.from_pretrained(model, dtype=dtype)
        except OSError as exc:  # transformers' "not a local folder nor a Hub model" error
            raise BackendError(load_failure_message(model, exc)) from exc
        self._model = hf_model.to(device).eval()
        id2label = {int(k): str(v) for k, v in hf_model.config.id2label.items()}
        self._perm = label_permutation(id2label)

    def logits(self, claims: Sequence[str], premises: Sequence[str]) -> NDArray[np.float32]:
        if len(claims) != len(premises):
            raise ValueError("claims and premises must have the same length")
        out = np.zeros((len(claims), 3), dtype=np.float32)
        order = np.argsort([len(p) + len(c) for p, c in zip(premises, claims, strict=True)])
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch = self._tokenizer(
                    [premises[i] for i in idx],
                    [claims[i] for i in idx],
                    truncation="only_first",
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**batch).logits.float().cpu().numpy()
                out[idx] = logits[:, self._perm]
        return out

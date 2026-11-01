"""Temperature scaling (Guo et al., 2017) and expected calibration error.

One scalar T divides the logits; it changes confidence but never the argmax, so accuracy
is untouched while the confidence shown in the hover card becomes meaningful. T is fit
on a split held out from FEVER *train*, never on the dev set we report.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from witness.errors import ConfigError


def softmax(logits: NDArray[np.float32], temperature: float = 1.0) -> NDArray[np.float64]:
    z = logits.astype(np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=1, keepdims=True), dtype=np.float64)


def nll(logits: NDArray[np.float32], labels: NDArray[np.int64], temperature: float) -> float:
    probs = softmax(logits, temperature)
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(
    logits: NDArray[np.float32], labels: NDArray[np.int64], lo: float = 0.05, hi: float = 10.0
) -> float:
    """Minimise NLL over T with golden-section search on log T (NLL is unimodal in T)."""
    if len(labels) == 0:
        raise ValueError("cannot fit a temperature on an empty split")
    a, b = np.log(lo), np.log(hi)
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    for _ in range(80):
        if nll(logits, labels, float(np.exp(c))) < nll(logits, labels, float(np.exp(d))):
            b = d
        else:
            a = c
        c, d = b - ratio * (b - a), a + ratio * (b - a)
    return float(np.exp((a + b) / 2.0))


def expected_calibration_error(
    probs: NDArray[np.float64], labels: NDArray[np.int64], bins: int = 15
) -> float:
    """ECE with equal-width confidence bins over the max-probability prediction."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def save_temperature(path: Path, temperature: float, *, fitted_on: str, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"temperature": temperature, "fitted_on": fitted_on, "examples": n}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_temperature(path: Path) -> float:
    """The fitted temperature, or 1.0 (uncalibrated) when no file exists."""
    if not path.is_file():
        return 1.0
    try:
        value = float(json.loads(path.read_text(encoding="utf-8"))["temperature"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(f"calibration file {path} is malformed: {exc}") from exc
    if value <= 0:
        raise ConfigError(f"calibration temperature must be positive, got {value}")
    return value

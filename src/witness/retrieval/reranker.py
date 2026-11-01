"""Sentence scoring with a cross-encoder (claim, "Title: sentence") -> relevance logit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from witness.config import RerankerConfig


class SentenceScorer(Protocol):
    """Scores (query, passage) pairs; higher means more relevant."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]: ...


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        from witness.errors import ConfigError

        raise ConfigError("device = 'cuda' was requested but no CUDA GPU is visible")
    return device


class CrossEncoderScorer:
    """A Hugging Face sequence-classification cross-encoder with one relevance logit."""

    def __init__(self, config: RerankerConfig, device: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.config = config
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(config.model)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(config.model, dtype=dtype)
        self._model = model.to(device).eval()

    def score(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        # Length-sorted batches waste far less compute on padding.
        order = np.argsort([len(q) + len(p) for q, p in pairs], kind="stable")
        out = np.zeros(len(pairs), dtype=np.float32)
        size = self.config.batch_size
        with torch.inference_mode():
            for start in range(0, len(order), size):
                idx = order[start : start + size]
                batch = self._tokenizer(
                    [pairs[i][0] for i in idx],
                    [pairs[i][1] for i in idx],
                    truncation="only_second",
                    max_length=self.config.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**batch).logits
                out[idx] = logits[:, 0].float().cpu().numpy()
        return out

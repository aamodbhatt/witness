"""Model-free stand-ins with the production interfaces, used by tests and the browser test."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from witness.checker import CheckResult
from witness.retrieval.linker import LinkedPage
from witness.retrieval.pipeline import EvidenceSentence
from witness.text import content_words
from witness.verify.verdict import decide
from witness.wiki.store import Page

PAGES = [
    Page(0, "Paris", ("Paris is the capital and most populous city of France.",
                      "The Seine flows through Paris."), ((), ())),
    Page(1, "Marie_Curie", ("Marie Curie was a Polish and naturalised-French physicist.",
                            "", "She won two Nobel Prizes."), ((), (), ())),
    Page(2, "Berlin", ("Berlin is the capital of Germany.",), ((),)),
]  # fmt: skip


class FakeStore:
    def pages(self, ids: Sequence[int]) -> list[Page]:
        return [PAGES[i] for i in ids]


class FakeLinker:
    def link(self, claim: str, k: int | None = None) -> list[LinkedPage]:
        hits = [LinkedPage(p.id, 1.0, p.title) for p in PAGES if p.title in claim]
        return hits[: k or 10]


class FakeSearch:
    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        words = set(content_words(query))
        scored = []
        for p in PAGES:
            overlap = len(words & set(content_words(" ".join(p.sentences))))
            if overlap:
                scored.append((p.id, float(overlap)))
        return sorted(scored, key=lambda t: -t[1])[:k]


class OverlapScorer:
    """Relevance = shared content words between the claim and "Title: sentence"."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        return np.asarray(
            [len(set(content_words(q)) & set(content_words(p))) for q, p in pairs],
            dtype=np.float32,
        )


class KeywordVerifier:
    """'not'/'never' in the claim refutes, 'capital' supports, anything else is NEI."""

    def logits(self, claims: Sequence[str], premises: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(claims), 3), dtype=np.float32)
        for i, claim in enumerate(claims):
            words = set(claim.lower().split())
            if words & {"not", "never"}:
                out[i] = (0.0, 4.0, 0.0)
            elif "capital" in words:
                out[i] = (4.0, 0.0, 0.0)
            else:
                out[i] = (0.0, 0.0, 4.0)
        return out


class FakeChecker:
    """Deterministic verdicts for API and browser tests (no retrieval, no models)."""

    def check_many(self, claims: Sequence[str]) -> list[CheckResult]:
        logits = KeywordVerifier().logits(claims, [""] * len(claims))
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        results = []
        for claim, row in zip(claims, probs, strict=True):
            evidence = [
                EvidenceSentence(0, "Paris", "Paris", 0, PAGES[0].sentences[0], 3.2,
                                 "https://en.wikipedia.org/wiki/Paris")
            ]  # fmt: skip
            results.append(CheckResult(claim, decide(row.tolist(), 0.6), evidence, {}))
        return results

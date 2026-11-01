"""Claim -> evidence set: entity linking ∪ BM25 pages, fused, then sentence reranking."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from witness.config import WitnessConfig
from witness.retrieval.linker import LinkedPage
from witness.retrieval.reranker import SentenceScorer
from witness.wiki.store import Page


class PageSource(Protocol):
    """Read access to pages by id (the :class:`WikiStore` in production)."""

    def pages(self, ids: Sequence[int]) -> list[Page]: ...


class Linker(Protocol):
    def link(self, claim: str, k: int | None = None) -> list[LinkedPage]: ...


class PageSearch(Protocol):
    def search(self, query: str, k: int) -> list[tuple[int, float]]: ...


@dataclass(frozen=True, slots=True)
class EvidenceSentence:
    page: int
    raw_title: str
    title: str
    line: int
    text: str
    score: float
    url: str

    def as_premise(self) -> str:
        return f"{self.title}: {self.text}"


@dataclass(frozen=True, slots=True)
class CandidatePages:
    linked: list[int]
    bm25: list[int]
    fused: list[int]


@dataclass
class Retrieval:
    claim: str
    candidates: CandidatePages
    evidence: list[EvidenceSentence]
    timings_ms: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Sequence[tuple[Sequence[int], float]], k: int, top: int
) -> list[int]:
    """Weighted RRF: score(d) = sum_r w_r / (k + rank_r(d)), ranks starting at 1.

    Ties break by first appearance across the input rankings so the result is
    deterministic.
    """
    scores: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    for ranking, weight in rankings:
        if weight <= 0:
            continue
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + weight / (k + rank)
            first_seen.setdefault(doc, len(first_seen))
    ordered = sorted(scores, key=lambda d: (-scores[d], first_seen[d]))
    return ordered[:top]


def sentence_pairs(
    claim: str, pages: Sequence[Page], max_per_page: int
) -> tuple[list[tuple[str, str]], list[tuple[Page, int]]]:
    """(claim, "Title: sentence") pairs for every non-empty sentence, plus their origin."""
    pairs: list[tuple[str, str]] = []
    origin: list[tuple[Page, int]] = []
    for page in pages:
        kept = 0
        for line, text in enumerate(page.sentences):
            if not text:
                continue
            pairs.append((claim, f"{page.title}: {text}"))
            origin.append((page, line))
            kept += 1
            if kept >= max_per_page:
                break
    return pairs, origin


def top_evidence(
    origin: Sequence[tuple[Page, int]], scores: NDArray[np.float32], k: int
) -> list[EvidenceSentence]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [
        EvidenceSentence(
            page=origin[i][0].id,
            raw_title=origin[i][0].raw_title,
            title=origin[i][0].title,
            line=origin[i][1],
            text=origin[i][0].sentences[origin[i][1]],
            score=float(scores[i]),
            url=origin[i][0].url,
        )
        for i in order
    ]


class EvidenceRetriever:
    """Composes the retrievers and the sentence scorer; stateless per call."""

    def __init__(
        self,
        store: PageSource,
        linker: Linker,
        bm25: PageSearch,
        scorer: SentenceScorer,
        config: WitnessConfig,
    ) -> None:
        self.store = store
        self.linker = linker
        self.bm25 = bm25
        self.scorer = scorer
        self.config = config

    def candidate_pages(self, claim: str) -> CandidatePages:
        linked = [p.page for p in self.linker.link(claim, self.config.linker.top_k)]
        bm25 = [d for d, _ in self.bm25.search(claim, self.config.bm25.top_k)]
        fusion = self.config.fusion
        fused = reciprocal_rank_fusion(
            [(linked, fusion.linker_weight), (bm25, fusion.bm25_weight)],
            k=fusion.rrf_k,
            top=fusion.top_pages,
        )
        return CandidatePages(linked, bm25, fused)

    def retrieve(self, claim: str) -> Retrieval:
        return self.retrieve_many([claim])[0]

    def retrieve_many(self, claims: Sequence[str]) -> list[Retrieval]:
        """Retrieve for several claims, scoring all their sentences in one batch."""
        t0 = time.perf_counter()
        candidates = [self.candidate_pages(c) for c in claims]
        t1 = time.perf_counter()
        all_pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int, list[tuple[Page, int]]]] = []
        for claim, cand in zip(claims, candidates, strict=True):
            pairs, origin = sentence_pairs(
                claim, self.store.pages(cand.fused), self.config.reranker.max_sentences_per_page
            )
            spans.append((len(all_pairs), len(all_pairs) + len(pairs), origin))
            all_pairs.extend(pairs)
        t2 = time.perf_counter()
        scores = self.scorer.score(all_pairs)
        t3 = time.perf_counter()
        per_claim = 1000.0 / max(len(claims), 1)
        timings = {
            "pages_ms": (t1 - t0) * per_claim,
            "fetch_ms": (t2 - t1) * per_claim,
            "rerank_ms": (t3 - t2) * per_claim,
        }
        return [
            Retrieval(
                claim=claim,
                candidates=cand,
                evidence=top_evidence(origin, scores[a:b], self.config.reranker.evidence_k),
                timings_ms=dict(timings),
            )
            for claim, cand, (a, b, origin) in zip(claims, candidates, spans, strict=True)
        ]

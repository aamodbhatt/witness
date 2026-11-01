"""The end-to-end fact checker: claims in, verdicts with evidence out.

Also the single place that wires concrete components from a :class:`WitnessConfig`, so
the CLI, the server and the evaluation build exactly the same system.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from witness.config import WitnessConfig
from witness.retrieval.pipeline import EvidenceRetriever, EvidenceSentence, Retrieval
from witness.verify.calibration import load_temperature, softmax
from witness.verify.verdict import Decision, decide
from witness.verify.verifier import Verifier


class ClaimRetriever(Protocol):
    def retrieve_many(self, claims: Sequence[str]) -> list[Retrieval]: ...


@dataclass
class CheckResult:
    claim: str
    decision: Decision
    evidence: list[EvidenceSentence]
    timings_ms: dict[str, float] = field(default_factory=dict)


class FactChecker:
    def __init__(
        self,
        retriever: ClaimRetriever,
        verifier: Verifier,
        *,
        temperature: float = 1.0,
        abstain_below: float = 0.6,
    ) -> None:
        self.retriever = retriever
        self.verifier = verifier
        self.temperature = temperature
        self.abstain_below = abstain_below

    def check_many(self, claims: Sequence[str]) -> list[CheckResult]:
        if not claims:
            return []
        retrievals = self.retriever.retrieve_many(claims)
        premises = [" ".join(e.as_premise() for e in r.evidence) for r in retrievals]
        t0 = time.perf_counter()
        logits = self.verifier.logits(list(claims), premises)
        verify_ms = (time.perf_counter() - t0) * 1000.0 / len(claims)
        probs = softmax(logits, self.temperature)
        results: list[CheckResult] = []
        for claim, retrieval, row in zip(claims, retrievals, probs, strict=True):
            timings = {**retrieval.timings_ms, "verify_ms": verify_ms}
            results.append(
                CheckResult(claim, decide(row.tolist(), self.abstain_below),
                            retrieval.evidence, timings)
            )  # fmt: skip
        return results

    def check(self, claim: str) -> CheckResult:
        return self.check_many([claim])[0]


def build_retriever(config: WitnessConfig, device: str) -> EvidenceRetriever:
    from witness.retrieval.bm25 import BM25Index
    from witness.retrieval.linker import EntityLinker
    from witness.retrieval.reranker import CrossEncoderScorer
    from witness.wiki.store import WikiStore

    index_dir = config.index_dir
    store = WikiStore(index_dir / "wiki.sqlite")
    inlinks = np.load(index_dir / "inlinks.npy")
    linker = EntityLinker(store, inlinks, config.linker)
    bm25 = BM25Index.load(index_dir / "bm25")
    scorer = CrossEncoderScorer(config.reranker, device)
    return EvidenceRetriever(store, linker, bm25, scorer, config)


def build_checker(config: WitnessConfig) -> FactChecker:
    from witness.retrieval.reranker import resolve_device
    from witness.verify.verifier import NLIVerifier

    device = resolve_device(config.device)
    verifier_cfg = config.verifier
    verifier = NLIVerifier(
        verifier_cfg.model, device,
        max_length=verifier_cfg.max_length, batch_size=verifier_cfg.batch_size,
    )  # fmt: skip
    return FactChecker(
        build_retriever(config, device),
        verifier,
        temperature=load_temperature(Path(verifier_cfg.calibration_file)),
        abstain_below=verifier_cfg.abstain_below,
    )

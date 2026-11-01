"""FEVER evaluation with ablations computed from one retrieval pass.

Stage 1 (``retrieve``) runs the linker and BM25 once per claim, scores every sentence of
the union of their pages with the cross-encoder once, and stores the top evidence for
each retrieval variant. Stage 2 (``verify``) runs each verifier over the stored evidence.
Every ablation therefore shares the same candidates and cross-encoder scores, so
differences between rows come from the component being ablated and nothing else.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from witness.config import WitnessConfig
from witness.evaluation.fever_data import FeverClaim
from witness.evaluation.fever_scorer import fever_score
from witness.retrieval.bm25 import BM25Index, Tokenizer
from witness.retrieval.linker import EntityLinker
from witness.retrieval.pipeline import reciprocal_rank_fusion, sentence_pairs
from witness.retrieval.reranker import SentenceScorer
from witness.verify.calibration import expected_calibration_error, softmax
from witness.verify.verifier import LABELS, Verifier
from witness.wiki.store import Page, WikiStore

log = logging.getLogger(__name__)

VARIANTS = ("fused", "linker", "bm25", "fused_no_rerank")


def _batches(items: Sequence[FeverClaim], size: int) -> Iterator[Sequence[FeverClaim]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _lexical_scores(claim: str, origin: Sequence[tuple[Page, int]]) -> NDArray[np.float32]:
    """Stemmed-overlap score: the "no cross-encoder" ablation."""
    tokenize = Tokenizer()
    query = set(tokenize(claim))
    out = np.zeros(len(origin), dtype=np.float32)
    for i, (page, line) in enumerate(origin):
        terms = tokenize(page.title + " " + page.sentences[line])
        out[i] = len(query & set(terms)) / np.sqrt(max(len(terms), 1)) - 1e-3 * line
    return out


def _top(
    origin: Sequence[tuple[Page, int]], scores: NDArray[np.float32], allowed: set[int], k: int
) -> list[list[Any]]:
    keep = [i for i, (page, _) in enumerate(origin) if page.id in allowed]
    keep.sort(key=lambda i: (-float(scores[i]), i))
    return [
        [origin[i][0].raw_title, origin[i][1], origin[i][0].title,
         origin[i][0].sentences[origin[i][1]], round(float(scores[i]), 4)]
        for i in keep[:k]
    ]  # fmt: skip


def compute_candidates(
    claims: Sequence[FeverClaim],
    linker: EntityLinker,
    bm25: BM25Index,
    config: WitnessConfig,
    out_path: Path,
) -> None:
    """Stage 1a (CPU only): linker and BM25 page rankings per claim, with timings."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fh:
        for i, claim in enumerate(claims):
            t0 = time.perf_counter()
            linked = [p.page for p in linker.link(claim.claim, config.linker.top_k)]
            t1 = time.perf_counter()
            hits = [d for d, _ in bm25.search(claim.claim, config.bm25.top_k)]
            t2 = time.perf_counter()
            row = {
                "id": claim.id, "linker": linked, "bm25": hits,
                "timing_ms": {"linker": (t1 - t0) * 1e3, "bm25": (t2 - t1) * 1e3},
            }  # fmt: skip
            fh.write(json.dumps(row) + "\n")
            if i % 2000 == 0:
                log.info("candidates %d/%d (%.0fs)", i, len(claims), time.perf_counter() - started)


def load_candidates(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return {int(row["id"]): row for row in map(json.loads, fh)}


def retrieve_split(
    claims: Sequence[FeverClaim],
    candidates: dict[int, dict[str, Any]],
    store: WikiStore,
    scorer: SentenceScorer,
    config: WitnessConfig,
    out_path: Path,
    batch_size: int = 64,
) -> None:
    """Stage 1b: fuse cached candidates, score their sentences, store per-variant evidence."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = config.reranker.evidence_k
    fusion = config.fusion
    started = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fh:
        for batch_no, batch in enumerate(_batches(claims, batch_size)):
            rows: list[dict[str, Any]] = []
            all_pairs: list[tuple[str, str]] = []
            spans: list[tuple[int, int, list[tuple[Page, int]]]] = []
            for claim in batch:
                cand = candidates[claim.id]
                linked, bm25 = list(cand["linker"]), list(cand["bm25"])
                fused = reciprocal_rank_fusion(
                    [(linked, fusion.linker_weight), (bm25, fusion.bm25_weight)],
                    k=fusion.rrf_k, top=fusion.top_pages,
                )  # fmt: skip
                union = list(dict.fromkeys(fused + linked + bm25))
                pages = store.pages(union)
                pairs, origin = sentence_pairs(
                    claim.claim, pages, config.reranker.max_sentences_per_page
                )
                spans.append((len(all_pairs), len(all_pairs) + len(pairs), origin))
                all_pairs.extend(pairs)
                titles = {p.id: p.raw_title for p in pages}
                rows.append({
                    "id": claim.id, "claim": claim.claim, "label": claim.label,
                    "evidence": [list(map(list, g)) for g in claim.raw_evidence],
                    "pages": {"linker": [titles[p] for p in linked],
                              "bm25": [titles[p] for p in bm25],
                              "fused": [titles[p] for p in fused]},
                    "_ids": {"linker": set(linked), "bm25": set(bm25), "fused": set(fused)},
                    "timing_ms": dict(cand["timing_ms"]),
                })  # fmt: skip
            t3 = time.perf_counter()
            scores = scorer.score(all_pairs)
            rerank_ms = (time.perf_counter() - t3) * 1e3 / max(len(batch), 1)
            for row, (a, b, origin) in zip(rows, spans, strict=True):
                ids = row.pop("_ids")
                claim_scores = scores[a:b]
                row["evidence_by_variant"] = {
                    "fused": _top(origin, claim_scores, ids["fused"], k),
                    "linker": _top(origin, claim_scores, ids["linker"], k),
                    "bm25": _top(origin, claim_scores, ids["bm25"], k),
                    "fused_no_rerank": _top(
                        origin, _lexical_scores(row["claim"], origin), ids["fused"], k
                    ),
                }
                row["timing_ms"]["rerank"] = rerank_ms
                row["n_candidate_sentences"] = b - a
                fh.write(json.dumps(row) + "\n")
            if batch_no % 10 == 0:
                done = min((batch_no + 1) * batch_size, len(claims))
                rate = done / (time.perf_counter() - started)
                log.info("retrieved %d/%d claims (%.1f claims/s)", done, len(claims), rate)


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def premises(records: Sequence[dict[str, Any]], variant: str) -> list[str]:
    return [" ".join(f"{e[2]}: {e[3]}" for e in r["evidence_by_variant"][variant]) for r in records]


def verifier_logits(
    verifier: Verifier, records: Sequence[dict[str, Any]], variant: str
) -> NDArray[np.float32]:
    return verifier.logits([r["claim"] for r in records], premises(records, variant))


def gold_labels(records: Sequence[dict[str, Any]]) -> NDArray[np.int64]:
    return np.asarray([LABELS.index(r["label"]) for r in records], dtype=np.int64)


def _doc_recall(records: Sequence[dict[str, Any]], variant: str) -> float:
    hits = total = 0
    for r in records:
        if r["label"] == "NOT ENOUGH INFO":
            continue
        total += 1
        pages = set(r["pages"]["fused" if variant == "fused_no_rerank" else variant])
        groups = [{e[2] for e in g} for g in r["evidence"]]
        hits += any(g <= pages for g in groups)
    return hits / max(total, 1)


def score_predictions(
    records: Sequence[dict[str, Any]],
    logits: NDArray[np.float32],
    variant: str,
    temperature: float = 1.0,
) -> dict[str, float]:
    """FEVER metrics, per-class F1 and calibration for one (verifier, variant) pair."""
    probs = softmax(logits, temperature)
    predicted = probs.argmax(axis=1)
    instances = [
        {
            "label": r["label"],
            "predicted_label": LABELS[int(p)],
            "predicted_evidence": [[e[0], e[1]] for e in r["evidence_by_variant"][variant]],
            "evidence": r["evidence"],
        }
        for r, p in zip(records, predicted, strict=True)
    ]
    scores = fever_score(instances)
    gold = gold_labels(records)
    out = {
        "fever_score": scores.fever_score,
        "label_accuracy": scores.label_accuracy,
        "evidence_recall@5": scores.evidence_recall,
        "evidence_precision@5": scores.evidence_precision,
        "doc_recall": _doc_recall(records, variant),
        "ece": expected_calibration_error(probs, gold),
        "n": float(len(records)),
    }
    for i, name in enumerate(("supports", "refutes", "nei")):
        tp = float(((predicted == i) & (gold == i)).sum())
        precision = tp / max(float((predicted == i).sum()), 1.0)
        recall = tp / max(float((gold == i).sum()), 1.0)
        out[f"f1_{name}"] = 2 * precision * recall / max(precision + recall, 1e-9)
    return out


def retrieval_only_metrics(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Doc recall and evidence recall@5 per variant (label-independent)."""
    out: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        verifiable = [r for r in records if r["label"] != "NOT ENOUGH INFO"]
        full = 0
        for r in verifiable:
            predicted = [[e[0], e[1]] for e in r["evidence_by_variant"][variant]]
            groups = [[[e[2], e[3]] for e in g] for g in r["evidence"]]
            full += any(all(s in predicted for s in g) for g in groups)
        out[variant] = {
            "doc_recall": _doc_recall(records, variant),
            "evidence_recall@5": full / max(len(verifiable), 1),
        }
    return out


def latency_summary(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("linker", "bm25", "rerank"):
        values = np.asarray([r["timing_ms"][key] for r in records])
        out[f"{key}_ms_mean"] = float(values.mean())
        out[f"{key}_ms_p95"] = float(np.percentile(values, 95))
    return out

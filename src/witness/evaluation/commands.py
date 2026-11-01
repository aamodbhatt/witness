"""Orchestration behind ``witness train-verifier`` and ``witness eval fever``."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from witness.config import WitnessConfig
from witness.errors import IndexStoreError
from witness.evaluation.fever_data import FeverClaim, load_claims, seeded_sample, split_train
from witness.evaluation.run_fever import (
    VARIANTS,
    compute_candidates,
    gold_labels,
    latency_summary,
    load_candidates,
    load_records,
    retrieval_only_metrics,
    retrieve_split,
    score_predictions,
    verifier_logits,
)
from witness.verify.calibration import (
    expected_calibration_error,
    fit_temperature,
    save_temperature,
    softmax,
)
from witness.verify.train_data import Example, ExampleBuilder

log = logging.getLogger(__name__)

_BUILDER: ExampleBuilder | None = None


def _init_builder(index_dir: str) -> None:
    global _BUILDER  # noqa: PLW0603 - one store connection per worker process
    _BUILDER = ExampleBuilder(Path(index_dir))


def _build_one(args: tuple[FeverClaim, int]) -> Example | None:
    assert _BUILDER is not None
    return _BUILDER.build(*args)


def make_examples(
    claims: Sequence[FeverClaim], index_dir: Path, seed: int, workers: int
) -> list[Example]:
    store_file = index_dir / "wiki.sqlite"
    if not store_file.is_file():
        # Checked here, not in the pool initializer: a failing initializer makes
        # multiprocessing respawn workers forever instead of raising.
        raise IndexStoreError(
            f"Wikipedia store not found at {store_file}. Training examples need the full "
            f"index: build it with `witness build-index --out {index_dir.as_posix()}`, or "
            "pass the --config whose index_dir you built"
        )
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_builder, initargs=(str(index_dir),)) as pool:
        built = pool.map(_build_one, [(c, seed) for c in claims], chunksize=64)
    return [e for e in built if e is not None]


def split_claims(raw_dir: Path, split: str, limit: int | None, seed: int) -> list[FeverClaim]:
    if split == "dev":
        claims = load_claims(raw_dir / "shared_task_dev.jsonl")
    elif split in ("train", "calib"):
        train, calib = split_train(load_claims(raw_dir / "train.jsonl"))
        claims = train if split == "train" else calib
    else:
        raise ValueError(f"unknown split {split!r}; expected dev, calib or train")
    return seeded_sample(claims, limit, seed)


def run_eval(
    config: WitnessConfig,
    *,
    split: str,
    raw_dir: Path,
    out_dir: Path,
    limit: int | None,
    verifiers: dict[str, str],
    main_verifier: str,
    stages: set[str],
    temperatures_file: Path | None,
    device: str,
) -> dict[str, Any]:
    """Run the requested stages: ``candidates`` (CPU), ``retrieve`` and ``verify`` (GPU)."""
    from witness.retrieval.bm25 import BM25Index
    from witness.retrieval.linker import EntityLinker
    from witness.retrieval.reranker import CrossEncoderScorer
    from witness.verify.verifier import NLIVerifier
    from witness.wiki.store import WikiStore

    unknown = stages - {"candidates", "retrieve", "verify"}
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}")
    candidates_path = out_dir / "candidates.jsonl"
    records_path = out_dir / "retrieval.jsonl"
    store = WikiStore(config.index_dir / "wiki.sqlite")
    if "candidates" in stages or "retrieve" in stages:
        claims = split_claims(raw_dir, split, limit, config.seed)
        if "candidates" in stages:
            log.info("page candidates for %d %s claims", len(claims), split)
            linker = EntityLinker(store, np.load(config.index_dir / "inlinks.npy"), config.linker)
            bm25 = BM25Index.load(config.index_dir / "bm25")
            compute_candidates(claims, linker, bm25, config, candidates_path)
        if "retrieve" in stages:
            log.info("sentence reranking for %d %s claims", len(claims), split)
            scorer = CrossEncoderScorer(config.reranker, device)
            retrieve_split(claims, load_candidates(candidates_path), store, scorer, config,
                           records_path)  # fmt: skip
    if "verify" not in stages:
        return {"split": split, "stages": sorted(stages)}
    records = load_records(records_path)
    labels = gold_labels(records)

    temperatures: dict[str, float] = {}
    if temperatures_file is not None and temperatures_file.is_file():
        temperatures = json.loads(temperatures_file.read_text(encoding="utf-8"))

    results: dict[str, Any] = {
        "split": split,
        "claims": len(records),
        "retrieval": retrieval_only_metrics(records),
        "latency": latency_summary(records),
        "verifiers": {},
    }
    fitted: dict[str, float] = {}
    for name, model in verifiers.items():
        verifier = NLIVerifier(
            model, device, max_length=config.verifier.max_length,
            batch_size=config.verifier.batch_size,
        )  # fmt: skip
        variants = VARIANTS if name == main_verifier else ("fused",)
        per_variant: dict[str, Any] = {}
        for variant in variants:
            logits = verifier_logits(verifier, records, variant)
            np.save(out_dir / f"logits_{name}_{variant}.npy", logits)
            temperature = temperatures.get(name, 1.0)
            per_variant[variant] = score_predictions(records, logits, variant, temperature)
            if variant == "fused":
                t_fit = fit_temperature(logits, labels)
                fitted[name] = t_fit
                per_variant[variant]["ece_uncalibrated"] = expected_calibration_error(
                    softmax(logits), labels
                )
                per_variant[variant]["temperature_used"] = temperature
        results["verifiers"][name] = per_variant
        del verifier
    if split == "calib":
        (out_dir / "temperatures.json").write_text(json.dumps(fitted, indent=2), "utf-8")
        if main_verifier in fitted and Path(verifiers[main_verifier]).is_dir():
            save_temperature(
                Path(verifiers[main_verifier]) / "calibration.json",
                fitted[main_verifier], fitted_on="FEVER train calibration bucket",
                n=len(records),
            )  # fmt: skip
        results["fitted_temperatures"] = fitted
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results

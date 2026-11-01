"""Turn evaluation runs into the committed results: JSON, a Markdown table and a plot.

uv run --group docs python scripts/report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HARDWARE = "single NVIDIA A40 (48 GB), ~8 vCPU, 50 GB RAM"
VERIFIER_NAMES = {
    "finetuned": "DeBERTa-v3-base, fine-tuned here (calibrated)",
    "nli_mnli_fever_anli": "DeBERTa-v3-base MNLI+FEVER-NLI+ANLI, off-the-shelf",
    "nli_mnli_snli": "DeBERTa-v3-base SNLI+MNLI, off-the-shelf (zero-shot)",
}
VARIANT_NAMES = {
    "fused": "linker ∪ BM25 → cross-encoder (full)",
    "linker": "entity linker only → cross-encoder",
    "bm25": "BM25 only → cross-encoder",
    "fused_no_rerank": "linker ∪ BM25, lexical sentence ranking",
}


def _load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _pct(x: float) -> str:
    return f"{100 * x:.2f}"


def markdown(dev: dict[str, Any], calib: dict[str, Any], build: dict[str, Any],
             train: dict[str, Any]) -> str:  # fmt: skip
    v = dev["verifiers"]
    lines = [
        f"# FEVER results ({int(dev['claims'])} claims of `shared_task_dev`)",
        "",
        f"Hardware: {HARDWARE}. Temperatures fit on {int(calib['claims'])} held-out "
        "FEVER *train* claims (never on dev).",
        "",
        "## Verifiers (full retrieval)",
        "",
        "| Verifier | Label acc. | FEVER score | F1 S / R / NEI | ECE before → after T |",
        "|---|---:|---:|---|---|",
    ]
    for name, label in VERIFIER_NAMES.items():
        if name not in v:
            continue
        m = v[name]["fused"]
        f1 = " / ".join(_pct(m[f"f1_{k}"]) for k in ("supports", "refutes", "nei"))
        lines.append(
            f"| {label} | {_pct(m['label_accuracy'])} | {_pct(m['fever_score'])} | {f1} | "
            f"{m['ece_uncalibrated']:.3f} → {m['ece']:.3f} (T={m['temperature_used']:.2f}) |"
        )
    lines += [
        "",
        "## Retrieval ablations (fine-tuned verifier)",
        "",
        "| Retrieval | Doc recall | Evidence recall@5 | Label acc. | FEVER score |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, label in VARIANT_NAMES.items():
        m = v["finetuned"][variant]
        r = dev["retrieval"][variant]
        lines.append(
            f"| {label} | {_pct(r['doc_recall'])} | {_pct(r['evidence_recall@5'])} | "
            f"{_pct(m['label_accuracy'])} | {_pct(m['fever_score'])} |"
        )
    lat = dev["latency"]
    lines += [
        "",
        "## Latency per claim (batched evaluation run)",
        "",
        f"- entity linking: {lat['linker_ms_mean']:.1f} ms mean, {lat['linker_ms_p95']:.1f} ms p95",
        f"- BM25 over {int(build['pages']):,} pages: {lat['bm25_ms_mean']:.1f} ms mean, "
        f"{lat['bm25_ms_p95']:.1f} ms p95",
        f"- page fetch + cross-encoder reranking: {lat['rerank_ms_mean']:.1f} ms mean",
        "",
        "## Index",
        "",
        f"- {int(build['pages']):,} non-empty pages, {int(build['aliases']):,} aliases, "
        f"{int(build['bm25_postings']):,} BM25 postings over {int(build['bm25_vocab']):,} terms",
        f"- store {build['store_bytes'] / 1e9:.2f} GB, BM25 {build['bm25_bytes'] / 1e9:.2f} GB, "
        f"built in {build['total_seconds'] / 60:.1f} min on CPU",
        "",
        "## Verifier training",
        "",
        f"- {int(train['examples']):,} examples, {int(train['steps'])} steps, "
        f"{train['train_seconds'] / 60:.1f} min on the A40 ({train['examples_per_second']} ex/s), "
        f"peak {train['peak_vram_gb']} GB VRAM",
    ]
    return "\n".join(lines) + "\n"


def plot(dev: dict[str, Any], out: Path) -> None:
    v = dev["verifiers"]
    rows: list[tuple[str, float, float]] = []
    for variant in ("fused", "linker", "bm25", "fused_no_rerank"):
        m = v["finetuned"][variant]
        rows.append((VARIANT_NAMES[variant], m["label_accuracy"], m["fever_score"]))
    for name in ("nli_mnli_fever_anli", "nli_mnli_snli"):
        if name in v:
            m = v[name]["fused"]
            rows.append((VERIFIER_NAMES[name], m["label_accuracy"], m["fever_score"]))
    rows.reverse()
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    ys = range(len(rows))
    height = 0.38
    ax.barh([y + height / 2 for y in ys], [100 * r[1] for r in rows], height,
            color="#4C78A8", label="Label accuracy")  # fmt: skip
    ax.barh([y - height / 2 for y in ys], [100 * r[2] for r in rows], height,
            color="#F58518", label="FEVER score")  # fmt: skip
    for y, r in zip(ys, rows, strict=True):
        ax.text(100 * r[1] + 0.4, y + height / 2, f"{100 * r[1]:.1f}", va="center", fontsize=8)
        ax.text(100 * r[2] + 0.4, y - height / 2, f"{100 * r[2]:.1f}", va="center", fontsize=8)
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("% on FEVER shared_task_dev")
    low = min(min(r[1], r[2]) for r in rows)
    ax.set_xlim(max(0, 100 * low - 15), 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("witness ablations: retrieval components and verifiers", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)


def main() -> None:
    dev = _load(ROOT / "runs/fever_dev/metrics.json")
    calib = _load(ROOT / "runs/fever_calib/metrics.json")
    build = _load(ROOT / "data/index/full/build_stats.json")
    train = _load(ROOT / "artifacts/verifier/train_stats.json")
    results = ROOT / "docs/results"
    results.mkdir(parents=True, exist_ok=True)
    payload = {"hardware": HARDWARE, "dev": dev, "calibration": calib, "index": build,
               "verifier_training": train}  # fmt: skip
    (results / "fever_dev.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (results / "fever_dev.md").write_text(markdown(dev, calib, build, train), encoding="utf-8")
    assets = ROOT / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    plot(dev, assets / "ablation.png")
    print((results / "fever_dev.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

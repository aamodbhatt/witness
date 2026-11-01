"""Per-page latency through the running server, and claim-extraction throughput.

uv run python scripts/bench_page.py --server http://127.0.0.1:8094
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from witness.claims.extract import extract_claims
from witness.claims.readable import readable_text
from witness.config import ClaimsConfig

ROOT = Path(__file__).resolve().parents[1]


def extraction_throughput(paragraphs: list[str], title: str | None, repeats: int) -> float:
    """Sentences per second through segmentation + filtering + decontextualisation."""
    config = ClaimsConfig(max_claims_per_page=10**9)
    corpus = paragraphs * repeats
    started = time.perf_counter()
    claims, skipped = extract_claims(corpus, title, config)
    elapsed = time.perf_counter() - started
    return (len(claims) + len(skipped)) / elapsed


def page_latency(server: str, title: str | None, paragraphs: list[str]) -> dict[str, float]:
    with httpx.Client(timeout=120.0) as client:
        started = time.perf_counter()
        claims = client.post(f"{server}/claims", json={"title": title, "paragraphs": paragraphs})
        claims.raise_for_status()
        found = claims.json()["claims"]
        extracted = time.perf_counter()
        first: float | None = None
        verdicts: list[dict[str, object]] = []
        body = {"claims": [{"id": c["id"], "text": c["text"], "query": c["query"]} for c in found]}
        with client.stream("POST", f"{server}/check", json=body) as stream:
            for line in stream.iter_lines():
                if line.startswith("data: ") and '"verdict"' in line:
                    first = first or time.perf_counter()
                    verdicts.append(json.loads(line[6:]))
        done = time.perf_counter()
    return {
        "claims": float(len(found)),
        "extract_ms": (extracted - started) * 1e3,
        "first_verdict_ms": ((first or done) - started) * 1e3,
        "page_ms": (done - started) * 1e3,
        "per_claim_ms": (done - extracted) * 1e3 / max(len(found), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8094")
    parser.add_argument("--page", type=Path, default=ROOT / "docs/demo/marie-curie.html")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/results/page_latency.json")
    args = parser.parse_args()
    page = readable_text(args.page.read_text(encoding="utf-8"))
    latency = page_latency(args.server, page.title, page.paragraphs)
    # Warm run measured second so model loading and CUDA kernels are not counted.
    latency = page_latency(args.server, page.title, page.paragraphs)
    result = {
        "page": args.page.name,
        "latency": latency,
        "extraction_sentences_per_second": extraction_throughput(
            page.paragraphs, page.title, repeats=400
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

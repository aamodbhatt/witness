# FEVER results (19998 claims of `shared_task_dev`)

Hardware: single NVIDIA A40 (48 GB), ~8 vCPU, 50 GB RAM. Temperatures fit on 2000 held-out FEVER *train* claims (never on dev).

## Verifiers (full retrieval)

| Verifier | Label acc. | FEVER score | F1 S / R / NEI | ECE before → after T |
|---|---:|---:|---|---|
| DeBERTa-v3-base, fine-tuned here (calibrated) | 76.41 | 70.29 | 83.64 / 79.30 / 63.86 | 0.125 → 0.050 (T=1.53) |
| DeBERTa-v3-base MNLI+FEVER-NLI+ANLI, off-the-shelf | 71.70 | 65.17 | 80.98 / 75.38 / 52.78 | 0.226 → 0.085 (T=2.43) |
| DeBERTa-v3-base SNLI+MNLI, off-the-shelf (zero-shot) | 63.29 | 57.55 | 70.91 / 69.59 / 45.61 | 0.324 → 0.052 (T=4.11) |

## Retrieval ablations (fine-tuned verifier)

| Retrieval | Doc recall | Evidence recall@5 | Label acc. | FEVER score |
|---|---:|---:|---:|---:|
| linker ∪ BM25 → cross-encoder (full) | 90.29 | 85.78 | 76.41 | 70.29 |
| entity linker only → cross-encoder | 86.13 | 83.75 | 76.01 | 71.02 |
| BM25 only → cross-encoder | 64.29 | 62.37 | 73.19 | 57.76 |
| linker ∪ BM25, lexical sentence ranking | 90.29 | 61.13 | 72.74 | 59.30 |

## Latency per claim (batched evaluation run)

- entity linking: 1.3 ms mean, 3.0 ms p95
- BM25 over 5,396,106 pages: 14.4 ms mean, 37.3 ms p95
- page fetch + cross-encoder reranking: 17.6 ms mean

## Index

- 5,396,106 non-empty pages, 9,584,579 aliases, 212,077,687 BM25 postings over 2,645,184 terms
- store 5.02 GB, BM25 1.74 GB, built in 8.1 min on CPU

## Verifier training

- 50,573 examples, 1581 steps, 11.3 min on the A40 (74.9 ex/s), peak 14.19 GB VRAM

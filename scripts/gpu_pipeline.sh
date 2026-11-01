#!/usr/bin/env bash
# Every GPU step of the published results, in one gpu-run slot:
#   1. cross-encoder sentence reranking for the calibration split and FEVER dev,
#   2. verifier fine-tuning,
#   3. verification with the fine-tuned and the off-the-shelf models,
#   4. temperature fit on the calibration split, applied to dev.
# Page candidates (CPU) must exist first: `witness eval fever --stage candidates`.
# Set SKIP_RERANK=1 to reuse existing runs/fever_*/retrieval.jsonl.
set -euo pipefail
CFG=configs/pod.toml
VERIFIERS=(
  --verifier finetuned=artifacts/verifier
  --verifier nli_mnli_fever_anli=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
  --verifier nli_mnli_snli=cross-encoder/nli-deberta-v3-base
)
if [ "${SKIP_RERANK:-0}" != 1 ]; then
  uv run witness eval fever --split calib --limit 2000 --stage retrieve --config "$CFG"
  uv run witness eval fever --split dev --stage retrieve --config "$CFG"
fi
uv run witness train-verifier --stage train --config "$CFG"
uv run witness eval fever --split calib --stage verify --config "$CFG" "${VERIFIERS[@]}"
uv run witness eval fever --split dev --stage verify --config "$CFG" "${VERIFIERS[@]}" \
  --temperatures runs/fever_calib/temperatures.json

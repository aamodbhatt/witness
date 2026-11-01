#!/usr/bin/env bash
# README session: serve the real models, record the GIF, measure page latency, capture
# CLI output, then stop the server. CFG=configs/cpu.toml keeps it off the GPU.
set -euo pipefail
CFG=${CFG:-configs/pod.toml}
mkdir -p runs/demo docs/assets docs/results
uv run witness serve --config "$CFG" > runs/demo/server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8094/health | grep -q '"ok"'; then break; fi
  sleep 1
done
curl -sf http://127.0.0.1:8094/health
uv run --group docs python scripts/record_demo.py --server http://127.0.0.1:8094
uv run python scripts/bench_page.py --server http://127.0.0.1:8094
uv run witness check --config "$CFG" \
  "Marie Curie was born in Warsaw." \
  "The Eiffel Tower is in Berlin." \
  "Barack Obama was the 44th President of the United States." \
  "Tilda Swinton is a vegan." | tee runs/demo/cli_check.txt

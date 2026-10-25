#!/usr/bin/env bash
# M6: per-hop compute latency of every exported streaming model with the C++ engine (desktop).
#   bash bench/rt_bench.sh [seconds=120]
# Writes results/m6/bench/<name>.json (idle-ish host) and <name>_contended.json (+8 busy threads).
set -euo pipefail
cd "$(dirname "$0")/.."
SECS=${1:-120}
EXP=data/cache/export
OUT=results/m6/bench
mkdir -p "$OUT"
make -s rt >/dev/null
for m in "$EXP"/b*_s0.onnx; do
  [[ "$m" == *_film.onnx ]] && continue
  n=$(basename "$m" .onnx)
  ./rt/build/clarity_bench --model "$m" --seconds "$SECS" --rt --json "$OUT/$n.raw.json" >/dev/null
  ./rt/build/clarity_bench --model "$m" --seconds "$SECS" --rt --contention 8 --json "$OUT/${n}_contended.raw.json" >/dev/null
  for f in "$OUT/$n.raw.json" "$OUT/${n}_contended.raw.json"; do
    uv run python -m bench.provenance_wrap "$f" "${f%.raw.json}.json" --ckpt "data/cache/ckpt/$n.pt" --onnx "$m"
    rm "$f"
  done
  echo "$n: $(uv run python -c "import json;d=json.load(open('$OUT/$n.json'));c=d['compute'];print(f\"p50 {c['p50_ms']:.3f} p99 {c['p99_ms']:.3f} p99.9 {c['p999_ms']:.3f} ms, hop {d['hop_ms']} ms\")")"
done

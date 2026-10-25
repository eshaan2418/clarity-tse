#!/usr/bin/env bash
# make reproduce-quick: < 1 h. Works from a clean clone (fixtures only); if the licence-open data
# and trained checkpoints are present it also verifies frozen test-set checksums, re-scores a
# subset with the cached checkpoints and regenerates every figure.
set -euo pipefail
cd "$(dirname "$0")/.."
t0=$(date +%s)
uv sync --frozen
make lint typecheck test rt-test
make train-quick-fixture
uv run pytest -q tests/test_latency_regression.py
make study-dryrun
if [ -d data/external/LibriSpeech ] && [ -d data/cache/testsets/test_main ]; then
  echo "== verifying frozen test-set checksums (first 8 items re-rendered)"
  uv run python -c "from clarity_tse.synth import testsets as t; b=t.verify('test_main', 8); print('mismatch', b); assert not b"
  if [ -f data/cache/ckpt/b16_s0.pt ]; then
    uv run python -m bench.evaluate --set test_main --systems unprocessed model:b16_s0 --limit 24 --no-haspi \
      --out results/quick/eval_subset.csv --jobs 8 --recompute
  fi
  [ -f results/m3/summary.json ] && uv run python -m bench.make_figures
else
  echo "== data not present (clean clone): skipped data-dependent steps; run 'make data' for them"
fi
uv run python scripts/check_numbers.py
bash scripts/scan_repo.sh
echo "reproduce-quick OK in $(( $(date +%s) - t0 )) s"

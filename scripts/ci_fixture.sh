#!/usr/bin/env bash
# CI: end-to-end tiny train + eval on licence-free fixtures (no downloads, no gated data).
set -euo pipefail
cd "$(dirname "$0")/.."
export CLARITY_EXTERNAL=$PWD/data/fixtures/external CLARITY_CACHE=$PWD/data/fixtures/cache CLARITY_EMBEDDER=mfcc OMP_NUM_THREADS=1
PY="uv run python -W ignore"
[ -d "$CLARITY_EXTERNAL/LibriSpeech" ] || $PY scripts/make_fixtures.py --out "$CLARITY_EXTERNAL"
[ -f results/ci/testsets/fixture_main.json ] || $PY -m bench.make_testsets --sets fixture_main --jobs 2
$PY -m bench.build_rir_bank --n 8 --jobs 2
$PY -m bench.train --config configs/ci_tiny.yaml --out results/ci/runs
$PY -m bench.evaluate --set fixture_main --systems unprocessed ds model:ci_tiny --no-haspi --out results/ci/eval.csv --jobs 2
echo "ci fixture pipeline OK"

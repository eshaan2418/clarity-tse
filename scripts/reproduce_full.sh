#!/usr/bin/env bash
# make reproduce-full: every experiment from raw licence-open data (~1-2 days on the M0 host).
set -euo pipefail
cd "$(dirname "$0")/.."
J=${JOBS:-12}
make data
uv run python -m bench.make_testsets --jobs "$J"
EMB_DEVICE=${EMB_DEVICE:-mps} uv run python scripts/precompute_embeddings.py train val test
uv run python -m bench.build_rir_bank --n 4000 --jobs "$J"
uv run python -m bench.build_train_pool --jobs "$J"
uv run python -m bench.make_configs
bash bench/train_sweep.sh
uv run python -m bench.export_onnx --ckpt $(ls results/m3/runs | grep -E '^b[0-9]+_s0$')
uv run python -m bench.export_ecapa
bash bench/rt_bench.sh 120
uv run python -m bench.run_baselines --jobs "$J"
uv run python -m bench.eval_streaming --jobs "$J"
uv run python -m bench.run_selection --ckpt b16_s0 --jobs "$J"
uv run python -m bench.train_rho --ckpt b16_s0
uv run python -m bench.eval_personalize --ckpt b16_s0 --jobs "$J" --hasqi
uv run python -m bench.energy_estimate
make rt-soak demo study-dryrun
uv run python -m bench.make_figures
uv run python scripts/check_numbers.py

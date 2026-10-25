# Clarity — see README.md. All targets run from repo root.
SHELL := /bin/bash
PY := uv run python
export OMP_NUM_THREADS ?= 1
JOBS ?= 12

.PHONY: help setup data fixtures test test-all lint typecheck rt rt-test ci testsets baselines \
        train-quick train eval rt-bench rt-soak android android-install android-latency demo \
        study-dryrun study-serve study-analyze reproduce-quick reproduce-full paper check-numbers scan

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1 | tr '\n' ' '; echo

setup:            ## install pinned python deps
	uv sync --frozen
data:             ## download licence-open corpora (LibriSpeech, ESC-50, DNSMOS, RNNoise)
	bash scripts/get_data.sh
fixtures:
	$(PY) scripts/make_fixtures.py --out data/fixtures/external

test:             ## fast unit tests on fixtures (no external data needed)
	uv run pytest -q -m "not slow"
test-all:
	uv run pytest -q -m "slow or not slow"
lint:
	uv run ruff check clarity_tse tests scripts bench study
typecheck:
	uv run mypy
rt:
	cmake -S rt -B rt/build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build rt/build -j
rt-test: rt
	ctest --test-dir rt/build --output-on-failure
ci: lint typecheck test rt-test train-quick-fixture   ## what .github/workflows/ci.yml runs

# ---- experiments (require `make data`) ----
testsets:         ## M1: frozen, checksummed test sets
	$(PY) -m bench.make_testsets --jobs $(JOBS)
baselines:        ## M2
	$(PY) -m bench.run_baselines --jobs $(JOBS)
train-quick-fixture:  ## tiny train+eval on fixtures (CI)
	bash scripts/ci_fixture.sh
unused-old-fixture:
	CLARITY_EXTERNAL=data/fixtures/external CLARITY_CACHE=data/fixtures/cache $(PY) -m bench.train --config configs/ci_tiny.yaml --out results/ci
train-quick:
	$(PY) -m bench.train --config configs/quick.yaml --out results/quick
train:            ## M3 latency sweep (all budgets, 3 seeds)
	bash bench/train_sweep.sh
eval:
	$(PY) -m bench.eval_streaming --jobs $(JOBS)
rt-bench:         ## M6 compute latency on desktop
	$(MAKE) rt && ./rt/build/clarity_bench --model results/m3/export/clarity_10ms.onnx --minutes 1
rt-soak:          ## M6 60-minute soak (CoreAudio clock, muted output)
	$(MAKE) rt && ./rt/build/clarity_rt --model results/m3/export/clarity_10ms.onnx --minutes 60 --mute --json results/m6/soak_60min.json
android:
	cd android && ./gradlew assembleDebug testDebugUnitTest
demo:
	$(PY) -m bench.demo
study-dryrun:
	$(PY) -m study.simulate && $(PY) -m study.analyze --data study/data_simulated
study-serve:
	$(PY) -m study.server
study-analyze:
	$(PY) -m study.analyze --data study/data
reproduce-quick:  ## <1h: fixtures + subset
	bash scripts/reproduce_quick.sh
reproduce-full:
	bash scripts/reproduce_full.sh
paper:
	$(PY) -m bench.make_figures && cd paper && tectonic main.tex
check-numbers:    ## every number in README/paper/bullets must trace to results/
	$(PY) scripts/check_numbers.py
scan:             ## secrets + large/personal file scan of git history
	bash scripts/scan_repo.sh

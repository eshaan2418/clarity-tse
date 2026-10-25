# Clarity — see README.md. All targets run from repo root.
SHELL := /bin/bash
PY := uv run python
export OMP_NUM_THREADS ?= 1
# never re-lock implicitly (e.g. when a machine-level UV_INDEX_URL differs)
export UV_FROZEN ?= 1
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
train-quick:      ## 1 short run of the operating-point architecture (smoke test on real data)
	$(PY) -m bench.train --config configs/m3/gen/b16_s0.yaml --out results/quick/runs --override '{"name":"quick_b16","train":{"steps":300,"eval_every":300},"loss":{"w_kd":0}}'

train:            ## M3 latency sweep (all budgets, 3 seeds)
	bash bench/train_sweep.sh
eval:             ## M3 quality-vs-latency curve + ablations (+ M4/M5 below)
	$(PY) -m bench.eval_streaming --jobs $(JOBS)
eval-select:
	$(PY) -m bench.run_selection --ckpt $(OP) --jobs $(JOBS)
eval-personalize:
	$(PY) -m bench.train_rho --ckpt $(OP) && $(PY) -m bench.eval_personalize --ckpt $(OP) --jobs $(JOBS) --hasqi
eval-real:        ## M9 (needs consented recordings in data/personal/raw/)
	$(PY) -m bench.eval_real --ckpt $(OP)
figures:
	$(PY) -m bench.make_figures
rt-bench:         ## M6 per-hop compute latency of every exported model (desktop)
	bash bench/rt_bench.sh 120
rt-soak:          ## M6 60-minute soak on the CoreAudio clock (muted output) at the operating point
	$(MAKE) rt && ./rt/build/clarity_rt --model data/cache/export/$(OP).onnx --minutes 60 --mute --json results/m6/soak_60min.raw.json \
	  && $(PY) -m bench.provenance_wrap results/m6/soak_60min.raw.json results/m6/soak_60min.json --ckpt data/cache/ckpt/$(OP).pt --onnx data/cache/export/$(OP).onnx \
	  && rm results/m6/soak_60min.raw.json
OP ?= b16_s0
ANDROID_ENV := JAVA_HOME=$${JAVA_HOME:-/opt/homebrew/opt/openjdk@17} ANDROID_HOME=$${ANDROID_HOME:-$$HOME/Library/Android/sdk}
android-assets:   ## bundle the operating-point model + enrollment embedder into the APK
	mkdir -p android/app/src/main/assets
	cp data/cache/export/$(OP).onnx android/app/src/main/assets/clarity.onnx
	cp data/cache/export/$(OP).cfg android/app/src/main/assets/clarity.cfg
	cp data/cache/export/$(OP)_film.onnx android/app/src/main/assets/clarity_film.onnx
	cp data/cache/export/ecapa.onnx android/app/src/main/assets/ecapa.onnx
android:
	cd android && $(ANDROID_ENV) ./gradlew --no-daemon assembleDebug testDebugUnitTest
android-install: android
	adb install -r android/app/build/outputs/apk/debug/app-debug.apk
android-test-device:  ## instrumented tests (device or emulator)
	cd android && $(ANDROID_ENV) ./gradlew --no-daemon connectedDebugAndroidTest
android-latency:  ## launch loopback measurement on a connected phone (result shown in app + logcat)
	adb shell am start -n org.clarity.app/.MainActivity --ez measure_latency true
energy:           ## M8: estimated energy operating points (+ parses on-device sessions if any)
	$(PY) -m bench.energy_estimate
demo:
	$(PY) -m bench.demo --model $(OP)
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

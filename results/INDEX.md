# results/INDEX.md — every experiment: command → output → one-line finding

All files carry a `provenance` block (git commit, hardware, evidence stage, seed, checkpoint SHA-256,
config, timestamp, load average). Evidence stages: `desktop` (Apple M5 Pro host), `estimated`,
`simulated` (simulated listeners / participants), `on-device` (none yet: no phone, BLOCKERS B5).

| Milestone | Command | Output | Finding |
|---|---|---|---|
| M1 | `make testsets` | `m1/testsets/{test_main,val_main}.json` | 240 test / 80 val scenes (4 types × 5 SNRs), per-array SHA-256 |
| M2 | `make baselines` | `m2/per_item.csv`, `m2/summary.{json,md}` | 2-mic beamformers give small ESTOI/HASPI gains; RNNoise lowers intelligibility metrics in babble while raising DNSMOS |
| M3 | `bash bench/train_sweep.sh`, `make eval` | `m3/runs/*/summary.json`, `m3/per_item.csv`, `m3/summary.{json,md}` | quality-vs-latency curve, seeds, ablations (see summary.md) |
| M4 | `python -m bench.tune_selection` (val), `make eval-select`, `python -m bench.summarize_selection` | `m4/tuned_policy.json`, `m4/per_scene.csv`, `m4/summary_ci.{json,md}` | full policy +0.111 ESTOI over unprocessed, 0.037 below oracle selection; IMU did **not** help in simulation (−imu better) |
| M5 | `make eval-personalize` | `m5/train_rho_b16_s0.json`, `m5/per_item.csv`, `m5/summary.json` | (simulated listener) audiogram-conditioned ρ beats enh+NAL-R only for moderate flat loss; worse for NH/mild; per-listener fine-tuning adds nothing |
| M6 | `make rt-bench`, `make rt-soak` | `m6/bench/*.json`, `m6/soak_60min*.json` | per-hop compute p99 2.5 ms (8 ms hop) on CoreAudio clock; loaded-host soak had 14 xruns/60 min (see unloaded rerun) |
| M7 | `make android`, `make android-test-device`, `python -m bench.system_latency` | `m7/system_latency.json`, `m7/android_build.json` | app builds; instrumented tests pass on emulator; system latency `estimated` only |
| M8 | `make energy` | `m8/energy.{json,md}` | `estimated` battery brackets (assumption ranges) |
| M9 | `make eval-real` / `python -m bench.eval_real --validate-on-study` | `m9/summary_validation_simulated.json` | blocked-with-fallback (no consented recordings); WER pipeline validated on simulated study stimuli |
| M10 | `make study-dryrun`, `python -m study.simulate --power` | `m10/analysis_simulated.json`, `m10/power_simulated.json` | analysis recovers an assumed ordering; power for planning only (assumed effects) |
| demo | `make demo` | `demo/demo.json`, `demo/audio/synthetic_demo_*.wav` | faced scene: ESTOI gain; off-axis scene: documented selection failure |

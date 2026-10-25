# PROGRESS

Legend: todo / in-progress / done / blocked-with-fallback / awaiting-human

| Milestone | Status | Evidence |
|---|---|---|
| M0 Bootstrap | done | `make test`, `make rt-test`, ENVIRONMENT.md, ruff+mypy clean |
| M1 Simulator & eval harness | done | `results/m1/testsets/{test_main,val_main}.json` (240 / 80 items, sha256 per array), `tests/test_metrics.py`, `tests/test_latency_regression.py` |
| M2 Baselines & offline ceiling | in-progress | baselines running → `results/m2/` |
| M3 Streaming extractor | in-progress | code + streaming==batch tests done; training pending ECAPA tables |
| M4 Target selection | todo (code done) | `clarity_tse/select/policy.py`, `bench/run_selection.py` |
| M5 Personalization | todo (code done) | `clarity_tse/personalize/`, `bench/train_rho.py`, `bench/eval_personalize.py` |
| M6 RT engine | todo (code done) | `rt/` builds, core tests pass |
| M7 Android | in-progress | sources done; gradle wrapper added; first build running |
| M8 Energy | todo | |
| M9 Real recordings | blocked-with-fallback | B6: no personal recordings |
| M10 Study package | in-progress | `study/` + simulated dry run `results/m10/analysis_simulated.json` |
| M11 Write-up | todo | |

## Single next action
Finish ECAPA utterance tables → train offline teacher `offline_s0` → `b10_s0`.

## Compute plan (DECISIONS D12, D15)
Serial GPU queue (`bench/train_sweep.sh`), CPU evaluation in parallel.

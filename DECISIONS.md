# DECISIONS

## 2026-10-24
- **D1 Python package name `clarity_tse`** (not `clarity`): pyclarity (used for HASPI/HASQI/MSBG/NAL-R reference implementations) owns the import name `clarity`. The prompt's layout (`dsp/`, `embed/`, `models/` ...) lives under `clarity_tse/`; `rt/`, `android/`, `study/`, `bench/`, `paper/`, `results/` stay at repo root.
- **D2 Compute rung: "Fallback 1" (smaller accelerator).** Apple M5 Pro MPS. Models are kept small (<~2 M params) and training budgets short; all model results labelled *reduced-scale*.
- **D3 Data rung: "Fallback 1".** LibriSpeech (CC BY 4.0) speech + ESC-50 environmental noise + synthetic babble/music-like noise + pyroomacoustics rooms. Clarity Challenge data is gated and not present.
- **D4 Sample rate 16 kHz** for all models/metrics (PESQ-WB, ESTOI, DNSMOS operate at 16 kHz; HASPI resamples internally to 24 kHz).
- **D5 Multi-mic rung: "Fallback 1" — simulated 2-mic binaural earbud pair** (ear spacing 0.18 m, free-field mics, no head-shadow HRTF) rendered with pyroomacoustics image-source RIRs. No HRTF dataset with a clean licence was integrated; limitation documented (no ILD from head shadow; ITD/phase cues only).
- **D6 Head-orientation rung: "Fallback 1" — simulated yaw trajectories**; head rotation renders by crossfading RIR sets computed on a yaw grid.
- **D7 Real-time rung: "Fallback 1" — desktop C++ engine** (CoreAudio output clock on macOS, file-injected input) + offline streaming simulation with measured per-hop compute. Android app is built but on-device numbers are unavailable without a device.
- **D8 Bluetooth latency: "Fallback 1" — published ranges, labelled `estimated`.**
- **D9 Speaker splits:** train speakers = train-clean-100; val = dev-clean; test = test-clean. LibriSpeech subsets are speaker-disjoint by construction. ESC-50 folds 1–3 train, 4 val, 5 test; room parameters drawn from disjoint seeded RNG streams and disjoint RT60 sub-ranges are not used (same distribution, different draws) — documented as a threat.

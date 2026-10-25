# HUMAN_TODO

Things only a human can do. Each step says exactly what to run; results land in `results/` with
provenance and are picked up by `make figures` / `scripts/check_numbers.py`.

## 1. (Optional) Clarity Challenge data
Register at claritychallenge.org, download CEC2, set `CLARITY_CEC_ROOT=/path` in `.env` (never commit it).
Not wired into the main pipeline yet; it would allow a CEC-comparable evaluation (BLOCKERS B2).

## 2. Android phone: system latency (M7) and energy (M8)
Needs: Android 10+ phone with USB debugging, a **wired/USB-C headset**, and either a loopback dongle
(headset jack → mic) or a measurement mic held at the earcup. Optional: Bluetooth earbuds (LE Audio if possible).
```
adb devices                       # exactly one phone (or set ANDROID_SERIAL)
make android-assets OP=b16_s0     # bundle the trained operating-point model + ECAPA enrollment model
make android-install
make android-latency              # repeat 5x per route: wired, BT classic, BT LE Audio
adb logcat -d | grep ClarityLatency   # copy the round-trip ms values into results/m7/device/<route>.json:
                                      #   {"route": "wired", "roundtrip_ms": [..5 values..], "device": "<model>"}
bash android/scripts/energy_session.sh 30 b16    # 30-min session, then again for b10 / b5 models
python -m bench.system_latency && python -m bench.energy_estimate && make figures
```
The app shows "label: on-device" for these numbers. Turn the volume down before the first start.

## 3. Real recordings (M9)
Record only people who have **all** agreed (everyone audible). Files stay on this machine (git-ignored).
* `data/personal/raw/scenes/<scene>.wav` — binaural if possible (in-ear mics L/R), else mono, ≥16 kHz.
* `data/personal/raw/scenes/<scene>.json` — `{"target": "<name>", "transcript": "<what the target said>",
  "target_azimuth_deg": 0, "tags": ["similar_voice" | "low_snr" | "reverberant" | "off_axis"]}`
* `data/personal/raw/enroll/<name>.wav` — 10–20 s of the target speaking alone.
Then `make eval-real` → `results/m9/summary_per_scene.json` (metrics only) and local A/B clips in
`data/personal/out/` for the listening test.

## 4. Listening study (M10)
1. Ethics: submit `study/PROTOCOL.md`, `study/CONSENT.md`, `study/INSTRUCTIONS.md` to your IRB/REB (or your
   organisation's equivalent). Do not test anyone before approval.
2. `uv run python -m study.make_stimuli --jobs 8` (needs the trained models) → `data/cache/study_stimuli/`
   and `study/stimuli_manifest.json`; then `uv run python -m study.randomize --participants 20`.
3. Calibrate: closed-back headphones on a coupler/artificial ear; play the calibration noise in the tool
   and set the level to 65 dB(A). Never change the level during a session.
4. `make study-serve` → open http://127.0.0.1:8765, enter the participant code (P00, P01, …).
5. After the sessions: `make study-analyze` → `results/m10/analysis.json` (real) — labelled separately from
   the simulated dry run.

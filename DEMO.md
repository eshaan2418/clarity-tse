# DEMO.md — 2-minute live demo (desktop; no phone required)

> Research prototype, **not a medical device**. Keep headphone volume low before pressing play.

**Setup (once):** `make setup data` and the trained operating-point model (`b16_s0`, 16 ms algorithmic
latency). Run `make demo`; it writes `results/demo/demo.json` and the clips in `results/demo/audio/`
(LibriSpeech speech, synthetic noise only: redistributable).

| Time | Do | Say |
|---|---|---|
| 0:00 | Open `results/figs/latency_budget.png` | "The target: isolate the person you're talking to within 20 ms of algorithmic + compute delay. Bluetooth is not in that budget; it is shown separately and is an estimate." |
| 0:15 | Play `synthetic_demo_faced_unprocessed.wav` | "A simulated restaurant at 0 dB SNR: the partner in front, four other talkers, babble and clatter, heard at the left ear." |
| 0:30 | Play `synthetic_demo_faced_clarity.wav` | "Same input, processed hop by hop by the exported ONNX model the C++ and Android engines run: 15 s voice enrollment, direction from the attention policy, then volume and the safety limiter." |
| 0:50 | Show `demo.json` → `scenes.faced` | "ESTOI rises from 0.382 to 0.486; with the true direction it would be 0.504, so selection costs little here." |
| 1:05 | Play `synthetic_demo_off_axis_clarity.wav` | "Failure case: the partner is 21 degrees off-axis and other talkers are nearer the nose. 'Focus on who I'm facing' locks onto the wrong person: ESTOI 0.255 against 0.299 unprocessed, although the extractor alone, given the true direction, reaches 0.430." |
| 1:30 | `make rt-soak` output / `results/m6/soak_60min.json` | "The C++ engine runs on the real CoreAudio clock: per-hop compute and xruns over 60 minutes (numbers in the README, labelled desktop)." |
| 1:45 | Open the app screenshot / `android/README.md` | "The Android app builds and its tests pass on an emulator, but there are no on-device latency or energy numbers yet: that needs a phone (HUMAN_TODO.md)." |

All numbers above are read from `results/demo/demo.json` (`scripts/check_numbers.py` verifies them).

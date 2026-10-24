# HUMAN_TODO

Things only a human can do. The pipeline picks each one up automatically once it's done.

1. **(Optional) Clarity Challenge data:** register at claritychallenge.org, download CEC2 and set `CLARITY_CEC_ROOT=/path` in `.env`. (Not yet wired into the main pipeline; it would let us run a CEC-comparable evaluation.)
2. **Android device:** plug in an Android 10+ phone with USB debugging on, then run `adb devices`, `make android-install`, and `make android-latency` (this needs a wired headset and a loopback dongle, or a measurement mic held at the earcup). See `android/README.md`.
3. **Real recordings (M9):** put consented 16 kHz+ WAV scenes in `data/personal/raw/scenes/` and enrollment clips in `data/personal/raw/enroll/<name>.wav`, then run `make eval-real`. Everyone audible must have given consent. These files are git-ignored.
4. **Listening study (M10):** get ethics/IRB approval (or your organization's equivalent), recruit 12–20 participants, calibrate headphones following `study/PROTOCOL.md`, and run `make study-serve` locally. Results are saved to `study/data/` and analysed automatically by `make study-analyze`.

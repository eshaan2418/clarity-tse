# Clarity Android app (research prototype — **not a medical device**)

Kotlin UI + NDK engine: Oboe full-duplex stream (AAudio low-latency/exclusive when available) →
hop FIFO → `clarity::Engine` (ONNX Runtime, the same C++ as `rt/`) → safety limiter → output.
The callback does not allocate or lock. There is no INTERNET permission, so audio cannot leave the phone.

Features: voice enrollment (15 s, on-device ECAPA ONNX → FiLM), audiogram entry (stored locally,
not validated), "focus on who I'm facing" (straight-ahead direction prior) vs "focus on this voice"
(enrolled embedding, no direction), A/B bypass (still limited), volume with hard cap, loopback
latency measurement, route/focus/process-death handling (streams re-open on route change; focus loss
stops processing; nothing resumes automatically at volume).

Build (JDK 17):
```
make android-assets OP=b16_s0   # copies the exported model + ecapa.onnx into app/src/main/assets (git-ignored)
make android                    # assembleDebug + unit tests
ANDROID_SERIAL=<device> make android-test-device   # instrumented tests
```
Simplifications vs the Python system (DECISIONS D14): the on-device "facing" mode uses a fixed
straight-ahead direction, not the full DOA/IMU tracker of `clarity_tse/select/policy.py`.
Phones expose at most a stereo capture path, and earbud microphones are not available to apps as a
synchronised binaural pair, so on a real phone the 2-channel model sees the phone's own mics, not
the simulated ear-level array it was trained on (BLOCKERS B3). Use the 1-channel model for honest
earbud tests.

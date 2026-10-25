# BLOCKERS

| # | Blocker | Fallback | Impact on claims |
|---|---|---|---|
| B1 | No CUDA GPU | MPS, small models, short training | Absolute quality below SOTA; relative comparisons and latency curve remain valid at reduced scale |
| B2 | Clarity Challenge / WHAM! / DNS not available | LibriSpeech + ESC-50 + synthetic noise in simulated rooms | Results are on simulated scenes, not CEC; no leaderboard comparability |
| B3 | No real earbud/phone mic array | Simulated binaural pair (free-field, no HRTF) | Beamforming/DOA gains are optimistic vs real head-worn arrays |
| B4 | No IMU stream | Simulated head-yaw trajectories | Target-selection results assume IMU noise model, not real sensor data |
| B5 | No Android device | Desktop RT engine; app built + unit tests; instrumented tests on an arm64 **emulator** (functional only) | No `on-device` latency/energy numbers; system latency and M8 are `estimated`; emulator timings are not evidence |
| B6 | No personal recordings | M9 blocked-with-fallback | No real-world evaluation |
| B7 | No listeners | Study package + simulated dry-run | No human intelligibility results; HL results are `simulated listener` only |
| B8 | Shared host: other users' workloads (QShield-X sims, other agents) drive load average ~100 on 18 cores | Every timing result records `os.getloadavg()`; latency benchmarks re-run when load is low; p99/p99.9 reported with the load at measurement | Compute-latency tails may be pessimistic; RTF is an upper bound under contention |
| B9 | No root/sudo on the desktop (no `powermetrics`) | Energy estimated from measured duty cycle × assumed power ranges | M8 numbers are `estimated` brackets, not measurements |
| B10 | Reduced training budget (6k steps, ~2 M-param models, MPS) | Same budget for all compared models | Absolute SI-SDRi/ESTOI far below published SOTA; offline ceiling is weak at this scale, so the "gap to ceiling" understates what a fully-trained non-causal model achieves |

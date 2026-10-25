# Clarity listening study — protocol (v1, prepared; **awaiting-human**: needs ethics approval before any participant is tested)

## Aim
Measure speech-in-noise intelligibility, perceived quality and listening effort for target-speaker
extraction (Clarity) at two latency operating points, compared with the unprocessed mixture, a
single-channel noise suppressor (RNNoise) and a classical beamformer (MPDR steered to the true talker).

## Design
* Within-subject, 5 conditions: `unprocessed`, `rnnoise`, `mpdr`, `clarity_lo` (lowest-latency point that
  meets the 20 ms budget), `clarity_hi` (40 ms point).
* 12–20 adult participants. Recruit normal-hearing listeners, plus people with self-reported hearing
  loss where possible (record self-report and, if available, a recent audiogram; this is **not** a
  screening or diagnosis).
* Stimuli: simulated scenes from `results/m1/testsets/test_main.json` at SNRs of −5, 0 and +5 dB
  (3 SNR × 4 scene types). Every participant hears each condition × SNR cell equally often. Sentences
  are never repeated within a participant (Latin-square assignment, `study/randomize.py`).
* Tasks:
  1. **Word recognition**: listen once, type what the target talker said. Score = % keywords correct
     (all content words in the LibriSpeech transcript, after normalisation), scored automatically.
  2. **MUSHRA-style quality** (ITU-R BS.1534-inspired): rate 5 conditions + hidden reference (clean
     target) + anchor (mixture low-passed at 3.5 kHz) on a 0–100 scale. Trials where the hidden
     reference is rated below 90 in more than 15 % of trials are flagged (post-screening as in BS.1534).
  3. **Listening effort**: 7-point scale (1 = no effort … 7 = extreme effort) after each recognition trial.
* About 45 minutes, with a break every 15 minutes.

## Playback, calibration and safety
* Closed-back headphones over a known interface. Before the session the experimenter calibrates with a
  sound-level meter + coupler (or an artificial ear) so that the speech-shaped calibration noise
  (`/calibration.wav` from the server) measures **65 dB(A)**. The server applies the same hard limiter
  as the app (−12 dBFS default ceiling). With calibration done, no stimulus exceeds 80 dB(A).
* Participants may lower the volume at any time and can stop at any point without giving a reason.
* Nothing is played before calibration is confirmed in the UI.

## Data and privacy
* The tool runs on `localhost`. Responses are written only to `study/data/<pid>.jsonl` on the test
  machine (git-ignored). Nothing is uploaded. Participant IDs are random codes; the code↔name key, if
  kept at all, is stored on paper and separately from the data.
* No audio of the participant is recorded.

## Analysis (pre-registered; `study/analyze.py`)
* Word recognition: binomial GLMM, `correct ~ condition * snr + (1|participant) + (1|item)`,
  fitted with statsmodels `BinomialBayesMixedGLM` (variational Bayes). We also fit a logit-transformed
  proportion LMM as a robustness check. Planned contrasts: clarity_lo vs each baseline, Holm-corrected.
* MUSHRA and effort: linear mixed model `rating ~ condition + (1|participant)`.
* Hearing-loss subgroup (if n ≥ 5): the same models with a `group` factor; exploratory.
* Power: with 16 participants × 36 trials, the design has about 80 % power (simulation in
  `study/simulate.py`, `--power`) to detect a 10-percentage-point improvement in word recognition,
  assuming between-participant SD 8 pp.

## Ethics
See `HUMAN_TODO.md` §4 and `study/CONSENT.md`. Do not run before your institution approves (IRB/REB
or an equivalent ethics review).

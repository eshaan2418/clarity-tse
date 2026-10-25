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
* Stimuli (`study/make_stimuli.py`): fresh simulated scenes with **test-split speakers only** (never
  seen in training), in which the target talker says exactly **one complete LibriSpeech sentence**
  (2.5–6 s, onset 0.5 s), at SNRs of −5, 0 and +5 dB in all 4 scene types. Each sentence is processed by
  every condition with the same code as the objective evaluation (Clarity gets the true direction and a
  10 s enrollment of the talker; MPDR gets the true direction). Transcripts are stored in
  `study/stimuli_manifest.json`. Every participant hears each condition × SNR cell twice; a sentence is
  never repeated within a participant, and across each block of 5 participants every sentence is heard in
  every condition (Latin square, `study/randomize.py`).
* Tasks:
  1. **Word recognition** (30 trials): listen once, type what the talker in front said. Score = %
     keywords correct: content words of the transcript (function words excluded), order-free match
     after normalisation (`study/score.py`), scored automatically.
  2. **MUSHRA-style quality** (ITU-R BS.1534-inspired; 4 screens at 0 dB SNR, one per scene type): rate
     5 conditions + hidden reference (clean direct+early target) + anchor (mixture low-passed at 3.5 kHz)
     on a 0–100 scale, unlimited replays, every version must be played. Participants who rate the hidden
     reference below 90 on more than 15 % of screens are excluded from the MUSHRA analysis (post-screening
     as in BS.1534).
  3. **Listening effort**: 7-point scale (1 = no effort … 7 = extreme effort) after each recognition trial.
* About 45 minutes, with a break every 15 minutes.

## Playback, calibration and safety
* Closed-back headphones over a known interface. Before the session the experimenter calibrates with a
  sound-level meter + coupler (or an artificial ear) so that the speech-shaped calibration noise
  (`/calibration.wav`, long-term spectrum of the study sentences, at the presented target-speech level)
  measures **65 dB(A)**. All stimuli share one presentation offset (−8 dB) chosen so the unprocessed
  mixtures stay below the limiter ceiling; the server applies the same hard limiter as the app (−12 dBFS
  ceiling) on every request. `make_stimuli.py` prints the maximum stimulus peak; with the 65 dB(A)
  calibration, the −12 dBFS ceiling corresponds to about 87 dB SPL peak (65 dB + (−12 − (−34)) dB), and RMS
  levels stay at conversational levels. Lower the calibration target if your ethics board asks for it.
* Participants may lower the volume at any time and can stop at any point without giving a reason.
* Nothing is played before calibration is confirmed in the UI.

## Data and privacy
* The tool runs on `localhost`. Responses are written only to `study/data/<pid>.jsonl` on the test
  machine (git-ignored). Nothing is uploaded. Participant IDs are random codes; the code↔name key, if
  kept at all, is stored on paper and separately from the data.
* No audio of the participant is recorded.

## Analysis (pre-registered; `study/analyze.py`)
* Word recognition: binomial GLMM, `correct ~ condition + snr + (1|participant) + (1|item)`,
  fitted with statsmodels `BinomialBayesMixedGLM` (variational Bayes). We also fit a logit-transformed
  proportion LMM as a robustness check. Planned contrasts: clarity_lo vs each baseline, Holm-corrected.
* MUSHRA and effort: linear mixed models `rating ~ condition + (1|participant)`,
  `effort ~ condition + snr + (1|participant)`; same planned contrasts.
* Hearing-loss subgroup (if n ≥ 5): the same models with a `group` factor; exploratory.
* Power: `python -m study.simulate --power` simulates the full design (30 word trials per participant)
  for 12/16/20 participants with an assumed +10 percentage-point effect (0.42 logit) and
  between-participant SD of about 8 pp; results in `results/m10/power_simulated.json` (planning only).

## Ethics
See `HUMAN_TODO.md` §4 and `study/CONSENT.md`. Do not run before your institution approves (IRB/REB
or an equivalent ethics review).

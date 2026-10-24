# PROGRESS

Legend: todo / in-progress / done / blocked-with-fallback / awaiting-human

| Milestone | Status | Evidence |
|---|---|---|
| M0 Bootstrap | done | `make test` (27 passed), `make rt-test`, ENVIRONMENT.md, ruff+mypy clean. Android Gradle skeleton deferred to M7 (SDK install), tiny train-eval CI job added with M3 |
| M1 Simulator & eval harness | in-progress | |
| M2 Baselines & offline ceiling | todo | |
| M3 Streaming extractor | todo | |
| M4 Target selection | todo | |
| M5 Personalization | todo | |
| M6 RT engine | todo | |
| M7 Android | todo | |
| M8 Energy | todo | |
| M9 Real recordings | todo | |
| M10 Study package | todo | |
| M11 Write-up | todo | |

## Single next action
M1: write `bench/make_testsets.py` (4 scene types x 5 SNRs x 12 scenes, 6 s, seeded, sha256 manifest), plus a latency harness test.

## M1 plan
- [x] corpus index (LibriSpeech/ESC-50), synthetic noises
- [x] scene simulator: rooms, 2–6 talkers, babble, clatter, music, SNR -5..15, head rotation, binaural rendering
- [x] metrics: SI-SDR(i), PESQ, STOI/ESTOI, HASPI/HASQI (audiogram), DNSMOS + tests
- [ ] frozen test sets with checksums (restaurant/party/street/kitchen) + val sets
- [ ] latency harness test (algorithmic from config, per-hop compute, RTF)
Acceptance: deterministic regeneration matches checksums; metric tests green.

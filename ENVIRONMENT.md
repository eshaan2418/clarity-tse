# ENVIRONMENT

Probed 2026-10-24 (M0).

| Item | Value |
|---|---|
| Host | MacBook Pro, Apple **M5 Pro** (arm64, 18 cores), 48 GB unified memory, macOS 27.0.1 |
| Disk | 768 GB free |
| GPU | No CUDA. Apple GPU via PyTorch **MPS** backend (torch 2.14.1) |
| Python | 3.12.14 (mise), deps pinned via `uv.lock` |
| C/C++ | Apple clang (Xcode CLT), CMake (Homebrew) |
| Java | Homebrew openjdk 27 (too new for Android Gradle Plugin; see M7) |
| Android SDK/NDK | **Not installed** at M0; `adb` absent; **no Android device attached** |
| Audio hardware | MacBook Pro built-in mic + speakers; no wired/BT test headset declared; no measurement mic |
| Docker | Docker Desktop CLI present (daemon status not relied upon) |
| Network | Available (OpenSLR, PyPI, GitHub reachable) |

## Dataset access status

| Dataset | Status | License | Use |
|---|---|---|---|
| LibriSpeech train-clean-100 / dev-clean / test-clean | Downloaded from OpenSLR | CC BY 4.0 | target/interferer speech, enrollment, babble |
| ESC-50 | Downloaded from GitHub | CC BY-NC 3.0 | environmental noise / clatter (research use; not redistributed) |
| WHAM! / DNS noise | Not downloaded (size); fallback to ESC-50 + synthetic | — | — |
| Clarity Challenge (CEC) | **Gated — not available** (no credentials in `.env`) | Clarity licence | — |
| WSJ0 | Not available (LDC) | — | — |
| Personal recordings | `data/personal/raw/` empty | — | M9 blocked-with-fallback |

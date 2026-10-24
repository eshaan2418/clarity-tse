"""Create a tiny licence-free fixture corpus in LibriSpeech + ESC-50 layout.

"Speakers" are synthetic vowel-like sources (glottal pulse train with speaker-specific f0
and formants, syllabic amplitude modulation). Used by CI so `make test` / `make train-quick`
need no downloaded or gated data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

SR = 16000


def synth_speaker_utt(rng: np.random.Generator, f0: float, formants: list[float], seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0_track = f0 * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t + rng.uniform(0, 6)))
    phase = np.cumsum(2 * np.pi * f0_track / SR)
    src = (np.sin(phase) > 0.95).astype(float) + 0.02 * rng.standard_normal(n)
    y = src
    for k, fc in enumerate(formants):
        fc_t = fc * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(2, 5) * t))
        # time-invariant approximation per 20 ms block
        out = np.zeros(n)
        bw = 80 + 40 * k
        r = np.exp(-np.pi * bw / SR)
        for s in range(0, n, 320):
            th = 2 * np.pi * fc_t[s] / SR
            b, a = [1 - r], [1, -2 * r * np.cos(th), r * r]
            out[s : s + 320] = lfilter(b, a, src[s : s + 320])
        y = y + out * (0.6 ** k)
    syl = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3, 5) * t + rng.uniform(0, 6))
    gate = (np.convolve(rng.random(n // 1600 + 1) > 0.2, np.ones(1), "same").repeat(1600)[:n]).astype(float)
    y = y * syl * gate
    return (0.05 * y / (np.sqrt(np.mean(y**2)) + 1e-9)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fixtures/external")
    ap.add_argument("--speakers", type=int, default=20)
    ap.add_argument("--utts", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.out)
    rng = np.random.default_rng(1234)
    for split, offset in [("train-clean-100", 0), ("dev-clean", 1000), ("test-clean", 2000)]:
        for s in range(args.speakers):
            spk = str(offset + s)
            f0 = rng.uniform(90, 240)
            formants = sorted(rng.uniform([500, 1200, 2300], [900, 2000, 3200]).tolist())
            d = root / "LibriSpeech" / split / spk / "1"
            d.mkdir(parents=True, exist_ok=True)
            for u in range(args.utts):
                sf.write(d / f"{spk}-1-{u:04d}.flac", synth_speaker_utt(rng, f0, formants, rng.uniform(3, 6)), SR)
    esc = root / "ESC-50-master"
    (esc / "audio").mkdir(parents=True, exist_ok=True)
    (esc / "meta").mkdir(parents=True, exist_ok=True)
    rows = []
    cats = ["glass_breaking", "pouring_water", "car_horn", "laughing", "engine", "footsteps"]
    for fold in range(1, 6):
        for c in cats:
            fn = f"{fold}-{c}.wav"
            n = 5 * SR
            x = rng.standard_normal(n) * np.exp(-np.arange(n) % 8000 / rng.uniform(500, 4000))
            sf.write(esc / "audio" / fn, (0.05 * x).astype(np.float32), SR)
            rows.append(dict(filename=fn, fold=fold, target=0, category=c, esc10=False, src_file=0, take="A"))
    with open(esc / "meta" / "esc50.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"fixtures written to {root}")


if __name__ == "__main__":
    main()

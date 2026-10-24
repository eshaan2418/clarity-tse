"""Corpus indexing for LibriSpeech (speech) and ESC-50 (environmental noise).

Splits are speaker-disjoint by construction: train = train-clean-100, val = dev-clean,
test = test-clean. ESC-50 folds 1-3 = train, 4 = val, 5 = test.
"""

from __future__ import annotations

import csv
import functools
import json
import os
import zlib
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[2]
EXTERNAL = Path(os.environ.get("CLARITY_EXTERNAL", REPO / "data" / "external"))
CACHE = Path(os.environ.get("CLARITY_CACHE", REPO / "data" / "cache"))
SR = 16000

LIBRI_SPLITS = {"train": "train-clean-100", "val": "dev-clean", "test": "test-clean"}
ESC_FOLDS = {"train": (1, 2, 3), "val": (4,), "test": (5,)}

NOISE_CATEGORIES = {
    "clatter": ["glass_breaking", "can_opening", "footsteps", "door_wood_knock", "mouse_click",
                "keyboard_typing", "door_wood_creaks", "clock_tick", "drinking_sipping", "clapping"],
    "kitchen": ["pouring_water", "water_drops", "can_opening", "glass_breaking", "washing_machine",
                "vacuum_cleaner", "brushing_teeth", "clock_alarm", "crackling_fire"],
    "street": ["car_horn", "engine", "train", "siren", "church_bells", "airplane", "helicopter",
               "footsteps", "wind", "rain"],
    "party": ["laughing", "clapping", "glass_breaking", "coughing", "fireworks", "drinking_sipping"],
}


@dataclass(frozen=True)
class Utterance:
    speaker: str
    path: str
    n_samples: int


def _resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g)


def read_audio(path: str | Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return _resample(x, sr).astype(np.float32)


@functools.lru_cache(maxsize=4)
def librispeech_index(split: str) -> dict[str, list[Utterance]]:
    """speaker -> utterances (cached to json)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"libri_{split}_{zlib.crc32(str(EXTERNAL).encode()):08x}.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
        return {k: [Utterance(**u) for u in v] for k, v in raw.items()}
    root = EXTERNAL / "LibriSpeech" / LIBRI_SPLITS[split]
    if not root.exists():
        raise FileNotFoundError(f"{root} missing; run `make data`")
    out: dict[str, list[Utterance]] = {}
    for p in sorted(root.rglob("*.flac")):
        info = sf.info(str(p))
        spk = p.parent.parent.name
        out.setdefault(spk, []).append(Utterance(spk, str(p.relative_to(EXTERNAL)), int(info.frames)))
    cache.write_text(json.dumps({k: [u.__dict__ for u in v] for k, v in out.items()}))
    return out


def load_utt(u: Utterance) -> np.ndarray:
    return read_audio(EXTERNAL / u.path)


@functools.lru_cache(maxsize=4)
def esc50_index(split: str) -> dict[str, list[str]]:
    meta = EXTERNAL / "ESC-50-master" / "meta" / "esc50.csv"
    if not meta.exists():
        return {}
    out: dict[str, list[str]] = {}
    with open(meta) as f:
        for row in csv.DictReader(f):
            if int(row["fold"]) in ESC_FOLDS[split]:
                out.setdefault(row["category"], []).append(
                    str((EXTERNAL / "ESC-50-master" / "audio" / row["filename"]).relative_to(EXTERNAL)))
    return out


def synthetic_noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Licence-free noise generators (used for listening samples and when ESC-50 is absent)."""
    if kind == "music":
        t = np.arange(n) / SR
        y = np.zeros(n)
        beat = rng.uniform(0.35, 0.6)
        n_notes = int(n / SR / beat) + 1
        for i in range(n_notes):
            start = int(i * beat * SR)
            dur = int(beat * SR * rng.uniform(0.8, 1.6))
            f0 = 110 * 2 ** (rng.integers(0, 24) / 12)
            seg = slice(start, min(n, start + dur))
            tt = t[seg] - t[start]
            env = np.exp(-tt * rng.uniform(2, 6)) * (1 - np.exp(-tt * 200))
            for h in range(1, 7):
                y[seg] += env * np.sin(2 * np.pi * f0 * h * tt + rng.uniform(0, 6.28)) / h
            if i % 2 == 0:  # percussive click
                m = min(n - start, int(0.03 * SR))
                y[start : start + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / 80.0) * 0.5
        return y.astype(np.float32)
    if kind == "pink":
        w = rng.standard_normal(n)
        f = np.fft.rfftfreq(n, 1 / SR)
        W = np.fft.rfft(w) / np.sqrt(np.maximum(f, 20.0))
        return np.fft.irfft(W, n=n).astype(np.float32)
    if kind == "clatter":
        y = np.zeros(n)
        for _ in range(int(rng.poisson(3 * n / SR))):
            s = int(rng.integers(0, n))
            m = min(n - s, int(rng.uniform(0.01, 0.08) * SR))
            fc = rng.uniform(1500, 6000)
            tt = np.arange(m) / SR
            y[s : s + m] += np.sin(2 * np.pi * fc * tt) * np.exp(-tt * rng.uniform(30, 120)) * rng.uniform(0.3, 1)
        return y.astype(np.float32)
    raise ValueError(kind)

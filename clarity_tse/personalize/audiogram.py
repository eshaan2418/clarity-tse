"""Audiogram profiles and validation.

Profiles are *synthetic* listener types spanning mild/moderate/severe and flat/sloping shapes,
loosely modelled on the standard-audiogram families of Bisgaard, Vlaming & Dahlquist (2010).
They are not copies of that paper's tables. All hearing-impaired results in this repo are for
these simulated listeners (label: ``simulated listener``).
"""

from __future__ import annotations

import numpy as np

FREQS = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000], dtype=float)

PROFILES: dict[str, list[float]] = {
    "NH": [0, 0, 0, 0, 0, 0, 0, 0],
    "mild_flat": [25, 25, 25, 30, 30, 30, 35, 35],
    "moderate_flat": [45, 45, 50, 50, 50, 55, 55, 60],
    "mild_sloping": [10, 10, 15, 25, 35, 40, 50, 55],
    "moderate_sloping": [20, 25, 30, 45, 55, 60, 65, 70],
    "severe_sloping": [30, 35, 45, 60, 70, 75, 80, 85],
}


def validate(levels: list[float] | np.ndarray) -> np.ndarray:
    """User-entered audiogram -> clipped float array (dB HL, -10..120). Raises on wrong length/NaN."""
    a = np.asarray(levels, dtype=float)
    if a.shape != (len(FREQS),) or not np.all(np.isfinite(a)):
        raise ValueError(f"audiogram needs {len(FREQS)} finite values at {FREQS.tolist()} Hz")
    return np.clip(a, -10, 120)


def random_audiogram(rng: np.random.Generator) -> np.ndarray:
    """Training distribution: smooth, mostly non-decreasing losses from normal to severe."""
    base = rng.uniform(0, 50)
    slope = rng.uniform(0, 45)  # extra loss at 8 kHz vs 250 Hz
    shape = np.clip((np.log2(FREQS / 250) / 5) ** rng.uniform(0.5, 2.0), 0, 1)
    a = base + slope * shape + rng.normal(0, 4, len(FREQS))
    return np.clip(np.maximum.accumulate(a) if rng.random() < 0.8 else a, 0, 95)

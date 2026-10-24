"""Hearing-safety output stage: gain cap + zero-lookahead peak limiter + hard ceiling.

Every playback path (Python demo, C++ engine, Android) must end in an equivalent stage. The hard
ceiling is applied *after* the smooth limiter, so no parameter combination can produce a sample
above ``ceiling`` (the property that ``tests/test_safety.py`` checks).

Levels are in dBFS. Mapping to dB SPL depends on the playback chain and must be calibrated
(see study/PROTOCOL.md); defaults are deliberately conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ABSOLUTE_MAX_CEILING_DBFS = -1.0  # no configuration may exceed this
DEFAULT_CEILING_DBFS = -12.0
MAX_USER_GAIN_DB = 20.0


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


@dataclass
class SafetyLimiter:
    ceiling_dbfs: float = DEFAULT_CEILING_DBFS
    user_gain_db: float = 0.0
    attack_ms: float = 0.5
    release_ms: float = 80.0
    sr: int = 16000
    _env: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self) -> None:
        # clamp configuration: the limiter cannot be configured into an unsafe state
        self.ceiling_dbfs = float(min(self.ceiling_dbfs, ABSOLUTE_MAX_CEILING_DBFS))
        if not np.isfinite(self.ceiling_dbfs):
            self.ceiling_dbfs = DEFAULT_CEILING_DBFS
        self.user_gain_db = float(np.clip(np.nan_to_num(self.user_gain_db, nan=0.0), -60.0, MAX_USER_GAIN_DB))
        self._env = 1.0

    @property
    def ceiling(self) -> float:
        # re-clamped on every use so post-construction mutation cannot bypass the cap
        c = self.ceiling_dbfs if np.isfinite(self.ceiling_dbfs) else DEFAULT_CEILING_DBFS
        return db_to_lin(min(c, ABSOLUTE_MAX_CEILING_DBFS))

    @property
    def gain(self) -> float:
        g = float(np.nan_to_num(self.user_gain_db, nan=0.0, posinf=MAX_USER_GAIN_DB, neginf=-60.0))
        return db_to_lin(float(np.clip(g, -60.0, MAX_USER_GAIN_DB)))

    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (T,) or (C, T). Returns float32 with |y| <= ceiling guaranteed."""
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        mono = x.ndim == 1
        y = np.atleast_2d(x) * self.gain
        c = self.ceiling
        a_att = np.exp(-1.0 / max(1.0, self.attack_ms * 1e-3 * self.sr))
        a_rel = np.exp(-1.0 / max(1.0, self.release_ms * 1e-3 * self.sr))
        peaks = np.abs(y).max(axis=0)
        gains = np.empty(y.shape[1])
        env = self._env
        for i, peak in enumerate(peaks):
            target = 1.0 if peak <= c else c / peak
            coef = a_att if target < env else a_rel
            env = coef * env + (1 - coef) * target
            gains[i] = min(env, target)  # instantaneous fallback guarantees no overshoot
        self._env = env
        out = np.clip(y * gains, -c, c).astype(np.float32)
        return out[0] if mono else out

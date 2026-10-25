"""Attention-steered target selection policy (tau).

Every block (default 100 ms) the policy decides *who* to extract and outputs the target's
azimuth relative to the head (for the extractor's directional feature) and the speaker
embedding to condition on.

State: a set of tracked talkers in the *world* frame (IMU yaw + head-relative DOA). Each track
has a world azimuth, an activity level, and a running voice-similarity score against each
enrolled embedding (measured on a delay-and-sum beam steered at the track).

Attention model (HMM over tracks with sticky transitions):
    P(attend j | evidence) ∝ facing_prior(rel_az_j) ^ w_face  x  voice_term_j ^ w_voice  x  activity_j
  * facing prior: von-Mises-like bump around 0 deg relative azimuth (people face whom they listen
    to) — head yaw from the IMU makes this update instantly on a head turn.
  * voice term (mode "voice"): similarity of the track's beam to the selected enrolled voice;
    (mode "facing"): max similarity over enrolled voices (enrolled people preferred over strangers),
    flat when nothing is enrolled.
Switching uses hysteresis: a new track must beat the current one by ``switch_margin`` for
``switch_hold_s`` seconds. A silent target is kept (track memory ``hold_silence_s``) so a pause
does not hand the extractor to a distractor.

Fallbacks: no enrollment -> direction-only (embedding = zeros); no mic array (single channel) ->
voice-only (theta = NaN); no IMU -> tracks kept in the head frame (yaw fixed at 0), so head turns
are only noticed through DOA changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from clarity_tse.dsp import array
from clarity_tse.dsp.stft import STFT, LatencyConfig


def wrap(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


@dataclass
class PolicyConfig:
    block_s: float = 0.1
    mode: str = "facing"  # "facing" | "voice"
    use_imu: bool = True
    use_doa: bool = True
    use_voice: bool = True
    face_sigma_deg: float = 25.0
    w_face: float = 1.0
    w_voice: float = 1.0
    switch_margin: float = 1.5  # posterior ratio new/current required
    switch_hold_s: float = 0.3
    hold_silence_s: float = 3.0
    track_merge_deg: float = 12.0
    doa_peaks: int = 3
    doa_window_s: float = 0.4
    voice_window_s: float = 1.5
    voice_every_s: float = 0.5
    voice_temp: float = 0.08  # softness of the voice similarity -> likelihood map
    vad_db: float = -50.0


@dataclass
class Track:
    world_az: float
    last_seen: float
    activity: float = 1.0
    sims: dict[int, float] = field(default_factory=dict)  # enrolled index -> EMA similarity


class TargetSelector:
    def __init__(self, cfg: PolicyConfig, enrolled: list[np.ndarray] | None = None, sr: int = 16000,
                 embed_fn=None, voice_index: int = 0):
        self.cfg = cfg
        self.sr = sr
        self.enrolled = [e / (np.linalg.norm(e) + 1e-8) for e in (enrolled or [])]
        self.embed_fn = embed_fn
        self.voice_index = voice_index
        self.tracks: list[Track] = []
        self.current: int | None = None
        self.challenger: tuple[int, float] | None = None
        self.t = 0.0
        self._last_voice_t = -1e9
        self.stft = STFT(LatencyConfig(512, 128))
        self.freqs = np.fft.rfftfreq(512, 1 / sr)
        self.grid = np.arange(-90.0, 90.1, 2.0)

    # --------------------------------------------------------------- DOA
    def _doa_peaks(self, x: np.ndarray) -> list[tuple[float, float]]:
        """Return up to N (lateral angle, strength) peaks of SRP-PHAT over the window."""
        X = self.stft.analysis(torch.tensor(x, dtype=torch.float64)).numpy()
        p = array.srp_phat(X, self.freqs, self.grid).mean(axis=1)
        peaks = []
        q = p.copy()
        for _ in range(self.cfg.doa_peaks):
            i = int(np.argmax(q))
            if q[i] <= 0.05 * max(p.max(), 1e-9):
                break
            peaks.append((float(self.grid[i]), float(q[i])))
            lo, hi = max(0, i - 8), min(len(q), i + 9)
            q[lo:hi] = -np.inf
        return peaks

    def _resolve_front(self, lateral: float) -> float:
        # A 2-mic interaural array only observes sin(theta); assume the frontal solution.
        return lateral

    # ------------------------------------------------------------- voice
    def _beam(self, x: np.ndarray, rel_az: float) -> np.ndarray:
        X = self.stft.analysis(torch.tensor(x, dtype=torch.float64)).numpy()
        Y = array.delay_and_sum(X, rel_az, self.freqs)
        return self.stft.synthesis(torch.tensor(Y), x.shape[1]).numpy()

    def _update_voice(self, x: np.ndarray, yaw: float) -> None:
        if not (self.cfg.use_voice and self.enrolled and self.embed_fn):
            return
        for tr in self.tracks:
            if self.t - tr.last_seen > 0.5:
                continue
            rel = float(wrap(tr.world_az - yaw))
            sig = self._beam(x, rel) if self.cfg.use_doa else x[0]
            if 10 * np.log10(np.mean(sig**2) + 1e-12) < self.cfg.vad_db:
                continue
            e = self.embed_fn(sig.astype(np.float32))
            for k, en in enumerate(self.enrolled):
                s = float(np.dot(e, en) / (np.linalg.norm(e) + 1e-8))
                tr.sims[k] = s if k not in tr.sims else 0.6 * tr.sims[k] + 0.4 * s

    # --------------------------------------------------------------- step
    def step(self, x_doa: np.ndarray, x_voice: np.ndarray | None, yaw_deg: float) -> tuple[float, np.ndarray, int | None]:
        """Advance one block.

        x_doa: (2, n) recent audio (doa_window_s); x_voice: (2, m) longer window or None;
        yaw_deg: IMU yaw at the end of the block (ignored when use_imu=False).
        Returns (theta_rel_deg or NaN, embedding (192,) or zeros, current track index).
        """
        cfg = self.cfg
        self.t += cfg.block_s
        yaw = yaw_deg if cfg.use_imu else 0.0
        energy_db = 10 * np.log10(np.mean(x_doa**2) + 1e-12)
        if cfg.use_doa and x_doa.shape[0] == 2 and energy_db > cfg.vad_db:
            for lat, strength in self._doa_peaks(x_doa):
                w_az = float(wrap(self._resolve_front(lat) + yaw))
                d = [abs(float(wrap(tr.world_az - w_az))) for tr in self.tracks]
                if d and min(d) < cfg.track_merge_deg:
                    tr = self.tracks[int(np.argmin(d))]
                    tr.world_az = float(wrap(tr.world_az + 0.3 * float(wrap(w_az - tr.world_az))))
                    tr.last_seen = self.t
                    tr.activity = 0.7 * tr.activity + 0.3 * 1.0
                else:
                    self.tracks.append(Track(world_az=w_az, last_seen=self.t))
        for tr in self.tracks:
            if tr.last_seen < self.t - 1e-9:
                tr.activity *= 0.9
        # forget stale tracks (but never the current one within hold_silence_s)
        keep = []
        for i, tr in enumerate(self.tracks):
            stale = self.t - tr.last_seen > cfg.hold_silence_s
            if not stale or (i == self.current and self.t - tr.last_seen <= 2 * cfg.hold_silence_s):
                keep.append(i)
        if self.current is not None:
            self.current = keep.index(self.current) if self.current in keep else None
        self.tracks = [self.tracks[i] for i in keep]
        if x_voice is not None and self.t - self._last_voice_t >= cfg.voice_every_s:
            self._update_voice(x_voice, yaw)
            self._last_voice_t = self.t
        return self._decide(yaw)

    def _score(self, tr: Track, yaw: float) -> float:
        cfg = self.cfg
        rel = float(wrap(tr.world_az - yaw))
        face = np.exp(-0.5 * (rel / cfg.face_sigma_deg) ** 2) if cfg.mode == "facing" else 1.0
        if self.enrolled and cfg.use_voice and tr.sims:
            if cfg.mode == "voice":
                s = tr.sims.get(self.voice_index, 0.0)
            else:
                s = max(tr.sims.values())
            voice = float(np.exp((s - 0.5) / cfg.voice_temp))
        else:
            voice = 1.0
        act = 0.2 + tr.activity
        return float(face**cfg.w_face * voice**cfg.w_voice * act)

    def _decide(self, yaw: float) -> tuple[float, np.ndarray, int | None]:
        cfg = self.cfg
        emb = np.zeros(192, dtype=np.float32)
        if cfg.mode == "voice" and self.enrolled:
            emb = self.enrolled[self.voice_index].astype(np.float32)
        if not self.tracks:
            if cfg.mode == "facing" and not cfg.use_doa:
                # voice-only fallback without direction
                return float("nan"), emb, None
            return (0.0 if cfg.mode == "facing" else float("nan")), emb, None
        scores = np.array([self._score(tr, yaw) for tr in self.tracks])
        best = int(np.argmax(scores))
        if self.current is None:
            self.current = best
        elif best != self.current:
            ratio = scores[best] / max(scores[self.current], 1e-12)
            if ratio > cfg.switch_margin:
                if self.challenger and self.challenger[0] == best:
                    if self.t - self.challenger[1] >= cfg.switch_hold_s:
                        self.current, self.challenger = best, None
                else:
                    self.challenger = (best, self.t)
            else:
                self.challenger = None
        else:
            self.challenger = None
        tr = self.tracks[self.current]
        theta = float(wrap(tr.world_az - yaw)) if cfg.use_doa else float("nan")
        if cfg.mode == "facing" and self.enrolled and cfg.use_voice and tr.sims:
            k = max(tr.sims, key=lambda kk: tr.sims[kk])
            if tr.sims[k] > 0.35:
                emb = self.enrolled[k].astype(np.float32)
        return theta, emb, self.current

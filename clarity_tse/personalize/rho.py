"""Listener personalisation (rho): audiogram-conditioned band gain + compression in the
extractor's own STFT domain, so it adds **no** algorithmic latency (contrast: NAL-R as a
linear-phase FIR adds nfir/2 samples).

Training signal: a differentiable intelligibility *proxy* — envelope correlation (STOI-like,
Taal et al. 2011) between (a) the processed signal passed through a simple audibility model for
the listener's audiogram and (b) the clean reference through the same model with normal hearing.
Evaluation never uses this proxy: it uses pyclarity HASPI v2 (Kates & Arehart), which has its own
auditory model, so we do not train on the test metric.

Safety: per-band gain is bounded to [0, MAX_BAND_GAIN_DB]; compression only reduces gain; the
global SafetyLimiter (dsp/limiter.py) still follows in every playback path.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from clarity_tse.personalize.audiogram import FREQS
from clarity_tse.synth.scene import SPL_AT_0DBFS

MAX_BAND_GAIN_DB = 35.0
N_BANDS = 24
NORMAL_THRESHOLD_DB = 15.0  # approx. band-level threshold (dB SPL) for normal hearing in this model


def erb_band_weights(n_fft: int, sr: int = 16000, n_bands: int = N_BANDS, fmin: float = 100.0, fmax: float = 7500.0):
    """(n_bands, F) triangular ERB-spaced weights, and band centre frequencies."""
    def erb(f):
        return 21.4 * np.log10(1 + 0.00437 * f)

    def inv(e):
        return (10 ** (e / 21.4) - 1) / 0.00437

    edges = inv(np.linspace(erb(fmin), erb(fmax), n_bands + 2))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    W = np.zeros((n_bands, len(f)))
    for b in range(n_bands):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        W[b] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    return W, edges[1:-1]


def audiogram_at(levels: torch.Tensor, centres: np.ndarray) -> torch.Tensor:
    """Interpolate (B, 8) audiograms to band centres on a log-frequency axis -> (B, n_bands)."""
    x = np.log2(FREQS)
    xc = np.clip(np.log2(centres), x[0], x[-1])
    idx = np.clip(np.searchsorted(x, xc) - 1, 0, len(x) - 2)
    frac = torch.tensor((xc - x[idx]) / (x[idx + 1] - x[idx]), dtype=levels.dtype, device=levels.device)
    lo, hi = levels[:, idx], levels[:, idx + 1]
    return lo + (hi - lo) * frac


class BandLevels(nn.Module):
    W: torch.Tensor

    """Causal, smoothed band levels in dB SPL from STFT power."""

    def __init__(self, n_fft: int, hop: int, sr: int = 16000, tau_ms: float = 10.0):
        super().__init__()
        W, centres = erb_band_weights(n_fft, sr)
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32), persistent=False)
        self.centres = centres
        self.alpha = math.exp(-hop / (tau_ms * 1e-3 * sr)) if tau_ms > 0 else 0.0
        # calibration: white noise with RMS 1 (sqrt-Hann-like windows) -> total ~ SPL_AT_0DBFS
        self.cal = 10 * math.log10(n_fft * n_fft / 4.0)

    def forward(self, pr: torch.Tensor) -> torch.Tensor:
        """pr: power (B, F, K) -> levels (B, n_bands, K) dB SPL."""
        bp = torch.einsum("bf,nfk->nbk", self.W, pr)
        if self.alpha > 0:
            # causal first-order smoothing, as a truncated exponential FIR (vectorised, export-friendly)
            L = max(2, int(math.ceil(math.log(1e-3) / math.log(self.alpha))))
            ker = (1 - self.alpha) * self.alpha ** torch.arange(L - 1, -1, -1, dtype=bp.dtype, device=bp.device)
            n, b, k = bp.shape
            x = nn.functional.pad(bp.reshape(n * b, 1, k), (L - 1, 0), mode="replicate")
            bp = nn.functional.conv1d(x, ker.view(1, 1, L)).view(n, b, k)
        return 10 * torch.log10(bp + 1e-12) - self.cal + SPL_AT_0DBFS


class RhoStage(nn.Module):
    Wn: torch.Tensor

    """Audiogram -> per-band (gain, compression ratio, knee); applied as a real gain per STFT bin."""

    def __init__(self, n_fft: int, hop: int, sr: int = 16000, hidden: int = 64, per_listener: bool = False):
        super().__init__()
        self.levels = BandLevels(n_fft, hop, sr)
        self.net = nn.Sequential(nn.Linear(8, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 3 * N_BANDS))
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        Wn = self.levels.W / (self.levels.W.sum(0, keepdim=True) + 1e-8)
        self.register_buffer("Wn", Wn, persistent=False)  # band -> bin interpolation
        self.offset = nn.Parameter(torch.zeros(3 * N_BANDS)) if per_listener else None

    def params(self, audiogram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hl = audiogram_at(audiogram, self.levels.centres)
        raw = self.net(audiogram / 100.0)
        if self.offset is not None:
            raw = raw + self.offset
        rg, rc, rk = raw.chunk(3, dim=-1)
        # initial guess = half-gain rule (Lybarger) on the band HL, refined by the network
        gain = torch.clamp(0.5 * hl + 10.0 * rg, 0.0, MAX_BAND_GAIN_DB)
        ratio = 1.0 + 2.0 * torch.sigmoid(rc - 1.0)  # 1..3
        knee = 50.0 + 15.0 * torch.tanh(rk)  # dB SPL band level
        return gain, ratio, knee

    def gain_db(self, yr: torch.Tensor, yi: torch.Tensor, audiogram: torch.Tensor) -> torch.Tensor:
        """Per-bin gain (B, F, K) in dB for spectrum Y=(yr, yi) and audiograms (B, 8)."""
        gain, ratio, knee = self.params(audiogram)
        L = self.levels(yr.square() + yi.square())  # (B, nb, K) input band level
        comp = (1 - 1 / ratio)[..., None] * torch.relu(L + gain[..., None] - knee[..., None])
        g_band = torch.clamp(gain[..., None] - comp, 0.0, MAX_BAND_GAIN_DB)
        return torch.einsum("bf,nbk->nfk", self.Wn, g_band)

    def forward(self, yr, yi, audiogram):
        g = 10 ** (self.gain_db(yr, yi, audiogram) / 20)
        return yr * g, yi * g


def nalr_gains_db(audiogram: np.ndarray) -> np.ndarray:
    """NAL-R insertion gain at the 6 NAL frequencies (same formula as pyclarity.enhancer.nalr)."""
    nal_f = np.array([250, 500, 1000, 2000, 4000, 6000])
    hl = np.interp(np.log2(nal_f), np.log2(FREQS), audiogram)
    bias = np.array([-17, -8, 1, -1, -2, -2])
    crit = hl[1] + hl[2] + hl[3]
    x = 0.05 * crit if crit <= 180 else 9.0 + 0.116 * (crit - 180)
    return np.clip(x + 0.31 * hl + bias, 0, None)


# ------------------------------------------------------------------ differentiable proxy
class AudibilityModel(nn.Module):
    win: torch.Tensor

    """Very simple hearing model: band level -> sensation level (soft-thresholded) w.r.t. the
    listener's thresholds. Normal hearing uses NORMAL_THRESHOLD_DB everywhere."""

    def __init__(self, n_fft: int = 512, hop: int = 128, sr: int = 16000):
        super().__init__()
        self.levels = BandLevels(n_fft, hop, sr, tau_ms=0.0)
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)

    def forward(self, x: torch.Tensor, audiogram: torch.Tensor) -> torch.Tensor:
        X = torch.stft(x, self.n_fft, self.hop, window=self.win, return_complex=True)
        L = self.levels(X.real.square() + X.imag.square())
        thr = NORMAL_THRESHOLD_DB + audiogram_at(audiogram, self.levels.centres)[..., None]
        # sensation level, softly floored at threshold; recruitment-like compression of SL
        sl = nn.functional.softplus((L - thr) / 4.0) * 4.0
        return torch.log1p(sl)


def intelligibility_proxy(env_proc: torch.Tensor, env_ref: torch.Tensor, seg: int = 30) -> torch.Tensor:
    """STOI-like mean short-time envelope correlation (B,)."""
    K = min(env_proc.shape[-1], env_ref.shape[-1])
    p, r = env_proc[..., :K], env_ref[..., :K]
    n = K // seg
    p = p[..., : n * seg].unfold(-1, seg, seg)
    r = r[..., : n * seg].unfold(-1, seg, seg)
    p = p - p.mean(-1, keepdim=True)
    r = r - r.mean(-1, keepdim=True)
    c = (p * r).sum(-1) / (p.norm(dim=-1) * r.norm(dim=-1) + 1e-6)
    # weight segments where the reference is audible/active
    w = (r.abs().mean(-1) > 1e-3).float()
    return (c * w).sum((-1, -2)) / w.sum((-1, -2)).clamp_min(1)

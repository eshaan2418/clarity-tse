"""Low-latency STFT with asymmetric analysis/synthesis windows.

Algorithmic latency of a WOLA system is set by the *synthesis* window support. Following the
asymmetric-window construction used for very-low-latency enhancement (Mauler & Martin, 2007;
Wang et al., "STFT-domain neural speech enhancement with very low algorithmic latency", TASLP 2023),
the analysis window has length ``n_fft`` (mostly past context) while the synthesis window is
non-zero only on the last ``2*hop`` samples. Product of the two windows is a periodic Hann of
length ``2*hop`` on that support, so with hop ``hop`` the system is perfectly reconstructing.

Frame ``k`` covers input samples ``[(k+1)*hop - n_fft, (k+1)*hop)``. After frame ``k`` is
synthesised, output samples ``[(k-1)*hop, k*hop)`` are final, so the worst-case algorithmic
delay is ``2*hop`` samples (+ ``lookahead*hop`` if the model peeks at future frames).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _hann_periodic(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def asymmetric_windows(n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (analysis, synthesis) windows of length ``n_fft``."""
    if n_fft < 2 * hop:
        raise ValueError("n_fft must be >= 2*hop")
    n, h = n_fft, hop
    wa = np.zeros(n)
    rise = np.sqrt(_hann_periodic(2 * (n - h)))[: n - h]
    wa[: n - h] = rise
    short = _hann_periodic(2 * h)
    wa[n - h :] = np.sqrt(short[h:])
    ws = np.zeros(n)
    seg = slice(n - 2 * h, n - h)
    num = short[:h]
    den = wa[seg]
    with np.errstate(divide="ignore", invalid="ignore"):
        ws[seg] = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
    ws[n - h :] = np.sqrt(short[h:])
    return wa, ws


@dataclass(frozen=True)
class LatencyConfig:
    """STFT/latency configuration. All values in samples at ``sr``."""

    n_fft: int
    hop: int
    lookahead: int = 0  # frames
    sr: int = 16000

    @property
    def algorithmic_latency_samples(self) -> int:
        return 2 * self.hop + self.lookahead * self.hop

    @property
    def algorithmic_latency_ms(self) -> float:
        return 1000.0 * self.algorithmic_latency_samples / self.sr

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1

    @classmethod
    def for_budget_ms(cls, budget_ms: float, sr: int = 16000, min_nfft: int = 512) -> LatencyConfig:
        """Zero-lookahead config whose algorithmic latency equals ``budget_ms``."""
        hop = int(round(budget_ms * sr / 1000 / 2))
        n_fft = max(min_nfft, 2 * hop)
        return cls(n_fft=n_fft, hop=hop, lookahead=0, sr=sr)


class STFT(torch.nn.Module):
    """Batch (full-sequence) analysis/synthesis matching :class:`StreamingSTFT` exactly."""

    wa: torch.Tensor
    ws: torch.Tensor

    def __init__(self, cfg: LatencyConfig):
        super().__init__()
        self.cfg = cfg
        wa, ws = asymmetric_windows(cfg.n_fft, cfg.hop)
        self.register_buffer("wa", torch.tensor(wa, dtype=torch.float32), persistent=False)
        self.register_buffer("ws", torch.tensor(ws, dtype=torch.float32), persistent=False)

    def num_frames(self, n_samples: int) -> int:
        # one extra frame so the final hop of samples is fully overlap-added
        return -(-n_samples // self.cfg.hop) + 1

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., T) -> complex (..., F, K)."""
        n, h = self.cfg.n_fft, self.cfg.hop
        t = x.shape[-1]
        k = self.num_frames(t)
        pad_right = k * h - t
        xp = torch.nn.functional.pad(x, (n - h, pad_right))
        frames = xp.unfold(-1, n, h)  # (..., K, N)
        spec = torch.fft.rfft(frames * self.wa, dim=-1)
        return spec.transpose(-1, -2)

    def synthesis(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """complex (..., F, K) -> (..., length)."""
        n, h = self.cfg.n_fft, self.cfg.hop
        frames = torch.fft.irfft(spec.transpose(-1, -2), n=n, dim=-1) * self.ws  # (..., K, N)
        k = frames.shape[-2]
        lead = frames.shape[:-2]
        total = (k - 1) * h + n
        # overlap-add via fold
        flat = frames.reshape(-1, k, n).transpose(1, 2)  # (B, N, K)
        out = torch.nn.functional.fold(flat, output_size=(1, total), kernel_size=(1, n), stride=(1, h))
        out = out.reshape(*lead, total)
        return out[..., n - h : n - h + length]


class StreamingSTFT:
    """Hop-by-hop analysis/synthesis with explicit state (numpy, used for equivalence tests)."""

    def __init__(self, cfg: LatencyConfig, channels: int = 1):
        self.cfg = cfg
        self.wa, self.ws = asymmetric_windows(cfg.n_fft, cfg.hop)
        self.inbuf = np.zeros((channels, cfg.n_fft))
        self.ola: np.ndarray = np.zeros(cfg.n_fft)

    def push(self, hop_samples: np.ndarray) -> np.ndarray:
        """(C, hop) -> complex (C, F) spectrum of the newest frame."""
        h = self.cfg.hop
        self.inbuf = np.concatenate([self.inbuf[:, h:], hop_samples], axis=1)
        return np.fft.rfft(self.inbuf * self.wa, axis=-1)

    def pull(self, spec: np.ndarray) -> np.ndarray:
        """complex (F,) -> hop output samples (delayed by one hop w.r.t. input index)."""
        n, h = self.cfg.n_fft, self.cfg.hop
        frame = np.fft.irfft(spec, n=n) * self.ws
        self.ola += frame
        out = self.ola[n - 2 * h : n - h].copy()
        self.ola = np.concatenate([self.ola[h:], np.zeros(h)])
        return out

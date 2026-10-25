"""Streaming target-speaker extraction (TSE) network.

Per STFT frame (asymmetric low-latency windows, dsp/stft.py) the network receives
  * log-power of each mic,
  * cos/sin of the inter-mic phase difference (IPD),
  * a directional feature cos(IPD - IPD_target(theta)) for the target azimuth theta supplied by
    the selection policy (zeros + flag=0 when no direction is available),
and is conditioned on a speaker embedding via FiLM (zeros + flag=0 when no voice is enrolled).
A temporal convolutional network (Conv-TasNet-style TCN blocks over frames) predicts bounded
complex per-mic filter weights; output = sum_c W_c * X_c (a learned, time-varying MISO beamformer
+ mask). |W_c| <= 1, so the model can only attenuate or pass, never amplify a TF bin by more than
the number of mics ("fail by attenuating").

Causal mode uses left-padded convolutions; ``lookahead`` frames are realised by applying the
weights computed at frame k to frame k-lookahead. Non-causal mode (offline ceiling) uses centred
convolutions. Streaming inference keeps explicit per-layer conv buffers (``StreamingTSE``) and is
tested to match batch inference.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from clarity_tse.dsp.array import EAR_SPACING, SPEED_OF_SOUND
from clarity_tse.dsp.stft import STFT, LatencyConfig

EMB_DIM = 192


@dataclass
class TSEConfig:
    n_fft: int = 512
    hop: int = 80
    lookahead: int = 0
    sr: int = 16000
    channels: int = 2  # 1 = single-mic variant
    bottleneck: int = 128
    hidden: int = 256
    blocks: int = 7  # per repeat, dilations 1..2^(blocks-1)
    repeats: int = 2
    kernel: int = 3
    causal: bool = True
    use_doa: bool = True
    use_voice: bool = True

    @property
    def latency(self) -> LatencyConfig:
        return LatencyConfig(self.n_fft, self.hop, self.lookahead if self.causal else 0, self.sr)

    def to_dict(self) -> dict:
        return asdict(self)


class FrameNorm(nn.Module):
    """LayerNorm over channels at each frame (causal)."""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, K)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TCNBlock(nn.Module):
    def __init__(self, b: int, h: int, k: int, dilation: int, causal: bool):
        super().__init__()
        self.inp = nn.Conv1d(b, h, 1)
        self.act1 = nn.PReLU(h)
        self.norm1 = FrameNorm(h)
        self.dw = nn.Conv1d(h, h, k, dilation=dilation, groups=h)
        self.act2 = nn.PReLU(h)
        self.norm2 = FrameNorm(h)
        self.out = nn.Conv1d(h, b, 1)
        self.pad = (k - 1) * dilation
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(self.act1(self.inp(x)))
        if self.causal:
            y = nn.functional.pad(y, (self.pad, 0))
        else:
            y = nn.functional.pad(y, (self.pad // 2, self.pad - self.pad // 2))
        y = self.norm2(self.act2(self.dw(y)))
        return x + self.out(y)

    def step(self, x: torch.Tensor, buf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single frame. x: (B, b, 1); buf: (B, h, pad) past depthwise inputs."""
        y = self.norm1(self.act1(self.inp(x)))
        full = torch.cat([buf, y], dim=2)
        new_buf = full[:, :, 1:]
        y = self.norm2(self.act2(self.dw(full)))
        return x + self.out(y), new_buf


class TSENet(nn.Module):
    def __init__(self, cfg: TSEConfig):
        super().__init__()
        self.cfg = cfg
        self.stft = STFT(cfg.latency)
        F = cfg.n_fft // 2 + 1
        self.F = F
        C = cfg.channels
        n_in = C * F + (2 * F + F + 1 if C == 2 else 0)
        self.inp = nn.Conv1d(n_in, cfg.bottleneck, 1)
        self.inp_norm = FrameNorm(cfg.bottleneck)
        self.blocks = nn.ModuleList(
            TCNBlock(cfg.bottleneck, cfg.hidden, cfg.kernel, 2**i, cfg.causal)
            for _ in range(cfg.repeats) for i in range(cfg.blocks))
        self.film = nn.Linear(EMB_DIM + 1, 2 * cfg.bottleneck * len(self.blocks))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.head = nn.Conv1d(cfg.bottleneck, 2 * C * F, 1)
        freqs = torch.fft.rfftfreq(cfg.n_fft, 1 / cfg.sr)
        self.register_buffer("omega", 2 * math.pi * freqs, persistent=False)

    # ---- conditioning ----
    def condition(self, emb: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        """emb: (B, 192), (B, K, 192) per-frame, or None -> FiLM (B, n_blocks, 2, b[, K])."""
        if emb is None or not self.cfg.use_voice:
            cond = torch.zeros(batch, EMB_DIM + 1, device=device)
        else:
            present = (emb.abs().sum(dim=-1, keepdim=True) > 0).float()
            cond = torch.cat([emb, present], dim=-1)
        if cond.dim() == 3:  # per-frame conditioning (target switches mid-utterance)
            K = cond.shape[1]
            film = self.film(cond).view(batch, K, len(self.blocks), 2, self.cfg.bottleneck)
            return film.permute(0, 2, 3, 4, 1)
        return self.film(cond).view(batch, len(self.blocks), 2, self.cfg.bottleneck)

    def features(self, re: torch.Tensor, im: torch.Tensor, theta: torch.Tensor | None) -> torch.Tensor:
        """re/im: (B, C, F, K) real tensors; theta: (B, K) degrees, NaN/None = unknown -> (B, n_in, K).

        Real arithmetic only (ONNX-exportable). IPD enters as cos/sin computed from the normalised
        cross-spectrum, and the directional feature as cos(IPD + omega*tau) expanded trigonometrically.
        """
        C = self.cfg.channels
        re, im = re[:, :C], im[:, :C]
        logp = torch.log10(re.square() + im.square() + 1e-10).flatten(1, 2) * 0.5 + 2.5
        if C == 1:
            return logp
        cr = re[:, 1] * re[:, 0] + im[:, 1] * im[:, 0]  # X_R * conj(X_L)
        ci = im[:, 1] * re[:, 0] - re[:, 1] * im[:, 0]
        mag = torch.sqrt(cr.square() + ci.square() + 1e-12)
        cos_ipd, sin_ipd = cr / mag, ci / mag
        if theta is None or not self.cfg.use_doa:
            df = torch.zeros_like(cos_ipd)
            flag = torch.zeros_like(cos_ipd[:, :1])
        else:
            avail = torch.isfinite(theta).to(re.dtype)  # (B, K)
            th = torch.where(avail > 0, theta, torch.zeros_like(theta))
            tau = EAR_SPACING * torch.sin(th * (math.pi / 180.0)) / SPEED_OF_SOUND  # (B, K)
            phi = self.omega[None, :, None] * tau[:, None, :]
            df = (cos_ipd * torch.cos(phi) - sin_ipd * torch.sin(phi)) * avail[:, None, :]
            flag = avail[:, None, :]
        return torch.cat([logp, cos_ipd, sin_ipd, df, flag], dim=1)

    def weights_from_hidden(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, b, K) -> real/imag weights (B, C, F, K) with |W| <= 1."""
        C, F = self.cfg.channels, self.F
        raw = self.head(h).view(h.shape[0], C, F, 2, -1)
        wr, wi = raw[:, :, :, 0], raw[:, :, :, 1]
        mag = torch.sqrt(wr.square() + wi.square() + 1e-12)
        scale = torch.tanh(mag) / mag
        return wr * scale, wi * scale

    @staticmethod
    def apply_weights(wr, wi, xr, xi) -> tuple[torch.Tensor, torch.Tensor]:
        """sum_c W_c X_c in real arithmetic. All (B, C, F, K) -> (B, F, K)."""
        return (wr * xr - wi * xi).sum(dim=1), (wr * xi + wi * xr).sum(dim=1)

    def forward(self, mix: torch.Tensor, emb: torch.Tensor | None = None,
                theta: torch.Tensor | None = None, post=None) -> torch.Tensor:
        """mix: (B, C_in, T); emb: (B, 192); theta: (B, K) frame-rate target azimuth (deg), NaN = unknown;
        post: optional callable (yr, yi) -> (yr, yi) applied to the output spectrum before synthesis.

        Returns time-aligned estimate (B, T) of the target at the reference (left) mic.
        """
        B, _, T = mix.shape
        X = self.stft.analysis(mix[:, : self.cfg.channels])  # (B, C, F, K)
        K = X.shape[-1]
        if theta is not None and theta.shape[-1] != K:
            theta = nn.functional.interpolate(theta[:, None], size=K, mode="nearest")[:, 0]
        h = self.inp_norm(self.inp(self.features(X.real, X.imag, theta)))
        film = self.condition(emb, B, mix.device)
        if film.dim() == 4:
            film = film[..., None]  # broadcast over frames
        for i, blk in enumerate(self.blocks):
            h = blk(h * (1 + film[:, i, 0]) + film[:, i, 1])
        wr, wi = self.weights_from_hidden(h)
        L = self.cfg.lookahead if self.cfg.causal else 0
        if L:
            # weights computed at frame k (seeing frames <= k) are applied to frame k-L
            wr = torch.cat([wr[..., L:], wr[..., -1:].expand(*wr.shape[:-1], L)], dim=-1)
            wi = torch.cat([wi[..., L:], wi[..., -1:].expand(*wi.shape[:-1], L)], dim=-1)
        yr, yi = self.apply_weights(wr, wi, X.real, X.imag)
        if post is not None:  # e.g. listener personalisation in the same STFT (no added latency)
            yr, yi = post(yr, yi)
        return self.stft.synthesis(torch.complex(yr, yi), T)

    def frames_for(self, n_samples: int) -> int:
        return self.stft.num_frames(n_samples)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def macs_per_frame(self) -> int:
        """Multiply-accumulates per hop for the network (excl. DFT and FiLM, which runs once)."""
        cfg = self.cfg
        n_in = self.inp.in_channels
        per_block = cfg.bottleneck * cfg.hidden * 2 + cfg.hidden * cfg.kernel
        dft = 2 * cfg.n_fft * self.F * (cfg.channels + 1)  # real DFT as matmul: analysis + synthesis
        return n_in * cfg.bottleneck + len(self.blocks) * per_block + cfg.bottleneck * self.head.out_channels + dft


class StreamingTSE(nn.Module):
    """Hop-in/hop-out wrapper with explicit state for ONNX export and C++/Android engines.

    forward(hop (1, C, H), theta (1,), film (1, n_blocks, 2, b), *states) -> (out (1, H), *new_states)
    The DFT is computed as matmuls (export-friendly). Output is delayed by (1 + lookahead) hops
    relative to input (the algorithmic latency is 2*H + lookahead*H, see dsp/stft.py).
    """

    def __init__(self, net: TSENet):
        super().__init__()
        assert net.cfg.causal
        self.net = net
        cfg = net.cfg
        n, F = cfg.n_fft, net.F
        k = torch.arange(n, dtype=torch.float64)
        f = torch.arange(F, dtype=torch.float64)
        ang = 2 * math.pi * k[:, None] * f[None, :] / n  # (N, F)
        wa = net.stft.wa.double()
        ws = net.stft.ws.double()
        self.register_buffer("dft_re", (wa[:, None] * torch.cos(ang)).float(), persistent=False)
        self.register_buffer("dft_im", (-wa[:, None] * torch.sin(ang)).float(), persistent=False)
        scale = torch.full((F,), 2.0, dtype=torch.float64)
        scale[0] = 1.0
        if n % 2 == 0:
            scale[-1] = 1.0
        self.register_buffer("idft_re", ((scale[:, None] * torch.cos(ang.T)) * ws[None, :] / n).float(), persistent=False)
        self.register_buffer("idft_im", ((-scale[:, None] * torch.sin(ang.T)) * ws[None, :] / n).float(), persistent=False)

    def state_shapes(self) -> list[tuple[int, ...]]:
        cfg = self.net.cfg
        shapes = [(1, cfg.channels, cfg.n_fft), (1, cfg.n_fft)]  # input buffer, OLA buffer
        shapes += [(1, cfg.hidden, blk.pad) for blk in self.net.blocks]
        if cfg.lookahead:
            shapes += [(1, cfg.channels, self.net.F, cfg.lookahead + 1, 2)]  # spectrum delay line (re/im)
        return shapes

    def init_states(self) -> list[torch.Tensor]:
        return [torch.zeros(s) for s in self.state_shapes()]

    def forward(self, hop: torch.Tensor, theta: torch.Tensor, film: torch.Tensor, *states: torch.Tensor):
        cfg = self.net.cfg
        H = cfg.hop
        inbuf, ola = states[0], states[1]
        conv_bufs = states[2 : 2 + len(self.net.blocks)]
        inbuf = torch.cat([inbuf[:, :, H:], hop], dim=2)
        re = (inbuf @ self.dft_re)[..., None]  # (1, C, F, 1)
        im = (inbuf @ self.dft_im)[..., None]
        th = theta.view(1, 1) if (cfg.use_doa and cfg.channels == 2) else None
        feats = self.net.features(re, im, th)  # theta = NaN -> direction unavailable
        h = self.net.inp_norm(self.net.inp(feats))
        new_bufs = []
        for i, (blk, buf) in enumerate(zip(self.net.blocks, conv_bufs, strict=True)):
            h, nb = blk.step(h * (1 + film[:, i, 0, :, None]) + film[:, i, 1, :, None], buf)
            new_bufs.append(nb)
        wr, wi = self.net.weights_from_hidden(h)  # (1, C, F, 1)
        extra = []
        if cfg.lookahead:
            delay = states[2 + len(self.net.blocks)]
            cur = torch.stack([re[..., 0], im[..., 0]], dim=-1)[:, :, :, None, :]  # (1, C, F, 1, 2)
            delay = torch.cat([delay[:, :, :, 1:], cur], dim=3)
            xr, xi = delay[:, :, :, 0, 0:1], delay[:, :, :, 0, 1:2]
            extra = [delay]
        else:
            xr, xi = re, im
        yr, yi = self.net.apply_weights(wr, wi, xr, xi)  # (1, F, 1)
        yr, yi = yr[..., 0], yi[..., 0]
        frame = yr @ self.idft_re + yi @ self.idft_im  # (1, N)
        ola = ola + frame
        n = cfg.n_fft
        out = ola[:, n - 2 * H : n - H]
        ola = torch.cat([ola[:, H:], torch.zeros_like(ola[:, :H])], dim=1)
        return (out, inbuf, ola, *new_bufs, *extra)

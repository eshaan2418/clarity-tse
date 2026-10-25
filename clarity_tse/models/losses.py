"""Training losses.

* Soft-thresholded SNR (Wisdom et al., "Unsupervised sound separation using mixture invariant
  training", NeurIPS 2020): -10 log10(|s|^2 / (|s - s_hat|^2 + tau |s|^2)), tau = 10^(-snr_max/10).
  It is *not* scale-invariant, so it rewards correct output level (needed to attenuate).
* Zero-reference loss for target-absent examples: 10 log10(|s_hat|^2 + tau |x|^2) (same paper),
  teaching the model to output silence when the enrolled voice is not present.
* Artifact loss: multi-resolution STFT L1 on power-law compressed magnitudes, plus an
  asymmetric penalty on *over*-estimated magnitude (energy added that is not in the target, i.e.
  leakage/musical noise). Ablated as "-artifact".
"""

from __future__ import annotations

import torch


def soft_snr_loss(est: torch.Tensor, ref: torch.Tensor, snr_max: float = 30.0) -> torch.Tensor:
    tau = 10 ** (-snr_max / 10)
    num = ref.square().sum(-1)
    den = (ref - est).square().sum(-1) + tau * num
    return -10 * torch.log10(num / (den + 1e-10) + 1e-10)


def absent_loss(est: torch.Tensor, mix_ref: torch.Tensor, snr_max: float = 30.0) -> torch.Tensor:
    tau = 10 ** (-snr_max / 10)
    return 10 * torch.log10(est.square().sum(-1) + tau * mix_ref.square().sum(-1) + 1e-10)


def mr_stft_loss(est: torch.Tensor, ref: torch.Tensor, over_weight: float = 1.0,
                 ffts: tuple[int, ...] = (256, 512, 1024), power: float = 0.3) -> torch.Tensor:
    total = est.new_zeros(est.shape[0])
    for n in ffts:
        win = torch.hann_window(n, device=est.device)
        E = torch.stft(est, n, n // 4, window=win, return_complex=True).abs().clamp_min(1e-8) ** power
        R = torch.stft(ref, n, n // 4, window=win, return_complex=True).abs().clamp_min(1e-8) ** power
        l1 = (E - R).abs().mean(dim=(-1, -2))
        over = torch.relu(E - R).square().mean(dim=(-1, -2))
        total = total + l1 + over_weight * over
    return total / len(ffts)


def tse_loss(est: torch.Tensor, ref: torch.Tensor, mix_ref: torch.Tensor, absent: torch.Tensor,
             w_spec: float = 1.0, w_over: float = 1.0, teacher: torch.Tensor | None = None,
             w_kd: float = 0.0) -> tuple[torch.Tensor, dict[str, float]]:
    present = 1.0 - absent
    l_snr = soft_snr_loss(est, ref)
    l_abs = absent_loss(est, mix_ref)
    loss = present * l_snr + absent * l_abs
    logs = {"snr": float((l_snr * present).sum() / present.sum().clamp_min(1))}
    if w_spec > 0:
        # scale spectral loss into a range comparable to the dB losses
        l_spec = mr_stft_loss(est, ref, over_weight=w_over) * 30.0
        loss = loss + w_spec * l_spec
        logs["spec"] = float(l_spec.mean())
    if teacher is not None and w_kd > 0:
        l_kd = soft_snr_loss(est, teacher)
        loss = loss + w_kd * present * l_kd
        logs["kd"] = float(l_kd.mean())
    return loss.mean(), logs

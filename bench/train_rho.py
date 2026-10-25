"""M5: train the listener-personalisation stage rho.

(a) generic audiogram-conditioned stage: trained over random audiograms;
(b) per-listener: (a) + per-band offsets fine-tuned for one audiogram profile.
Inputs are the *extractor's outputs* on training-pool mixtures (train split; no test data).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from bench.train import load_model, sha256_file
from clarity_tse.eval.latency import write_result
from clarity_tse.personalize.audiogram import PROFILES, random_audiogram
from clarity_tse.personalize.rho import AudibilityModel, BandLevels, RhoStage, intelligibility_proxy
from clarity_tse.synth import corpus
from clarity_tse.synth.train_data import MixConfig, PoolDataset

CKPT = corpus.CACHE / "ckpt"


def extractor_outputs(ckpt: str, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    net = load_model(CKPT / f"{ckpt}.pt", device)
    ds = PoolDataset("train", MixConfig(p_absent=0, p_no_voice=0, p_no_doa=0, doa_noise_deg=5.0), seed=99, crop_s=2.5)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), n, replace=False)
    ys, refs, mixes = [], [], []
    with torch.no_grad():
        for i in range(0, n, 16):
            b = [ds[int(j)] for j in idx[i : i + 16]]
            mix = torch.stack([x["mix"] for x in b]).to(device)
            # undo the random level augmentation: our calibration puts the target at -26 dBFS
            ref = torch.stack([x["ref"] for x in b]).to(device)
            g = 10 ** (-26 / 20) / ref.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            mix, ref = mix * g[:, :, None], ref * g
            K = net.frames_for(mix.shape[-1])
            th = torch.stack([x["theta"] for x in b]).to(device)[:, None].expand(-1, K)
            y = net(mix, torch.stack([x["emb"] for x in b]).to(device), th)
            ys.append(y.cpu())
            refs.append(ref.cpu())
            mixes.append(mix[:, 0].cpu())
    return torch.cat(ys), torch.cat(refs), torch.cat(mixes)


def rho_loss(rho, stft, am, y, ref, ag):
    X = stft.analysis(y)
    zr, zi = rho(X.real, X.imag, ag)
    z = stft.synthesis(torch.complex(zr, zi), y.shape[-1])
    p = intelligibility_proxy(am(z, ag), am(ref, torch.zeros_like(ag)))
    # discomfort guard: penalise band levels above 100 dB SPL
    Z = torch.stft(z, 512, 128, window=am.win, return_complex=True)
    lv = am.levels(Z.real.square() + Z.imag.square())  # not |Z|^2 via abs(): d|Z| is NaN at Z = 0
    pen = torch.relu(lv - 100).mean((-1, -2))
    return (-p + 0.05 * pen).mean(), float(p.mean())


SKIPPED = [0, 0]  # (steps skipped for non-finite loss/gradients, total steps); reported in the results file


def safe_step(loss, params, opt) -> None:
    SKIPPED[1] += 1
    if not torch.isfinite(loss):
        SKIPPED[0] += 1
        return
    loss.backward()
    if not all(torch.isfinite(q.grad).all() for q in params if q.grad is not None):
        SKIPPED[0] += 1
        opt.zero_grad()
        return
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ft-steps", type=int, default=300)
    ap.add_argument("--out", default="results/m5")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    y, ref, _ = extractor_outputs(a.ckpt, a.n, dev)
    net = load_model(CKPT / f"{a.ckpt}.pt")
    stft = net.stft
    rho = RhoStage(net.cfg.n_fft, net.cfg.hop)
    am = AudibilityModel()
    opt = torch.optim.Adam(rho.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    log = []
    for step in range(a.steps):
        idx = torch.tensor(rng.choice(len(y), 16, replace=False))
        ag = torch.tensor(np.stack([random_audiogram(rng) for _ in range(16)]), dtype=torch.float32)
        loss, p = rho_loss(rho, stft, am, y[idx], ref[idx], ag)
        opt.zero_grad()
        safe_step(loss, list(rho.parameters()), opt)
        if step % 100 == 0:
            log.append(dict(step=step, loss=float(loss), proxy=p))
            print(step, round(float(loss), 4), round(p, 4), flush=True)
    CKPT.mkdir(exist_ok=True, parents=True)
    torch.save(dict(state_dict=rho.state_dict(), n_fft=net.cfg.n_fft, hop=net.cfg.hop, per_listener=False),
               CKPT / f"rho_generic_{a.ckpt}.pt")
    hashes = {"generic": sha256_file(CKPT / f"rho_generic_{a.ckpt}.pt")}
    ft_log = {}
    for name, levels in PROFILES.items():
        r2 = RhoStage(net.cfg.n_fft, net.cfg.hop, per_listener=True)
        r2.load_state_dict(rho.state_dict(), strict=False)
        for p_ in r2.net.parameters():
            p_.requires_grad_(False)
        o2 = torch.optim.Adam([r2.offset], lr=2e-2)
        ag = torch.tensor([levels] * 16, dtype=torch.float32)
        for _ in range(a.ft_steps):
            idx = torch.tensor(rng.choice(len(y), 16, replace=False))
            loss, p = rho_loss(r2, stft, am, y[idx], ref[idx], ag)
            o2.zero_grad()
            safe_step(loss, [r2.offset], o2)
        ft_log[name] = p
        path = CKPT / f"rho_{name}_{a.ckpt}.pt"
        torch.save(dict(state_dict=r2.state_dict(), n_fft=net.cfg.n_fft, hop=net.cfg.hop, per_listener=True), path)
        hashes[name] = sha256_file(path)
        print("ft", name, round(p, 4), flush=True)
    write_result(Path(a.out) / f"train_rho_{a.ckpt}.json", dict(log=log, finetune_proxy=ft_log, checkpoints=hashes,
                 skipped_steps=SKIPPED[0], total_steps=SKIPPED[1]),
                 config=vars(a), seed=0)


def load_rho(path: Path) -> RhoStage:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    r = RhoStage(ck["n_fft"], ck["hop"], per_listener=ck["per_listener"])
    r.load_state_dict(ck["state_dict"])
    return r.eval()


_ = (BandLevels, copy, json)

if __name__ == "__main__":
    main()

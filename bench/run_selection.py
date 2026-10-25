"""M4: evaluate the attention-steered target-selection policy and its ablations.

For every scenario we run the policy block-by-block (100 ms) on the binaural mixture + simulated
IMU yaw, then feed its per-frame (theta, embedding) to the streaming extractor. Metrics:
  * selection accuracy   : fraction of active blocks where the selected track is the attended talker
                           (world azimuth within 15 deg of the attended talker's azimuth)
  * wrong-talker rate    : fraction of blocks where the selection points at another *talker*
  * switch latency       : time from the end of a head turn until the policy selects the new partner
  * end-to-end ESTOI / SI-SDR of the extractor output vs the attended talker's early image,
    compared with oracle selection (true theta + true enrollment) and the unprocessed mixture.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clarity_tse.eval.latency import write_result

VARIANTS: dict[str, dict[str, Any]] = {
    "full": dict(),
    "-imu": dict(use_imu=False),
    "-doa": dict(use_doa=False),
    "-voice": dict(use_voice=False),
    "voice_mode": dict(mode="voice"),
}
IMU_NOISE_DEG = 2.0  # white noise on simulated IMU yaw
IMU_DRIFT_DEG_PER_S = 0.5


def _imu(yaw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(len(yaw)) / 16000
    return yaw + rng.normal(0, IMU_NOISE_DEG, len(yaw)) + IMU_DRIFT_DEG_PER_S * t * rng.choice([-1, 1])


def run_policy(sc: dict[str, Any], variant: str, embed_fn, enrolled_names: list[str],
               overrides: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    from clarity_tse.select.policy import PolicyConfig, TargetSelector

    cfg = PolicyConfig(**{**(overrides or {}), **VARIANTS[variant]})
    enrolled = [embed_fn(sc["enroll"][nm]) for nm in enrolled_names] if cfg.use_voice else []
    sel = TargetSelector(cfg, enrolled, embed_fn=embed_fn, voice_index=0)
    n = sc["mix"].shape[1]
    blk = int(cfg.block_s * 16000)
    w_doa = int(cfg.doa_window_s * 16000)
    w_voice = int(cfg.voice_window_s * 16000)
    imu = _imu(sc["yaw"], np.random.default_rng(sc["seed"]))
    thetas, embs, sel_world = [], [], []
    for e in range(blk, n + 1, blk):
        x_doa = sc["mix"][:, max(0, e - w_doa) : e]
        x_voice = sc["mix"][:, max(0, e - w_voice) : e] if e >= w_voice // 2 else None
        th, emb, cur = sel.step(x_doa, x_voice, float(imu[e - 1]), yaw_doa=float(imu[max(0, e - w_doa) : e].mean()),
                                yaw_voice=float(imu[max(0, e - w_voice) : e].mean()))
        thetas.append(th)
        embs.append(emb)
        if cur is not None and cfg.use_doa:
            # report selected world azimuth in the *true* world frame for scoring
            sel_world.append(float(sc["yaw"][e - 1] + th) if np.isfinite(th) else np.nan)
        else:
            sel_world.append(np.nan)
    return dict(theta=np.array(thetas), emb=np.stack(embs), sel_world=np.array(sel_world), block=blk)


def _wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


def selection_metrics(sc: dict[str, Any], pol: dict[str, np.ndarray], variant: str) -> dict[str, float]:
    blk = pol["block"]
    ends = np.arange(blk, sc["mix"].shape[1] + 1, blk) - 1
    att = sc["attended"][ends]
    n_t = len(sc["world_az"])
    talker_az = np.stack([w[ends] for w in sc["world_az"]])  # (T, nb)
    sel = pol["sel_world"]
    voice_only = not np.isfinite(sel).any()
    if voice_only:
        # direction-free variant: judge by which talker the conditioning embedding matches best
        return dict(sel_acc=float("nan"), wrong_rate=float("nan"), switch_latency_s=float("nan"))
    d = np.abs(_wrap(talker_az - sel[None]))  # (T, nb)
    nearest = np.argmin(d, axis=0)
    valid = np.isfinite(sel) & (np.min(d, axis=0) < 15)
    correct = valid & (nearest == att)
    wrong = valid & (nearest != att) & (nearest < n_t)
    lat = []
    for t_turn, new in sc["turn_times"]:
        b0 = int(t_turn * 16000 / blk)
        idx = np.where(correct[b0:] & (att[b0:] == new))[0]
        lat.append(idx[0] * blk / 16000 if len(idx) else float("nan"))
    return dict(sel_acc=float(correct.mean()), wrong_rate=float(wrong.mean()),
                switch_latency_s=float(np.nanmean(lat)) if lat else float("nan"))


def _work(args: tuple) -> list[dict[str, Any]]:
    kind, seed, ckpt = args[:3]
    split = args[3] if len(args) > 3 else "test"
    overrides = args[4] if len(args) > 4 else None
    variants = args[5] if len(args) > 5 else list(VARIANTS)
    warnings.simplefilter("ignore")
    import torch

    from bench.train import load_model
    from clarity_tse.embed import speaker
    from clarity_tse.eval import metrics as M
    from clarity_tse.synth import corpus, scenarios

    torch.set_num_threads(1)
    sc = scenarios.make(kind, seed, split=split)
    net = load_model(corpus.CACHE / "ckpt" / f"{ckpt}.pt")
    hop = net.cfg.hop
    T = sc["mix"].shape[1]
    K = net.frames_for(T)
    ref = sc["attended_ref"]
    rows = []

    def extract(theta_f: np.ndarray | None, emb_f: np.ndarray | None) -> np.ndarray:
        with torch.no_grad():
            th = torch.tensor(theta_f, dtype=torch.float32)[None] if theta_f is not None else None
            e = torch.tensor(emb_f, dtype=torch.float32)[None] if emb_f is not None else None
            return net(torch.tensor(sc["mix"])[None], e, th)[0].numpy()

    def score(y: np.ndarray) -> dict[str, float]:
        return dict(estoi=M.stoi(y, ref, extended=True), si_sdr=M.si_sdr(y, ref))

    frame_end = np.minimum((np.arange(K) + 1) * hop - 1, T - 1)
    # oracle: true attended direction + true attended enrollment
    true_theta = np.where(sc["attended"] == 0, _wrap(sc["world_az"][0] - sc["yaw"]), _wrap(sc["world_az"][1] - sc["yaw"]))
    e_true = {nm: speaker.embed(sc["enroll"][nm]) for nm in ("A", "B")}
    emb_or = np.stack([e_true["A"] if a == 0 else e_true["B"] for a in sc["attended"][frame_end]])
    base = dict(kind=kind, seed=seed)
    if variants == list(VARIANTS):
        rows.append(dict(base, variant="unprocessed", **score(sc["mix"][0])))
        rows.append(dict(base, variant="oracle", **score(extract(true_theta[frame_end], emb_or))))
    enrolled_names = ["A", "B"] if kind == "turn" else ["A"]
    for v in variants:
        pol = run_policy(sc, v, speaker.embed, enrolled_names, overrides)
        blk = pol["block"]
        # policy decision for block b is available at its end -> applies to frames after it (causal)
        bidx = np.clip(frame_end // blk - 1, 0, len(pol["theta"]) - 1)
        before = frame_end < blk
        th_f = pol["theta"][bidx].astype(np.float32)
        th_f[before] = np.nan
        em_f = pol["emb"][bidx].astype(np.float32)
        em_f[before] = 0
        y = extract(th_f, em_f)
        rows.append(dict(base, variant=v, **score(y), **selection_metrics(sc, pol, v)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default="results/m4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--policy", default="results/m4/tuned_policy.json", help="tuned PolicyConfig overrides (from val)")
    a = ap.parse_args()
    overrides = json.loads(Path(a.policy).read_text())["best"] if Path(a.policy).exists() else {}
    from clarity_tse.synth.scenarios import KINDS, _speaker_means

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _speaker_means(a.split)  # warm embedding cache before forking
    jobs = [(k, s, a.ckpt, a.split, overrides) for k in KINDS for s in range(a.seeds)]
    with get_context("spawn").Pool(a.jobs) as p:
        rows = [r for chunk in p.imap_unordered(_work, jobs) for r in chunk]
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_scene.csv", index=False)
    summ = df.groupby(["kind", "variant"]).mean(numeric_only=True).drop(columns="seed").round(4)
    print(summ.to_string())
    write_result(out / "summary.json", dict(table=json.loads(summ.reset_index().to_json(orient="records"))),
                 config=dict(ckpt=a.ckpt, seeds=a.seeds, split=a.split, policy_overrides=overrides, imu_noise_deg=IMU_NOISE_DEG,
                             imu_drift_deg_per_s=IMU_DRIFT_DEG_PER_S, variants=VARIANTS))


if __name__ == "__main__":
    main()

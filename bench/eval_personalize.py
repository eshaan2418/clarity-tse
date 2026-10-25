"""M5 evaluation: HASPI/HASQI per audiogram profile (label: simulated listener).

Systems (all evaluated with pyclarity HASPI v2 using the profile's audiogram):
  unprocessed      : reference-mic mixture, no gain
  nalr             : NAL-R (pyclarity, 220-tap linear-phase FIR) on the mixture  ("generic amplification")
  enh              : Clarity extractor, no gain
  enh+nalr         : Clarity extractor, then NAL-R                                 ("generic enhancement + NAL-R")
  enh+rho_generic  : extractor with audiogram-conditioned rho in the same STFT (no added latency)
  enh+rho_listener : extractor with per-listener fine-tuned rho
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from clarity_tse.eval.latency import write_result

NALR_TAPS = 220


def _work(args):
    ckpt, item_id, profiles, do_hasqi = args
    warnings.simplefilter("ignore")
    import torch
    from clarity.enhancer.nalr import NALR
    from clarity.utils.audiogram import Audiogram

    from bench.train import load_model
    from bench.train_rho import load_rho
    from clarity_tse.eval import metrics as M
    from clarity_tse.eval.items import enrollment_embeddings, frame_theta, oracle_theta
    from clarity_tse.personalize.audiogram import FREQS, PROFILES
    from clarity_tse.synth import corpus, testsets

    torch.set_num_threads(1)
    it = next(x for x in testsets.manifest("test_main")["items"] if x["id"] == item_id)
    a = testsets.load("test_main", item_id)
    th = oracle_theta(it["meta"], a["yaw"])
    net = load_model(corpus.CACHE / "ckpt" / f"{ckpt}.pt")
    mix = torch.tensor(a["mix"])[None]
    K = net.frames_for(mix.shape[-1])
    tht = torch.tensor(frame_theta(th, net.cfg.hop, K))[None]
    e = torch.tensor(enrollment_embeddings("test_main")[item_id])[None]
    ref = a["target_early"]
    with torch.no_grad():
        enh = net(mix, e, tht)[0].numpy()
    nalr = NALR(NALR_TAPS, 16000)
    d = NALR_TAPS // 2
    rows = []
    for prof in profiles:
        ag = np.array(PROFILES[prof], float)
        fir, _ = nalr.build(Audiogram(levels=ag, frequencies=FREQS))

        def amp(x):
            return np.convolve(x, fir)[d : d + len(x)]  # delay-compensated for scoring; delay reported separately

        outs = {"unprocessed": a["mix"][0], "nalr": amp(a["mix"][0]), "enh": enh, "enh+nalr": amp(enh)}
        agt = torch.tensor([ag], dtype=torch.float32)
        for tag, path in (("enh+rho_generic", f"rho_generic_{ckpt}.pt"), ("enh+rho_listener", f"rho_{prof}_{ckpt}.pt")):
            rho = load_rho(corpus.CACHE / "ckpt" / path)
            with torch.no_grad():
                outs[tag] = net(mix, e, tht, post=lambda yr, yi, r=rho: r(yr, yi, agt))[0].numpy()
        for sys_name, y in outs.items():
            r = dict(id=item_id, kind=it["kind"], snr_db=it["snr_db"], profile=prof, system=sys_name,
                     haspi=M.haspi(y, ref, audiogram=ag), out_rms_dbfs=float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9)),
                     peak_dbfs=float(20 * np.log10(np.abs(y).max() + 1e-9)))
            if do_hasqi:
                r["hasqi"] = M.hasqi(y, ref, audiogram=ag)
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--hasqi", action="store_true")
    ap.add_argument("--out", default="results/m5")
    a = ap.parse_args()
    from clarity_tse.eval.items import enrollment_embeddings, stratified_ids
    from clarity_tse.personalize.audiogram import PROFILES

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    enrollment_embeddings("test_main")
    ids = stratified_ids("test_main", a.per_cell)
    jobs = [(a.ckpt, i, list(PROFILES), a.hasqi) for i in ids]
    with get_context("spawn").Pool(a.jobs) as p:
        rows = [r for ch in p.imap_unordered(_work, jobs) for r in ch]
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_item.csv", index=False)
    from clarity_tse.eval.stats import bootstrap_ci, paired_test
    table = []
    for (prof, sysn), g in df.groupby(["profile", "system"]):
        m, lo, hi = bootstrap_ci(g.haspi.values)
        table.append(dict(profile=prof, system=sysn, haspi=m, ci_lo=lo, ci_hi=hi, n=len(g)))
    tests = []
    for prof, g in df.groupby("profile"):
        piv = g.pivot(index="id", columns="system", values="haspi")
        for x, y in (("enh+rho_listener", "enh+nalr"), ("enh+rho_generic", "enh+nalr"), ("enh+nalr", "nalr"),
                     ("enh+rho_listener", "enh+rho_generic")):
            tests.append(dict(profile=prof, a=x, b=y, **paired_test(piv[x].values, piv[y].values)))
    write_result(out / "summary.json", dict(label="simulated listener", table=table, paired=tests,
                 nalr_delay_ms=1000 * (NALR_TAPS // 2) / 16000, rho_added_algorithmic_latency_ms=0.0),
                 config=vars(a))
    print(pd.DataFrame(table).pivot(index="system", columns="profile", values="haspi").round(3).to_string())
    print(json.dumps(tests[:4], indent=1))


if __name__ == "__main__":
    main()

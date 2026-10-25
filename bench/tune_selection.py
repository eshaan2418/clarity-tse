"""Tune the few free parameters of the selection policy on **validation-split** scenarios only
(dev-clean speakers; the test split is never used for tuning). Objective: mean over scenario kinds of
(selection accuracy - wrong-talker rate) for the full policy.

    uv run python -m bench.tune_selection --ckpt b16_s0 --seeds 4 --jobs 8
"""

from __future__ import annotations

import argparse
import itertools
import os
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from bench.run_selection import _work
from clarity_tse.eval.latency import write_result

GRID = dict(voice_neutral=[0.25, 0.35, 0.45], voice_max_ratio=[2.0, 4.0], face_sigma_deg=[20.0, 30.0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    from clarity_tse.synth.scenarios import KINDS, _speaker_means

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _speaker_means("val")
    combos = [dict(zip(GRID, v, strict=True)) for v in itertools.product(*GRID.values())]
    jobs = [(k, s, a.ckpt, "val", c, ["full"]) for c in combos for k in KINDS for s in range(a.seeds)]
    with get_context("spawn").Pool(a.jobs) as p:
        res = p.map(_work, jobs, chunksize=1)
    rows = []
    for (k, s, _, _, c, _), r in zip(jobs, res, strict=True):
        rows += [dict(**c, kind=k, seed=s, **{m: x[m] for m in ("sel_acc", "wrong_rate", "estoi")}) for x in r]
    df = pd.DataFrame(rows)
    keys = list(GRID)
    per = df.groupby([*keys, "kind"])[["sel_acc", "wrong_rate", "estoi"]].mean().reset_index()
    per["obj"] = per.sel_acc - per.wrong_rate
    agg = per.groupby(keys)[["obj", "sel_acc", "wrong_rate", "estoi"]].mean().reset_index().sort_values("obj", ascending=False)
    best = {k: float(agg.iloc[0][k]) for k in keys}
    Path("results/m4").mkdir(parents=True, exist_ok=True)
    write_result("results/m4/tuned_policy.json", dict(best=best, split="val", objective="mean_kind(sel_acc - wrong_rate)",
                 grid=GRID, table=agg.to_dict(orient="records")), config=vars(a))
    print(agg.round(3).to_string())
    print("best", best)


if __name__ == "__main__":
    main()

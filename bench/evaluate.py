"""Evaluate systems on a frozen test set (per-item metrics -> CSV, outputs cached for later analyses).

System spec strings:
    unprocessed | rnnoise | ds | mpdr | mvdr_oracle
    model:<ckpt_name>[:full|nodoa|novoice]      (checkpoint under data/cache/ckpt/<ckpt_name>.pt)

    uv run python -m bench.evaluate --set test_main --systems unprocessed rnnoise ds mpdr \
        --out results/m2/per_item.csv --jobs 12
"""

from __future__ import annotations

import argparse
import os
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clarity_tse.eval.latency import write_result
from clarity_tse.synth import corpus, testsets

OUT_CACHE = corpus.CACHE / "outputs"
_MODELS: dict[str, Any] = {}


def output_path(set_name: str, system: str, item_id: str) -> Path:
    return OUT_CACHE / set_name / system.replace(":", "__") / f"{item_id}.npy"


def run_system(system: str, a: dict[str, np.ndarray], emb: np.ndarray | None) -> np.ndarray:
    from clarity_tse.baselines import classical as C

    if system in C.BASELINES:
        return C.BASELINES[system](a["mix"], theta=a["theta"], interference=a["interference"])
    if system.startswith("model:"):
        import torch

        from bench.train import load_model
        from clarity_tse.eval.items import frame_theta

        parts = system.split(":")
        name, mode = parts[1], (parts[2] if len(parts) > 2 else "full")
        if name not in _MODELS:
            torch.set_num_threads(1)
            _MODELS[name] = load_model(corpus.CACHE / "ckpt" / f"{name}.pt", "cpu")
        net = _MODELS[name]
        mix = torch.tensor(a["mix"])[None]
        K = net.frames_for(mix.shape[-1])
        th = torch.tensor(frame_theta(a["theta"], net.cfg.hop, K))[None]
        e = torch.tensor(emb)[None] if emb is not None else None
        if mode == "nodoa":
            th = None
        if mode == "novoice":
            e = None
        with torch.no_grad():
            return net(mix, e, th)[0].numpy()
    raise ValueError(system)


def _work(args: tuple[str, str, list[str], bool, bool]) -> list[dict[str, Any]]:
    set_name, item_id, systems, do_haspi, recompute = args
    warnings.simplefilter("ignore")
    from clarity_tse.eval import metrics as M
    from clarity_tse.eval.items import enrollment_embeddings, oracle_theta

    man_item = next(it for it in testsets.manifest(set_name)["items"] if it["id"] == item_id)
    a = testsets.load(set_name, item_id)
    a["theta"] = oracle_theta(man_item["meta"], a["yaw"])
    emb = enrollment_embeddings(set_name)[item_id]
    ref = a["target_early"]
    rows = []
    base_sisdr = M.si_sdr(a["mix"][0], ref)
    for s in systems:
        p = output_path(set_name, s, item_id)
        if p.exists() and not recompute:
            y = np.load(p).astype(np.float32)
        else:
            y = run_system(s, a, emb).astype(np.float32)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, y.astype(np.float16) if np.abs(y).max() < 60000 else y)
        r = dict(id=item_id, kind=man_item["kind"], snr_db=man_item["snr_db"], system=s,
                 si_sdr=M.si_sdr(y, ref), estoi=M.stoi(y, ref, extended=True), stoi=M.stoi(y, ref),
                 pesq=M.pesq_wb(y, ref))
        r["si_sdri"] = r["si_sdr"] - base_sisdr
        if do_haspi:
            r["haspi"] = M.haspi(y, ref)
        if M.DNSMOS_PATH.exists():  # downloaded by `make data`; absent on CI fixtures -> NaN
            d = M.dnsmos(y)
            r.update(dnsmos_sig=d["SIG"], dnsmos_bak=d["BAK"], dnsmos_ovrl=d["OVRL"])
        else:
            r.update(dnsmos_sig=float("nan"), dnsmos_bak=float("nan"), dnsmos_ovrl=float("nan"))
        rows.append(r)
    return rows


def evaluate(set_name: str, systems: list[str], out_csv: Path, jobs: int, haspi: bool,
             ids: list[str] | None = None, recompute: bool = False) -> pd.DataFrame:
    man = testsets.manifest(set_name)
    all_ids = [it["id"] for it in man["items"]] if ids is None else ids
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from clarity_tse.eval.items import enrollment_embeddings

    enrollment_embeddings(set_name)  # compute/cache once before forking workers
    from clarity_tse.eval import metrics as _M

    if not _M.DNSMOS_PATH.exists():
        print("note: DNSMOS model not found (run `make data`); DNSMOS columns will be NaN")
    ctx = get_context("spawn")
    with ctx.Pool(jobs) as p:
        rows = [r for chunk in p.imap_unordered(_work, [(set_name, i, systems, haspi, recompute) for i in all_ids])
                for r in chunk]
    df = pd.DataFrame(rows).sort_values(["system", "id"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        old = old[~old.system.isin(systems)]
        df = pd.concat([old, df]).sort_values(["system", "id"])
    df.to_csv(out_csv, index=False)
    write_result(out_csv.with_suffix(".provenance.json"), dict(set=set_name, systems=sorted(df.system.unique()),
                 n_items=len(all_ids), csv=str(out_csv)), config=dict(haspi=haspi))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test_main")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--no-haspi", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ids = None
    if a.limit:
        ids = [it["id"] for it in testsets.manifest(a.set)["items"]][: a.limit]
    df = evaluate(a.set, a.systems, Path(a.out), a.jobs, not a.no_haspi, ids, a.recompute)
    print(df.groupby("system")[[c for c in ("si_sdri", "pesq", "estoi", "haspi", "dnsmos_ovrl") if c in df]].mean().round(3))


if __name__ == "__main__":
    main()

"""M3: quality-vs-latency curve, seed variance, ablations, and cost of every trained extractor.

Evaluates every checkpoint that has ``results/m3/runs/<name>/summary.json`` on the frozen test
set (outputs cached by bench.evaluate), plus inference-time modes of the operating point
(direction-only / voice-only). Joins per-hop compute measured by the C++ engine
(``results/m6/bench/<name>.json``, written by bench/rt_bench.sh) when present.

    uv run python -m bench.eval_streaming --jobs 10
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bench.evaluate import evaluate
from bench.summarize import ci_table, fmt
from clarity_tse.eval.latency import write_result
from clarity_tse.eval.stats import holm, paired_test

RUNS = Path("results/m3/runs")
M6 = Path("results/m6/bench")
OPERATING_POINT = "b16"  # largest budget whose algorithmic + p99 compute fits 20 ms (checked below)
METRICS = ("si_sdri", "estoi", "pesq", "haspi", "dnsmos_ovrl")


def runs() -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(RUNS.glob("*/summary.json")):
        s = json.loads(p.read_text())
        if Path(s["checkpoint"]).exists():
            out[s["name"]] = s
    return out


def budget_of(name: str) -> str | None:
    m = re.match(r"(b\d+)_s\d+$", name)
    return m.group(1) if m else ("offline" if name.startswith("offline") else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--set", default="test_main")
    ap.add_argument("--out", default="results/m3")
    ap.add_argument("--no-haspi", action="store_true")
    ap.add_argument("--recompute", action="store_true", help="re-score systems already in per_item.csv")
    ap.add_argument("--only-eval", action="store_true", help="score finished runs, skip the summary")
    a = ap.parse_args()
    rs = runs()
    systems = [f"model:{n}" for n in rs]
    op0 = f"{OPERATING_POINT}_s0"
    if op0 in rs:
        systems += [f"model:{op0}:nodoa", f"model:{op0}:novoice"]
    out = Path(a.out)
    csv = out / "per_item.csv"
    done = set(pd.read_csv(csv).system.unique()) if csv.exists() and not a.recompute else set()
    todo = [s for s in systems if s not in done]
    if todo:
        evaluate(a.set, todo, csv, a.jobs, haspi=not a.no_haspi)
    if a.only_eval:
        return
    df = pd.read_csv(csv)
    base = pd.read_csv("results/m2/per_item.csv") if Path("results/m2/per_item.csv").exists() else None
    if base is not None:
        df = pd.concat([df, base[~base.system.isin(df.system.unique())]])
    df = df[df.system.isin([*systems, "unprocessed", "rnnoise", "mpdr", "mvdr_oracle"])]

    # --- latency curve: per budget, per-seed means, then mean +- sd over seeds; item CI from seed 0
    curve = []
    for name, s in rs.items():
        b = budget_of(name)
        if b is None:
            continue
        g = df[df.system == f"model:{name}"]
        hop = yaml.safe_load((RUNS / name / "config.yaml").read_text())["model"].get("hop", 80)
        row: dict[str, Any] = dict(name=name, budget=b, algorithmic_latency_ms=s["algorithmic_latency_ms"],
                                   n_params=s["n_params"], macs_per_frame=s["macs_per_frame"], hop=hop,
                                   mmacs_per_s=s["macs_per_frame"] * 16000 / hop / 1e6)
        for m in METRICS:
            if m in g:
                row[m] = float(g[m].mean())
        bench = M6 / f"{name}.json"
        if bench.exists():
            bj = json.loads(bench.read_text())
            row.update(compute_p50_ms=bj["compute"]["p50_ms"], compute_p99_ms=bj["compute"]["p99_ms"],
                       compute_p999_ms=bj["compute"]["p999_ms"], rtf=bj["compute"]["rtf"],
                       algo_plus_p99_ms=row["algorithmic_latency_ms"] + bj["compute"]["p99_ms"])
        curve.append(row)
    cdf = pd.DataFrame(curve)
    agg = []
    for b, g in cdf.groupby("budget"):
        seeds = g[g.name.str.fullmatch(rf"{b}_s\d+") | g.name.str.startswith("offline")]
        if seeds.empty:
            continue
        r: dict[str, Any] = dict(budget=b, n_seeds=len(seeds), algorithmic_latency_ms=float(seeds.algorithmic_latency_ms.iloc[0]),
                                 n_params=int(seeds.n_params.iloc[0]), mmacs_per_s=float(seeds.mmacs_per_s.iloc[0]))
        for m in METRICS:
            if m in seeds:
                r[m] = float(seeds[m].mean())
                r[f"{m}_seed_sd"] = float(seeds[m].std(ddof=1)) if len(seeds) > 1 else float("nan")
        s0 = df[df.system == f"model:{seeds.name.iloc[0]}"]
        for m in METRICS:
            if m in s0:
                ci = ci_table(s0.assign(system="x"), ["system"], (m,))[0]
                r[f"{m}_ci_lo_seed0"], r[f"{m}_ci_hi_seed0"] = ci[f"{m}_lo"], ci[f"{m}_hi"]
        for k in ("compute_p50_ms", "compute_p99_ms", "compute_p999_ms", "rtf", "algo_plus_p99_ms"):
            if k in seeds and seeds[k].notna().any():
                r[k] = float(seeds[k].dropna().iloc[0])
        agg.append(r)
    agg.sort(key=lambda r: (r["budget"] == "offline", r["algorithmic_latency_ms"]))

    # --- ablations at the operating point (seed 0) and inference-time modes at the operating point
    def paired(a_sys: str, b_sys: str) -> list[dict[str, Any]]:
        piv = {m: df.pivot_table(index="id", columns="system", values=m) for m in METRICS if m in df}
        res = []
        for m, p in piv.items():
            if a_sys in p and b_sys in p:
                res.append(dict(metric=m, a=a_sys, b=b_sys, **paired_test(p[a_sys].to_numpy(), p[b_sys].to_numpy())))
        return res

    abl = []
    for n in rs:
        if n.startswith(f"{OPERATING_POINT}_") and not re.fullmatch(rf"{OPERATING_POINT}_s\d+", n):
            abl += paired(f"model:{n}", f"model:{op0}")
    if "b2_scratch_s0" in rs and "b2_s0" in rs:
        abl += paired("model:b2_scratch_s0", "model:b2_s0")
    for mode in ("nodoa", "novoice"):
        abl += paired(f"model:{op0}:{mode}", f"model:{op0}")
    for m in METRICS:
        idx = [i for i, r in enumerate(abl) if r["metric"] == m]
        for i, p in zip(idx, holm([abl[i]["wilcoxon_p"] for i in idx]), strict=True):
            abl[i]["p_holm"] = p
    # vs baselines at operating point
    vs = []
    for b in ("unprocessed", "rnnoise", "mpdr", "mvdr_oracle"):
        vs += paired(f"model:{op0}", b)
    overall = ci_table(df, ["system"], METRICS)
    write_result(out / "summary.json", dict(curve=agg, per_run=curve, ablations=abl, op_vs_baselines=vs,
                                            overall=overall, operating_point=op0), config=vars(a))
    md = "## Quality vs latency (mean over seeds; ± = seed SD; [CI] = item bootstrap 95% CI of seed 0)\n\n"
    md += "| budget | alg. ms | seeds | params | MMAC/s | SI-SDRi dB | ESTOI | HASPI | PESQ | p99 compute ms | alg+p99 ms |\n"
    md += "|---|---|---|---|---|---|---|---|---|---|---|\n"
    for r in agg:
        def cell(m, nd, r=r):
            if m not in r:
                return "–"
            s = f"{r[m]:.{nd}f}"
            if np.isfinite(r.get(f"{m}_seed_sd", np.nan)):
                s += f" ± {r[f'{m}_seed_sd']:.{nd}f}"
            if f"{m}_ci_lo_seed0" in r:
                s += f" [{r[f'{m}_ci_lo_seed0']:.{nd}f}, {r[f'{m}_ci_hi_seed0']:.{nd}f}]"
            return s
        md += (f"| {r['budget']} | {r['algorithmic_latency_ms']:.1f} | {r['n_seeds']} | {r['n_params'] / 1e6:.2f} M | "
               f"{r['mmacs_per_s']:.0f} | {cell('si_sdri', 2)} | {cell('estoi', 3)} | {cell('haspi', 3)} | {cell('pesq', 2)} | "
               f"{r.get('compute_p99_ms', float('nan')):.3f} | {r.get('algo_plus_p99_ms', float('nan')):.2f} |\n")
    md += "\n## Ablations (paired, a − b; Holm-adjusted Wilcoxon p)\n\n| a | b | metric | mean diff [95% CI] | p_holm |\n|---|---|---|---|---|\n"
    for r in abl:
        md += f"| {r['a']} | {r['b']} | {r['metric']} | {r['mean_diff']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r.get('p_holm', float('nan')):.2g} |\n"
    md += f"\n## Operating point {op0} vs baselines (paired, a − b)\n\n| b | metric | mean diff [95% CI] | p |\n|---|---|---|---|\n"
    for r in vs:
        md += f"| {r['b']} | {r['metric']} | {r['mean_diff']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['wilcoxon_p']:.2g} |\n"
    md += "\n## All systems (mean [95% CI])\n\n| system | n | " + " | ".join(METRICS) + " |\n|" + "---|" * (len(METRICS) + 2) + "\n"
    for r in overall:
        md += f"| {r['system']} | {r['n']} | " + " | ".join(fmt(r, m, 3) for m in METRICS) + " |\n"
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

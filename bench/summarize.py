"""Summarise a per-item metrics CSV into tables with bootstrap 95% CIs and paired tests.

    uv run python -m bench.summarize --csv results/m2/per_item.csv --ref unprocessed --out results/m2/summary

Writes <out>.json (machine-readable, with provenance) and <out>.md (tables used in docs).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clarity_tse.eval.latency import write_result
from clarity_tse.eval.stats import bootstrap_ci, holm, paired_test

METRICS = ("si_sdri", "pesq", "estoi", "stoi", "haspi", "dnsmos_ovrl")


def ci_table(df: pd.DataFrame, by: list[str], metrics: tuple[str, ...] = METRICS) -> list[dict[str, Any]]:
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        r: dict[str, Any] = dict(zip(by, key, strict=True))
        r["n"] = int(len(g))
        for m in metrics:
            if m in g:
                mean, lo, hi = bootstrap_ci(g[m].to_numpy())
                r[m], r[f"{m}_lo"], r[f"{m}_hi"] = mean, lo, hi
        rows.append(r)
    return rows


def paired_vs(df: pd.DataFrame, ref: str, metrics: tuple[str, ...] = METRICS) -> list[dict[str, Any]]:
    out = []
    for m in metrics:
        if m not in df:
            continue
        piv = df.pivot_table(index="id", columns="system", values=m)
        if ref not in piv:
            continue
        rows = [dict(metric=m, a=s, b=ref, **paired_test(piv[s].to_numpy(), piv[ref].to_numpy()))
                for s in piv.columns if s != ref]
        for r, p in zip(rows, holm([r["wilcoxon_p"] for r in rows]), strict=True):
            r["p_holm"] = p
        out += rows
    return out


def fmt(r: dict[str, Any], m: str, nd: int = 2) -> str:
    if m not in r or not np.isfinite(r[m]):
        return "–"
    return f"{r[m]:.{nd}f} [{r[m + '_lo']:.{nd}f}, {r[m + '_hi']:.{nd}f}]"


def markdown(overall: list[dict[str, Any]], metrics: tuple[str, ...], order: list[str] | None = None) -> str:
    present = [m for m in metrics if any(m in r for r in overall)]
    rows = sorted(overall, key=lambda r: order.index(r["system"]) if order and r["system"] in order else 99)
    s = "| system | n | " + " | ".join(present) + " |\n|" + "---|" * (len(present) + 2) + "\n"
    for r in rows:
        s += f"| {r['system']} | {r['n']} | " + " | ".join(fmt(r, m, 3 if m in ('estoi', 'stoi', 'haspi') else 2)
                                                        for m in present) + " |\n"
    return s


def pivot_md(piv: pd.DataFrame) -> str:
    cols = list(piv.columns)
    s = "| " + str(piv.index.name or "") + " | " + " | ".join(str(c) for c in cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    for idx, row in piv.iterrows():
        s += f"| {idx} | " + " | ".join(f"{v:.3f}" if np.isfinite(v) else "–" for v in row.to_numpy()) + " |\n"
    return s


def summarize(csv: Path, ref: str, out: Path, order: list[str] | None = None) -> dict[str, Any]:
    df = pd.read_csv(csv)
    overall = ci_table(df, ["system"])
    res = dict(csv=str(csv), overall=overall, by_kind=ci_table(df, ["system", "kind"]),
               by_snr=ci_table(df, ["system", "snr_db"]), paired_vs_ref=paired_vs(df, ref), ref=ref)
    write_result(out.with_suffix(".json"), res, config=dict(ref=ref))
    md = "## Overall (mean [95% bootstrap CI])\n\n" + markdown(overall, METRICS, order)
    for m in ("si_sdri", "estoi", "haspi"):
        if m not in df:
            continue
        piv = df.pivot_table(index="system", columns="snr_db", values=m)
        md += f"\n## {m} by input SNR (dB)\n\n" + pivot_md(piv) + "\n"
        piv = df.pivot_table(index="system", columns="kind", values=m)
        md += f"\n## {m} by scene type\n\n" + pivot_md(piv) + "\n"
    out.with_suffix(".md").write_text(md)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ref", default="unprocessed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    summarize(Path(a.csv), a.ref, Path(a.out))
    print(Path(a.out).with_suffix(".md").read_text())


if __name__ == "__main__":
    main()

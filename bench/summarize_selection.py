"""M4 summary with bootstrap 95% CIs (over scenes) and paired tests of the full policy vs each
ablation / oracle on end-to-end ESTOI (pooled over scenario kinds), from results/m4/per_scene.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clarity_tse.eval.latency import write_result
from clarity_tse.eval.stats import bootstrap_ci, holm, paired_test

METRICS = ("estoi", "si_sdr", "sel_acc", "wrong_rate", "switch_latency_s")


def main() -> None:
    df = pd.read_csv("results/m4/per_scene.csv")
    rows = []
    for keys, g in [*df.groupby(["kind", "variant"]), *((("all", v), g) for v, g in df.groupby("variant"))]:
        r = dict(kind=keys[0], variant=keys[1], n=len(g))
        for m in METRICS:
            if g[m].notna().any():
                r[m], r[f"{m}_lo"], r[f"{m}_hi"] = bootstrap_ci(g[m].to_numpy())
        rows.append(r)
    df["scene"] = df.kind + "_" + df.seed.astype(str)
    tests = []
    for m in ("estoi", "si_sdr"):
        piv = df.pivot_table(index="scene", columns="variant", values=m)
        t = [dict(metric=m, a="full", b=v, **paired_test(piv["full"].to_numpy(), piv[v].to_numpy()))
             for v in piv.columns if v != "full"]
        for x, p in zip(t, holm([x["wilcoxon_p"] for x in t]), strict=True):
            x["p_holm"] = p
        tests += t
    piv = df.pivot_table(index="scene", columns="variant", values="sel_acc")
    for v in ("-imu", "-voice"):
        tests.append(dict(metric="sel_acc", a="full", b=v, **paired_test(piv["full"].to_numpy(), piv[v].to_numpy())))
    import dataclasses
    import json as _json

    from clarity_tse.select.policy import PolicyConfig

    tuned = _json.loads(Path("results/m4/tuned_policy.json").read_text())["best"]
    policy = dataclasses.asdict(PolicyConfig(**tuned))
    write_result("results/m4/summary_ci.json", dict(table=rows, paired=tests, policy_config_used=policy),
                 config=dict(source="results/m4/per_scene.csv"))
    md = "| kind | variant | n | ESTOI [95% CI] | sel. acc. | wrong-talker | switch latency s |\n|---|---|---|---|---|---|---|\n"
    order = ["oracle", "full", "-imu", "-voice", "-doa", "voice_mode", "unprocessed"]
    for r in sorted(rows, key=lambda r: (r["kind"] != "all", r["kind"], order.index(r["variant"]))):
        def c(m, nd=3, r=r):
            return f"{r[m]:.{nd}f} [{r[m + '_lo']:.{nd}f}, {r[m + '_hi']:.{nd}f}]" if m in r else "–"
        md += f"| {r['kind']} | {r['variant']} | {r['n']} | {c('estoi')} | {c('sel_acc')} | {c('wrong_rate')} | {c('switch_latency_s', 2)} |\n"
    md += "\n| metric | a | b | mean diff [95% CI] | p (Holm) |\n|---|---|---|---|---|\n"
    for t in tests:
        md += f"| {t['metric']} | {t['a']} | {t['b']} | {t['mean_diff']:.3f} [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}] | {t.get('p_holm', t['wilcoxon_p']):.2g} |\n"
    Path("results/m4/summary_ci.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

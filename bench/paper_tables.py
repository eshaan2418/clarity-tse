"""Generate the paper's LaTeX tables directly from results/ (no hand-typed numbers).

    uv run python -m bench.paper_tables      -> paper/tables/*.tex
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("paper/tables")


def load(p: str):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def ci(r: dict, m: str, nd: int = 3) -> str:
    if m not in r or r[m] is None:
        return "--"
    if f"{m}_lo" in r:
        return f"{r[m]:.{nd}f} {{\\scriptsize[{r[m + '_lo']:.{nd}f}, {r[m + '_hi']:.{nd}f}]}}"
    return f"{r[m]:.{nd}f}"


def esc(s: str) -> str:
    return s.replace("_", "\\_").replace("+", "{+}")


def baselines() -> None:
    m2, m3 = load("results/m2/summary.json"), load("results/m3/summary.json")
    if not m2:
        return
    rows = {r["system"]: r for r in m2["overall"]}
    if m3:
        rows.update({r["system"]: r for r in m3["overall"]})
    names = [("unprocessed", "Unprocessed (left ear)"), ("rnnoise", "RNNoise (1-ch NS)"), ("ds", "Delay-and-sum (true DOA)"),
             ("mpdr", "MPDR (true DOA)"), ("mvdr_oracle", "MVDR (true DOA, oracle noise cov.)"),
             (f"model:{(m3 or {}).get('operating_point', 'b16_s0')}", "Clarity, 16\\,ms (ours)"),
             ("model:offline_s0", "Non-causal ceiling (ours)")]
    s = "\\begin{tabular}{lcccc}\\toprule\nSystem & SI-SDRi (dB) & ESTOI & HASPI & DNSMOS OVRL\\\\\\midrule\n"
    for k, lab in names:
        if k in rows:
            r = rows[k]
            s += f"{lab} & {ci(r, 'si_sdri', 2)} & {ci(r, 'estoi')} & {ci(r, 'haspi')} & {ci(r, 'dnsmos_ovrl', 2)}\\\\\n"
    s += "\\bottomrule\\end{tabular}\n"
    (OUT / "baselines.tex").write_text(s)


def curve() -> None:
    m3 = load("results/m3/summary.json")
    if not m3:
        return
    s = ("\\begin{tabular}{lccccccc}\\toprule\nBudget & Alg.\\ (ms) & Seeds & MMAC/s & p99 compute (ms) & SI-SDRi (dB) & ESTOI & HASPI\\\\\\midrule\n")
    for r in m3["curve"]:
        def sd(m, nd, r=r):
            if m not in r:
                return "--"
            v = f"{r[m]:.{nd}f}"
            if r.get(f"{m}_seed_sd") == r.get(f"{m}_seed_sd") and r.get(f"{m}_seed_sd") is not None:
                v += f"$\\pm${r[f'{m}_seed_sd']:.{nd}f}"
            return v
        comp = f"{r['compute_p99_ms']:.2f}" if "compute_p99_ms" in r else "--"
        lab = "offline" if r["budget"] == "offline" else r["budget"][1:] + "\\,ms"
        alg = "$\\infty$" if r["budget"] == "offline" else f"{r['algorithmic_latency_ms']:.0f}"
        s += f"{lab} & {alg} & {r['n_seeds']} & {r['mmacs_per_s']:.0f} & {comp} & {sd('si_sdri', 2)} & {sd('estoi', 3)} & {sd('haspi', 3)}\\\\\n"
    s += "\\bottomrule\\end{tabular}\n"
    (OUT / "curve.tex").write_text(s)
    s = "\\begin{tabular}{llcc}\\toprule\nAblation (a) & vs (b) & $\\Delta$ESTOI [95\\% CI] & $p_\\mathrm{Holm}$\\\\\\midrule\n"
    for r in m3["ablations"]:
        if r["metric"] != "estoi":
            continue
        s += (f"{esc(r['a'].replace('model:', ''))} & {esc(r['b'].replace('model:', ''))} & {r['mean_diff']:+.3f} "
              f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] & {r.get('p_holm', r['wilcoxon_p']):.2g}\\\\\n")
    s += "\\bottomrule\\end{tabular}\n"
    (OUT / "ablations.tex").write_text(s)


def selection() -> None:
    m4 = load("results/m4/summary_ci.json")
    if not m4:
        return
    s = "\\begin{tabular}{lcccc}\\toprule\nVariant & ESTOI & Sel.\\ acc. & Wrong-talker & Switch lat.\\ (s)\\\\\\midrule\n"
    for v in ["oracle", "full", "-imu", "-voice", "-doa", "voice_mode", "unprocessed"]:
        r = next((x for x in m4["table"] if x["kind"] == "all" and x["variant"] == v), None)
        if r:
            s += f"{esc(v)} & {ci(r, 'estoi')} & {ci(r, 'sel_acc')} & {ci(r, 'wrong_rate')} & {ci(r, 'switch_latency_s', 2)}\\\\\n"
    s += "\\bottomrule\\end{tabular}\n"
    (OUT / "selection.tex").write_text(s)


def personalization() -> None:
    m5 = load("results/m5/summary.json")
    if not m5:
        return
    profs = ["NH", "mild_flat", "mild_sloping", "moderate_flat", "moderate_sloping", "severe_sloping"]
    systems = ["unprocessed", "nalr", "enh", "enh+nalr", "enh+rho_generic", "enh+rho_listener"]
    tab = {(r["profile"], r["system"]): r for r in m5["table"]}
    s = "\\begin{tabular}{l" + "c" * len(profs) + "}\\toprule\nSystem & " + " & ".join(p.replace("_", " ") for p in profs) + "\\\\\\midrule\n"
    for sy in systems:
        s += esc(sy) + " & " + " & ".join(f"{tab[(p, sy)]['haspi']:.3f}" if (p, sy) in tab else "--" for p in profs) + "\\\\\n"
    s += "\\bottomrule\\end{tabular}\n"
    (OUT / "personalization.tex").write_text(s)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (baselines, curve, selection, personalization):
        f()
    print(sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()

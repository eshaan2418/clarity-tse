"""Regenerate every figure in README/paper from results/ (no figure is drawn by hand).

    uv run python -m bench.make_figures          -> paper/figs/*.pdf + results/figs/*.png

Style: one y-axis per panel, thin lines/markers, fixed entity->colour mapping (categorical slots in
fixed order), baselines as neutral reference lines with direct labels, recessive grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT_PDF = Path("paper/figs")
OUT_PNG = Path("results/figs")
TEXT, TEXT2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# entity -> colour, fixed across all figures
COLOR = {"clarity": SLOT[0], "ceiling": SLOT[6], "mvdr_oracle": SLOT[1], "mpdr": SLOT[2], "rnnoise": SLOT[3],
         "ds": SLOT[4], "unprocessed": MUTED, "nalr": SLOT[1], "enh+nalr": SLOT[2], "enh+rho_generic": SLOT[0],
         "enh+rho_listener": SLOT[6], "enh": SLOT[4]}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                     "ytick.color": TEXT2, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "legend.frameon": False, "lines.linewidth": 1.5,
                     "savefig.bbox": "tight", "savefig.dpi": 200})


def save(fig, name: str) -> None:
    OUT_PDF.mkdir(parents=True, exist_ok=True)
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF / f"{name}.pdf")
    fig.savefig(OUT_PNG / f"{name}.png")
    plt.close(fig)
    print("figure", name)


def load(p: str) -> dict | None:
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def latency_curve() -> None:
    m3, m2 = load("results/m3/summary.json"), load("results/m2/summary.json")
    if not m3:
        return
    pts = [r for r in m3["curve"] if r["budget"] != "offline"]
    ceil = next((r for r in m3["curve"] if r["budget"] == "offline"), None)
    base = {r["system"]: r for r in (m2 or {}).get("overall", [])}
    metrics = [("estoi", "ESTOI"), ("haspi", "HASPI (normal hearing)"), ("si_sdri", "SI-SDRi (dB)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    for ax, (m, lab) in zip(axes, metrics, strict=True):
        x = [r["algorithmic_latency_ms"] for r in pts if m in r]
        y = [r[m] for r in pts if m in r]
        sd = [r.get(f"{m}_seed_sd", np.nan) for r in pts if m in r]
        ax.axvspan(20, 45, color="#f1f0ec", zorder=0)
        ax.text(21, 0.25, "over 20 ms\nbudget", transform=ax.get_xaxis_transform(), color=TEXT2, fontsize=7)
        ax.errorbar(x, y, yerr=[0 if not np.isfinite(s) else s for s in sd], color=COLOR["clarity"], marker="o",
                    ms=5, capsize=2, label="Clarity (causal)")
        if ceil and m in ceil:
            ax.axhline(ceil[m], color=COLOR["ceiling"], ls="--", lw=1.2)
            ax.text(44, ceil[m], "offline ceiling", color=TEXT2, fontsize=7, ha="right", va="bottom")
        for b in ("unprocessed", "mvdr_oracle"):
            if b in base and m in base[b]:
                ax.axhline(base[b][m], color=COLOR[b], ls=":", lw=1.2)
                ax.text(44, base[b][m], {"unprocessed": "unprocessed", "mvdr_oracle": "oracle MVDR"}[b],
                        color=TEXT2, fontsize=7, ha="right", va="top")
        ax.set_xlabel("algorithmic latency (ms)")
        ax.set_ylabel(lab)
        ax.set_xlim(0, 45)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.12), fontsize=7)
    fig.suptitle("Quality vs algorithmic latency (test_main, 240 scenes; error bars = seed SD where 3 seeds)",
                 color=TEXT, fontsize=9)
    save(fig, "latency_curve")


def latency_budget() -> None:
    s = load("results/m7/system_latency.json")
    if not s:
        return
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    names = {"wired_pro_audio_bound": "wired, pro-audio\nbound", "wired_mpc_bound": "wired, perf.-class\nbound",
             "bt_le_audio": "BT LE Audio\n(claimed)", "bt_classic_a2dp": "BT classic\n(claimed)"}
    for i, r in enumerate(s["routes"]):
        left = 0.0
        segs = [("algorithmic", r["algorithmic_ms"], SLOT[0], None), ("compute p99 (desktop)", r["compute_p99_ms"], SLOT[2], None),
                ("OS audio round trip (est., upper bound)", r["os_roundtrip_ms_range"][1], "#d9d8d3", "//"),
                ("Bluetooth (est., upper)", r["bt_ms_range"][1], "#bfbdb6", "\\\\")]
        for lab, w, c, h in segs:
            if w <= 0:
                continue
            ax.barh(i, w, left=left, color=c, hatch=h, edgecolor=SURFACE, linewidth=2, height=0.6,
                    label=lab if i == len(s["routes"]) - 1 or lab.startswith("Bluetooth") else None)
            left += w
        ax.text(left + 2, i, f"≤ {left:.0f} ms", va="center", color=TEXT2, fontsize=7)
    ax.axvline(20, color=TEXT2, lw=0.8, ls="--")
    ax.text(21, len(s["routes"]) - 0.45, "20 ms budget", color=TEXT2, fontsize=7)
    ax.set_yticks(range(len(s["routes"])), [names.get(r["route"], r["route"]) for r in s["routes"]], fontsize=7)
    ax.set_xlabel("latency (ms); hatched = estimated, not measured on a device")
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h, strict=True))
    ax.legend(uniq.values(), uniq.keys(), fontsize=6, loc="lower right")
    save(fig, "latency_budget")


def selection() -> None:
    s = load("results/m4/summary.json")
    if not s:
        return
    import pandas as pd

    df = pd.DataFrame(s["table"])
    kinds = list(dict.fromkeys(df.kind))
    pal = {"oracle": SLOT[6], "full": SLOT[0], "-imu": SLOT[1], "-doa": SLOT[2], "voice_mode": SLOT[4], "unprocessed": MUTED}
    lab = {"oracle": "oracle selection", "full": "full policy", "-imu": "no IMU", "-doa": "no direction",
           "voice_mode": "voice-only mode", "unprocessed": "unprocessed"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    for ax, m, ylab, variants in ((axes[0], "estoi", "end-to-end ESTOI", ["oracle", "full", "-imu", "-doa", "unprocessed"]),
                                  (axes[1], "sel_acc", "selection accuracy", ["full", "-imu", "voice_mode"])):
        w = 0.8 / len(variants)
        for j, v in enumerate(variants):
            g = df[df.variant == v].set_index("kind").reindex(kinds)
            ax.bar(np.arange(len(kinds)) + (j - len(variants) / 2) * w + w / 2, g[m], w * 0.9, color=pal[v], label=lab[v])
        ax.set_xticks(range(len(kinds)), kinds)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=6, ncol=len(variants), loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, "selection")


def personalization() -> None:
    s = load("results/m5/summary.json")
    if not s:
        return
    import pandas as pd

    df = pd.DataFrame(s["table"])
    profs = [p for p in ["NH", "mild_flat", "mild_sloping", "moderate_flat", "moderate_sloping", "severe_sloping"]
             if p in set(df.profile)]
    systems = ["unprocessed", "nalr", "enh", "enh+nalr", "enh+rho_generic", "enh+rho_listener"]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    w = 0.8 / len(systems)
    for j, sy in enumerate(systems):
        g = df[df.system == sy].set_index("profile").reindex(profs)
        x = np.arange(len(profs)) + (j - len(systems) / 2) * w + w / 2
        ax.bar(x, g.haspi, w * 0.9, color=COLOR[sy], label=sy)
        ax.errorbar(x, g.haspi, yerr=[g.haspi - g.ci_lo, g.ci_hi - g.haspi], fmt="none", ecolor=TEXT2, lw=0.6)
    ax.set_xticks(range(len(profs)), [p.replace("_", "\n") for p in profs], fontsize=7)
    ax.set_ylabel("HASPI (simulated listener)")
    ax.legend(fontsize=6, ncol=3)
    save(fig, "personalization")


def compute_cdf() -> None:
    files = sorted(Path("results/m6/bench").glob("b*_s0.json"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(5, 2.6))
    for i, f in enumerate(files):
        d = json.loads(f.read_text())
        c = d["compute"]
        qs = [("p50", c["p50_ms"]), ("p90", c["p90_ms"]), ("p99", c["p99_ms"]), ("p99.9", c["p999_ms"])]
        ax.plot([q[1] for q in qs], [0.5, 0.9, 0.99, 0.999], marker="o", ms=4, color=SLOT[i % 8],
                label=f"{f.stem.split('_')[0]} (hop {d['hop_ms']:.0f} ms)")
    ax.set_xscale("log")
    ax.set_xlabel("per-hop compute (ms), C++ engine, 1 thread, desktop")
    ax.set_ylabel("quantile")
    ax.legend(fontsize=6)
    save(fig, "compute_quantiles")


def energy() -> None:
    e = load("results/m8/energy.json")
    if not e or not e["operating_points"]:
        return
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for r in e["operating_points"]:
        lo, hi = r["pct_battery_per_hour_range"]
        ax.plot([lo, hi], [r["estoi"]] * 2, color=SLOT[0], lw=2)
        ax.plot([(lo * hi) ** 0.5], [r["estoi"]], "o", color=SLOT[0], ms=5)
        ax.text(hi * 1.08, r["estoi"], r["budget"].replace("b", "") + " ms", va="center", fontsize=7, color=TEXT2)
    ax.set_xscale("log")
    ax.set_xlabel("estimated model %battery / hour (assumption ranges; not measured)")
    ax.set_ylabel("ESTOI")
    save(fig, "energy_quality")


def main() -> None:
    for f in (latency_curve, latency_budget, selection, personalization, compute_cdf, energy):
        f()


if __name__ == "__main__":
    main()

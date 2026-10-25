"""M8: energy-vs-quality operating points.

Evidence stage ``estimated`` unless on-device sessions exist under results/m8/device/ (written by
android/scripts/energy_session.sh -> bench/parse_energy.py), which are then reported as ``on-device``.

Estimate (no phone available, BLOCKERS B5; no power counters on the desktop without root):
  duty      = measured desktop single-thread compute per hop / hop duration   (results/m6/bench)
  duty_dev  = duty x SLOWDOWN                (phone big core slower than the M5 Pro P-core: ASSUMED range)
  P_model   = duty_dev x P_CORE              (active power of one phone big core: ASSUMED range)
  %bat/h    = P_model x 1 h / BATTERY_WH x 100
These ranges are assumptions, not measurements or citations; they bracket the answer and show how
the operating points rank. Audio I/O, display and radio power are excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

from clarity_tse.eval.latency import write_result

SLOWDOWN = (2.0, 5.0)      # ASSUMED: phone big core vs M5 Pro P-core, single thread, this workload
P_CORE_W = (0.5, 1.5)      # ASSUMED: active power of one big core while running inference
BATTERY_WH = 4500 * 3.85 / 1000  # ASSUMED: 4500 mAh, 3.85 V nominal (typical current phone)


def main() -> None:
    m3 = json.loads(Path("results/m3/summary.json").read_text())
    rows = []
    for r in m3["curve"]:
        if r["budget"] == "offline":
            continue
        b = Path(f"results/m6/bench/{r['budget']}_s0.json")
        if not b.exists():
            continue
        c = json.loads(b.read_text())["compute"]
        duty = c["mean_ms"] / json.loads(b.read_text())["hop_ms"]
        lo = duty * SLOWDOWN[0] * P_CORE_W[0]
        hi = duty * SLOWDOWN[1] * P_CORE_W[1]
        rows.append(dict(budget=r["budget"], algorithmic_latency_ms=r["algorithmic_latency_ms"],
                         mmacs_per_s=r["mmacs_per_s"], desktop_duty=duty,
                         device_duty_range=[duty * SLOWDOWN[0], duty * SLOWDOWN[1]],
                         device_realtime_feasible=duty * SLOWDOWN[1] < 1.0,
                         power_w_range=[lo, hi],
                         pct_battery_per_hour_range=[100 * lo / BATTERY_WH, 100 * hi / BATTERY_WH],
                         estoi=r.get("estoi"), haspi=r.get("haspi"), si_sdri=r.get("si_sdri"),
                         stage="estimated"))
    device = [json.loads(p.read_text()) for p in sorted(Path("results/m8/device").glob("*.json"))] \
        if Path("results/m8/device").exists() else []
    write_result("results/m8/energy.json", dict(operating_points=rows, on_device_sessions=device,
                 assumptions=dict(slowdown=SLOWDOWN, p_core_w=P_CORE_W, battery_wh=BATTERY_WH),
                 label="estimated" if not device else "on-device + estimated"),
                 stage="estimated" if not device else "on-device")
    md = ("| budget | alg. ms | MMAC/s | desktop duty | est. device duty | est. model power W | est. %battery/h | ESTOI | HASPI |\n"
          "|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        md += (f"| {r['budget']} | {r['algorithmic_latency_ms']:.0f} | {r['mmacs_per_s']:.0f} | {r['desktop_duty']:.3f} | "
               f"{r['device_duty_range'][0]:.2f}–{r['device_duty_range'][1]:.2f} | {r['power_w_range'][0]:.2f}–{r['power_w_range'][1]:.2f} | "
               f"{r['pct_battery_per_hour_range'][0]:.1f}–{r['pct_battery_per_hour_range'][1]:.1f} | {r['estoi']:.3f} | "
               f"{(r['haspi'] or float('nan')):.3f} |\n")
    Path("results/m8/energy.md").write_text("Evidence stage: **estimated** (assumption ranges in energy.json)\n\n" + md)
    print(md)


if __name__ == "__main__":
    main()

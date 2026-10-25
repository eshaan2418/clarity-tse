"""Latency budget per route, every component labelled with its evidence stage.

  algorithmic  : from the operating-point config                         (exact)
  compute      : C++ engine p99 per hop, desktop                         (desktop; results/m6/bench)
  OS audio I/O : Android requirement bounds for round-trip mic->speaker  (estimated: CDD bounds, not a device)
  Bluetooth    : published/spec ranges                                   (estimated)
On-device loopback measurements (results/m7/device/*.json, from the app's latency mode) replace the
OS/Bluetooth estimates when present.
"""

from __future__ import annotations

import json
from pathlib import Path

from clarity_tse.eval.latency import write_result

SOURCES = {
    "android_cdd_15": "Android 15 Compatibility Definition Document, sec. 5.6 (Media Performance Class: round-trip "
                      "<= 80 ms on >= 1 path) and sec. 5.10 (android.hardware.audio.pro: round-trip <= 25 ms on >= 1 path). "
                      "https://source.android.com/docs/compatibility/15/android-15-cdd",
    "bt_sig_le_audio": "Bluetooth SIG / vendor material on LE Audio (LC3): claimed ~20-30 ms minimum vs ~100-200 ms "
                       "for classic Bluetooth audio; no controlled measurement found. "
                       "https://www.bluetooth.com/blog/introducing-le-audio-the-use-cases/",
}
OS_RT_MS = {"wired_pro_audio_bound": [0.0, 25.0], "wired_mpc_bound": [0.0, 80.0]}
BT_MS = {"bt_classic_a2dp": [100.0, 200.0], "bt_le_audio": [20.0, 30.0]}


def main() -> None:
    op = json.loads(Path("results/m3/summary.json").read_text())["operating_point"]
    cfg = json.loads(Path(f"data/cache/export/{op}.json").read_text()) if Path(f"data/cache/export/{op}.json").exists() else {}
    alg = cfg.get("algorithmic_latency_ms")
    # compute on the real playback path: hop compute measured inside the CoreAudio IO callback during the
    # 60-min soak (unloaded run preferred), else the standalone benchmark loop
    src = next(p for p in ("results/m6/soak_60min.json", "results/m6/soak_60min_loaded.json", f"results/m6/bench/{op}.json")
               if Path(p).exists())
    d = json.loads(Path(src).read_text())
    comp = (d.get("hop_compute") or d["compute"])["p99_ms"]
    routes = []
    for name, (lo, hi) in OS_RT_MS.items():
        routes.append(dict(route=name, algorithmic_ms=alg, compute_p99_ms=comp, os_roundtrip_ms_range=[lo, hi],
                           bt_ms_range=[0.0, 0.0], total_ms_range=[alg + comp + lo, alg + comp + hi],
                           stage=dict(algorithmic="exact", compute="desktop", os="estimated (requirement bound)")))
    for name, (lo, hi) in BT_MS.items():
        routes.append(dict(route=name, algorithmic_ms=alg, compute_p99_ms=comp, os_roundtrip_ms_range=[0.0, 25.0],
                           bt_ms_range=[lo, hi], total_ms_range=[alg + comp + lo, alg + comp + 25.0 + hi],
                           stage=dict(algorithmic="exact", compute="desktop", os="estimated", bt="estimated (published claims)")))
    device = [json.loads(p.read_text()) for p in sorted(Path("results/m7/device").glob("*.json"))] \
        if Path("results/m7/device").exists() else []
    write_result("results/m7/system_latency.json", dict(operating_point=op, algorithmic_plus_compute_ms=alg + comp,
                 routes=routes, compute_source=src, on_device_measurements=device, sources=SOURCES), stage="estimated" if not device else "on-device")
    for r in routes:
        print(r["route"], [round(x, 1) for x in r["total_ms_range"]])


if __name__ == "__main__":
    main()

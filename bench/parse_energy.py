"""Parse an on-device energy session (android/scripts/energy_session.sh) into results/m8.

Reads batterystats (app-attributed estimated mAh and discharge) and, if the ``perfetto`` Python
package's trace processor is available, mean power-rail power over the session.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from clarity_tse.eval.latency import write_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    d = Path(a.session)
    bs = (d / "batterystats.txt").read_text(errors="ignore")
    res: dict = dict(tag=a.tag, minutes=a.minutes, device=(d / "device.txt").read_text().split("\n"))
    m = re.search(r"Discharge:\s*([\d.]+)\s*mAh", bs) or re.search(r"discharged.*?([\d.]+)\s*mAh", bs)
    if m:
        res["discharge_mah"] = float(m.group(1))
    m = re.search(r"Capacity:\s*(\d+)", bs)
    if m:
        res["capacity_mah"] = float(m.group(1))
        if "discharge_mah" in res:
            res["pct_battery_per_hour"] = 100 * res["discharge_mah"] / res["capacity_mah"] * 60 / a.minutes
    m = re.search(r"UID u0a\d+:\s*([\d.]+)", bs)
    if m:
        res["app_estimated_mah"] = float(m.group(1))
    try:
        from perfetto.trace_processor import TraceProcessor

        tp = TraceProcessor(trace=str(d / "clarity.pftrace"))
        q = tp.query("select t.name, (max(c.value)-min(c.value)) / ((max(c.ts)-min(c.ts))/1e9) / 1e6 as watts "
                     "from counter c join counter_track t on c.track_id = t.id "
                     "where t.name like 'power.rails%' group by t.name").as_pandas_dataframe()
        res["rails_mean_w"] = dict(zip(q.name, q.watts, strict=True))
        res["rails_total_mean_w"] = float(q.watts.sum())
    except Exception as e:  # trace processor not installed or no rails on this device
        res["rails_error"] = str(e)[:200]
    write_result(Path("results/m8/device") / f"{d.name}.json", res, stage="on-device", config=vars(a))
    print(res)


if __name__ == "__main__":
    main()

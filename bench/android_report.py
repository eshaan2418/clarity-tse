"""Record the Android build / unit-test / instrumented-test outcome as a results file.

Parses the Gradle JUnit XML reports and (optionally) the ClarityTest logcat line from the device the
instrumented tests ran on. Emulator runs are labelled ``emulator`` and are functional evidence only:
emulator audio timing says nothing about phone latency.

    uv run python -m bench.android_report --serial emulator-5560 --kind emulator
"""

from __future__ import annotations

import argparse
import glob
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from clarity_tse.eval.latency import write_result


def junit(pattern: str) -> dict:
    tot = fail = err = 0
    for f in glob.glob(pattern):
        r = ET.parse(f).getroot()
        tot += int(r.get("tests", 0))
        fail += int(r.get("failures", 0))
        err += int(r.get("errors", 0))
    return dict(tests=tot, failures=fail, errors=err)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None)
    ap.add_argument("--kind", default="emulator", choices=["emulator", "device"])
    ap.add_argument("--model", default="b16_s0")
    a = ap.parse_args()
    apk = Path("android/app/build/outputs/apk/debug/app-debug.apk")
    res = dict(apk_built=apk.exists(), apk_bytes=apk.stat().st_size if apk.exists() else None,
               unit_tests=junit("android/app/build/test-results/testDebugUnitTest/*.xml"),
               instrumented_tests=junit("android/app/build/outputs/androidTest-results/connected/debug/*.xml"),
               instrumented_target=a.kind, bundled_model=a.model)
    if a.serial:
        adb = str(Path.home() / "Library/Android/sdk/platform-tools/adb")
        log = subprocess.run([adb, "-s", a.serial, "logcat", "-d", "-s", "ClarityTest"], capture_output=True, text=True).stdout
        m = re.findall(r"callbacks=([\d.]+) xruns=([\d.]+) meanHopMs=([\d.]+) maxHopMs=([\d.]+) rate=([\d.]+)", log)
        if m:
            c, x, mean, mx, rate = map(float, m[-1])
            res["instrumented_engine_run"] = dict(callbacks=c, xruns=x, mean_hop_ms=mean, max_hop_ms=mx, rate=rate,
                                                  note=f"{a.kind}: functional check only, not latency evidence")
        res["device_model"] = subprocess.run([adb, "-s", a.serial, "shell", "getprop", "ro.product.model"],
                                             capture_output=True, text=True).stdout.strip()
    write_result("results/m7/android_build.json", res, stage=a.kind if a.kind == "emulator" else "on-device")
    print(res)


if __name__ == "__main__":
    main()

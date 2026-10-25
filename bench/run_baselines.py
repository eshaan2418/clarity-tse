"""M2: classical baselines (+ offline ceiling if trained) on the frozen test set, with CIs.

    uv run python -m bench.run_baselines --jobs 12 [--ceiling offline_s0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.evaluate import evaluate
from bench.summarize import summarize
from clarity_tse.synth import corpus

ORDER = ["unprocessed", "rnnoise", "ds", "mpdr", "mvdr_oracle"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--set", default="test_main")
    ap.add_argument("--ceiling", default="offline_s0")
    ap.add_argument("--out", default="results/m2")
    a = ap.parse_args()
    systems = list(ORDER)
    if (corpus.CACHE / "ckpt" / f"{a.ceiling}.pt").exists():
        systems.append(f"model:{a.ceiling}")
    out = Path(a.out)
    evaluate(a.set, systems, out / "per_item.csv", a.jobs, haspi=True)
    summarize(out / "per_item.csv", "unprocessed", out / "summary", order=[*ORDER, f"model:{a.ceiling}"])
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()

"""Counterbalanced trial lists: each participant gets every condition x SNR cell equally often,
items never repeat within a participant, and condition-to-item assignment rotates (Latin square)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = ["unprocessed", "rnnoise", "mpdr", "clarity_lo", "clarity_hi"]
SNRS = [-5, 0, 5]


def items_by_snr(manifest: dict) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {s: [] for s in SNRS}
    for it in manifest["items"]:
        if int(it["snr_db"]) in out:
            out[int(it["snr_db"])].append(it["id"])
    return out


def trial_list(pid_index: int, manifest: dict, per_cell: int = 2, seed: int = 2026) -> list[dict]:
    rng = np.random.default_rng([seed, pid_index])
    trials = []
    pools = items_by_snr(manifest)
    for s in SNRS:
        items = sorted(pools[s])[: per_cell * len(CONDITIONS)]
        # Latin-square rotation: item block b gets condition (b + pid) mod C
        for b in range(len(CONDITIONS)):
            cond = CONDITIONS[(b + pid_index) % len(CONDITIONS)]
            for it in items[b * per_cell : (b + 1) * per_cell]:
                trials.append(dict(item=it, condition=cond, snr=s))
    order = rng.permutation(len(trials))
    return [trials[i] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", type=int, default=20)
    ap.add_argument("--manifest", default="results/m1/testsets/test_main.json")
    ap.add_argument("--out", default="study/trial_lists")
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in range(a.participants):
        (out / f"P{p:02d}.json").write_text(json.dumps(trial_list(p, man), indent=1))
    print("wrote", a.participants, "lists to", out)


if __name__ == "__main__":
    main()

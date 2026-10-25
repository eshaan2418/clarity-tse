"""Simulated participants for dry-running the analysis (and a power simulation).

Simulated responses are clearly labelled (``simulated: true``) and written to study/data_simulated/,
never to study/data/. Effect sizes are *assumptions* for the dry run, not results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from study.randomize import CONDITIONS, SNRS

# assumed logit effects for the dry run only
ASSUMED = {"unprocessed": 0.0, "rnnoise": 0.1, "mpdr": 0.3, "clarity_lo": 0.8, "clarity_hi": 0.9}


def simulate(n_participants: int, seed: int = 0, words_per_trial: int = 8, trials_per_cell: int = 2) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_participants):
        u_p = rng.normal(0, 0.5)
        for s in SNRS:
            for c in CONDITIONS:
                for t in range(trials_per_cell):
                    item = f"{s}_{c}_{t}_{p}"
                    eta = -0.3 + 0.18 * s + ASSUMED[c] + u_p + rng.normal(0, 0.3)
                    k = int(rng.binomial(words_per_trial, 1 / (1 + np.exp(-eta))))
                    rows.append(dict(participant=f"S{p:02d}", condition=c, snr=s, item=item, words=words_per_trial,
                                     correct=k, effort=int(np.clip(round(5 - 2 * ASSUMED[c] - 0.15 * s + rng.normal(0, 1)), 1, 7)),
                                     mushra=float(np.clip(30 + 40 * ASSUMED[c] + 2 * s + rng.normal(0, 10), 0, 100)),
                                     simulated=True))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="study/data_simulated")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = simulate(a.n)
    with open(out / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "simulated trials ->", out)


if __name__ == "__main__":
    main()

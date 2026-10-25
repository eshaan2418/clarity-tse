"""Simulated participants for dry-running the full study pipeline (trial lists -> server-format
responses -> keyword scoring -> mixed models) without audio or real people.

Simulated responses are labelled ``simulated: true`` and written to study/data_simulated/ (never to
study/data/). Effect sizes below are *assumptions* used only to check that the analysis recovers a
known ordering; they are not results and must never be reported as such.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from study.randomize import CONDITIONS, SNRS, trial_list
from study.score import keywords

# assumed logit effects for the dry run only
ASSUMED = {"unprocessed": 0.0, "rnnoise": 0.1, "mpdr": 0.3, "clarity_lo": 0.8, "clarity_hi": 0.9}
VOCAB = ("garden river morning silver window paper captain winter yellow mountain letter horse "
         "candle village orange market thunder basket bridge lantern").upper().split()


def fake_manifest(rng: np.random.Generator) -> dict:
    """Same schema as study/stimuli_manifest.json, with synthetic sentences."""
    items = []
    for s in SNRS:
        for k in range(12):
            words = list(rng.choice(VOCAB, size=int(rng.integers(5, 9)), replace=False))
            items.append(dict(item=f"sim_word_{s:+03d}_{k:02d}", task="word", snr=float(s),
                              transcript="THE " + " ".join(words)))
    items += [dict(item=f"sim_mushra_{k}", task="mushra", snr=0.0, transcript="THE GARDEN") for k in range(4)]
    return dict(items=items, simulated=True)


def simulate(n_participants: int, seed: int = 0, effects: dict | None = None,
             sd_participant: float = 0.5) -> tuple[list[dict], dict]:
    effects = effects or ASSUMED
    rng = np.random.default_rng(seed)
    man = fake_manifest(rng)
    trans = {it["item"]: it["transcript"] for it in man["items"]}
    rows = []
    for p in range(n_participants):
        pid = f"S{p:02d}"
        u_p = rng.normal(0, sd_participant)
        lists = trial_list(p, man)
        for t in lists["word"]:
            kw = keywords(trans[t["item"]])
            eta = -0.3 + 0.18 * t["snr"] + effects[t["condition"]] + u_p + rng.normal(0, 0.3)
            heard = [w for w in kw if rng.random() < 1 / (1 + np.exp(-eta))]
            rows.append(dict(type="word", participant=pid, item=t["item"], condition=t["condition"], snr=t["snr"],
                             typed=" ".join(heard).lower(), simulated=True,
                             effort=int(np.clip(round(5 - 2 * ASSUMED[t["condition"]] - 0.15 * t["snr"] + rng.normal(0, 1)), 1, 7))))
        careless = p == n_participants - 1  # one participant fails reference post-screening
        for t in lists["mushra"]:
            for c in t["stimuli"]:
                if c == "reference":
                    r = rng.uniform(40, 80) if careless else rng.uniform(92, 100)
                elif c == "anchor":
                    r = rng.uniform(5, 25)
                else:
                    r = 30 + 40 * ASSUMED[c] + rng.normal(0, 10)
                rows.append(dict(type="mushra", participant=pid, item=t["item"], condition=c, snr=t["snr"],
                                 rating=float(np.clip(r, 0, 100)), simulated=True))
    return rows, man


def power(ns=(12, 16, 20), reps: int = 200, effect_logit: float = 0.42, alpha: float = 0.05) -> dict:
    """Power to detect clarity_lo > unprocessed (Holm-adjusted, elogit LMM) when the true effect is
    ``effect_logit`` (0.42 logit ~ +10 percentage points around 45 % correct), other conditions null.
    Participant SD 0.35 logit ~ 8 pp."""
    import warnings

    import pandas as pd

    from study.analyze import analyze

    eff = {c: 0.0 for c in CONDITIONS}
    eff["clarity_lo"] = effect_logit
    out = {}
    for n in ns:
        hits = 0
        for r in range(reps):
            rows, man = simulate(n, seed=10_000 + r, effects=eff, sd_participant=0.35)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = analyze(pd.DataFrame([x for x in rows if x["type"] == "word"]), man)
            c = res["elogit_lmm_clarity_lo_minus_condition"]["unprocessed"]
            hits += int(c["p_holm"] < alpha and c["coef"] > 0)
        out[str(n)] = hits / reps
    return dict(power=out, reps=reps, effect_logit=effect_logit, alpha=alpha, sd_participant_logit=0.35,
                simulated=True, note="assumed effect sizes; planning only")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="study/data_simulated")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    if a.power:
        from clarity_tse.eval.latency import write_result

        res = power(reps=a.reps)
        write_result("results/m10/power_simulated.json", res, config=vars(a), seed=10_000)
        print(res)
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, man = simulate(a.n)
    with open(out / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "stimuli_manifest.json").write_text(json.dumps(man, indent=1))
    print(len(rows), "simulated rows ->", out, "| conditions", CONDITIONS)


if __name__ == "__main__":
    main()

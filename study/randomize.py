"""Counterbalanced trial lists from study/stimuli_manifest.json (made by study/make_stimuli.py).

Word-recognition block: per SNR, 10 items split into 5 blocks of 2; block b gets condition
(b + participant) mod 5 (Latin-square rotation), so every participant hears every condition x SNR
cell equally often, never hears an item twice, and across 5 consecutive participants every item is
heard in every condition. Trial order is randomised per participant.
MUSHRA block: each MUSHRA item with its 5 conditions + hidden reference + anchor in random order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = ["unprocessed", "rnnoise", "mpdr", "clarity_lo", "clarity_hi"]
SNRS = [-5, 0, 5]
PER_CELL = 2
MANIFEST = Path(__file__).resolve().parent / "stimuli_manifest.json"


def trial_list(pid_index: int, manifest: dict, per_cell: int = PER_CELL, seed: int = 2026) -> dict:
    rng = np.random.default_rng([seed, pid_index])
    word = []
    for s in SNRS:
        items = sorted(it["item"] for it in manifest["items"] if it["task"] == "word" and int(it["snr"]) == s)
        items = items[: per_cell * len(CONDITIONS)]
        assert len(items) == per_cell * len(CONDITIONS), f"need {per_cell * len(CONDITIONS)} items at {s} dB"
        for b in range(len(CONDITIONS)):
            cond = CONDITIONS[(b + pid_index) % len(CONDITIONS)]
            word += [dict(item=it, condition=cond, snr=s) for it in items[b * per_cell : (b + 1) * per_cell]]
    word = [word[i] for i in rng.permutation(len(word))]
    mushra = []
    for it in sorted((x for x in manifest["items"] if x["task"] == "mushra"), key=lambda x: x["item"]):
        stims = [*CONDITIONS, "reference", "anchor"]
        mushra.append(dict(item=it["item"], snr=it["snr"], stimuli=[stims[i] for i in rng.permutation(len(stims))]))
    mushra = [mushra[i] for i in rng.permutation(len(mushra))]
    return dict(participant_index=pid_index, word=word, mushra=mushra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", type=int, default=20)
    ap.add_argument("--manifest", default=str(MANIFEST))
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

"""Render listening-study stimuli (local only; audio is git-ignored under data/cache/study_stimuli/).

Each item is a fresh test-split scene (speakers never used in training) in which the target says
exactly one complete LibriSpeech sentence, so typed responses can be scored against its transcript.
Every item is processed by every condition with the same code path as the objective evaluation
(bench.evaluate.run_system; oracle direction + 10 s enrollment for Clarity, true direction for MPDR):

  unprocessed | rnnoise | mpdr | clarity_lo (operating point, 16 ms) | clarity_hi (40 ms)

MUSHRA items additionally get the hidden reference (clean direct+early target) and the anchor
(unprocessed, low-passed at 3.5 kHz). All stimuli share one presentation offset (PRESENT_DB) so the
unprocessed mixtures stay below the limiter ceiling; the calibration noise uses the same offset.

    uv run python -m study.make_stimuli --jobs 8
"""

from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from clarity_tse.synth import corpus, scene
from study.score import keywords

OUT = corpus.CACHE / "study_stimuli"
MANIFEST = Path(__file__).resolve().parent / "stimuli_manifest.json"
CONDITION_SYSTEMS = {"unprocessed": "unprocessed", "rnnoise": "rnnoise", "mpdr": "mpdr",
                     "clarity_lo": "model:b16_s0", "clarity_hi": "model:b40_s0"}
SNRS = (-5.0, 0.0, 5.0)
KINDS = tuple(scene.SCENE_TYPES)
WORD_ITEMS_PER_CELL = 3  # per (kind, snr): 12 items per SNR (randomize.py uses 10)
MUSHRA_SNR = 0.0
PRESENT_DB = -8.0
def pick_sentence(spk: str, rng: np.random.Generator) -> str:
    utts = [u for u in corpus.librispeech_index("test")[spk] if 2.5 * 16000 <= u.n_samples <= 6.0 * 16000]
    utts = [u for u in utts if len(keywords(corpus.transcript(u.path))) >= 4] or utts
    return utts[int(rng.integers(len(utts)))].path


def item_specs() -> list[dict]:
    specs = []
    i = 0
    plan = [(k, s, "word") for k in KINDS for s in SNRS for _ in range(WORD_ITEMS_PER_CELL)]
    plan += [(k, MUSHRA_SNR, "mushra") for k in KINDS]
    for kind, snr, task in plan:
        sp = scene.sample_spec(kind, "test", seed=70_000 + i, snr_db=snr, duration_s=7.0, head_motion="static")
        rng = np.random.default_rng([70_000, i])
        sp["target_utt"] = pick_sentence(sp["sources"][0]["speaker"], rng)
        n = next(u.n_samples for u in corpus.librispeech_index("test")[sp["sources"][0]["speaker"]]
                 if u.path == sp["target_utt"])
        sp["duration_s"] = round(n / 16000 + 1.0, 2)
        sp["id"] = f"st_{task}_{kind}_{int(snr):+03d}_{i:03d}"
        sp["task"] = task
        specs.append(sp)
        i += 1
    return specs


def _render(sp: dict) -> dict:
    warnings.simplefilter("ignore")
    import torch

    from bench.evaluate import run_system
    from clarity_tse.embed import speaker

    torch.set_num_threads(1)
    r = scene.render(sp, enroll_seconds=10.0)
    a = dict(mix=r.mix, interference=r.interference,
             theta=scene.relative_azimuth(r.meta["target_world_az"], r.yaw).astype(np.float32))
    emb = speaker.embed(r.enrollment)
    g = 10 ** (PRESENT_DB / 20)
    outs = {c: run_system(s, a, emb) for c, s in CONDITION_SYSTEMS.items()}
    if sp["task"] == "mushra":
        outs["reference"] = r.target_early
        outs["anchor"] = sosfilt(butter(8, 3500, fs=16000, output="sos"), r.mix[0])
    peaks = {}
    for c, y in outs.items():
        (OUT / c).mkdir(parents=True, exist_ok=True)
        y = np.asarray(y, np.float32) * g
        peaks[c] = float(20 * np.log10(np.abs(y).max() + 1e-9))
        sf.write(OUT / c / f"{sp['id']}.wav", y, 16000, subtype="FLOAT")
    text = corpus.transcript(sp["target_utt"])
    return dict(item=sp["id"], task=sp["task"], kind=sp["kind"], snr=sp["snr_db"], transcript=text,
                n_keywords=len(keywords(text)), duration_s=sp["duration_s"], peak_dbfs=peaks,
                target_rel_az=r.meta["target_rel_az_start"])


def calibration_noise(seconds: float = 20.0) -> np.ndarray:
    """Speech-shaped noise (long-term spectrum of the study sentences) at the presented speech level."""
    specs = item_specs()
    acc = np.zeros(513)
    for sp in specs[:24]:
        x = corpus.load_utt(corpus.Utterance("", sp["target_utt"], 0))
        X = np.abs(np.fft.rfft(x[: len(x) // 1024 * 1024].reshape(-1, 1024) * np.hanning(1024), axis=1)) ** 2
        acc += X.mean(0)
    rng = np.random.default_rng(0)
    n = int(seconds * 16000)
    N = np.fft.rfft(rng.standard_normal(n))
    shape = np.interp(np.fft.rfftfreq(n, 1 / 16000), np.fft.rfftfreq(1024, 1 / 16000), np.sqrt(acc))
    y = np.fft.irfft(N * shape, n)
    level = 10 ** ((scene.TARGET_DBFS + PRESENT_DB) / 20)
    return (y / np.sqrt(np.mean(y**2)) * level).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    specs = item_specs()
    with get_context("spawn").Pool(a.jobs) as p:
        items = p.map(_render, specs, chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(OUT / "calibration.wav", calibration_noise(), 16000, subtype="FLOAT")
    MANIFEST.write_text(json.dumps(dict(present_db=PRESENT_DB, conditions=list(CONDITION_SYSTEMS),
                                        systems=CONDITION_SYSTEMS, items=items), indent=1))
    worst = max(max(it["peak_dbfs"].values()) for it in items)
    print(len(items), "items ->", OUT, f"(max stimulus peak {worst:.1f} dBFS)")


if __name__ == "__main__":
    main()

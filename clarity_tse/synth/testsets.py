"""Frozen, checksummed evaluation sets.

Audio is cached under data/cache/testsets/<set>/ (git-ignored, regenerable). The manifest with
specs and per-array SHA-256 is committed under results/m1/testsets/<set>.json so any drift in
regeneration is detected (``verify``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from clarity_tse.synth import corpus, scene

REPO = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO / "results" / "m1" / "testsets"
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0)
SET_DEFS: dict[str, dict[str, Any]] = {
    # name: split, scenes per (type, snr), duration, head motion
    "test_main": dict(split="test", per_cell=12, duration_s=6.0, head_motion="drift"),
    "val_main": dict(split="val", per_cell=4, duration_s=6.0, head_motion="drift"),
    "fixture_main": dict(split="test", per_cell=1, duration_s=2.0, head_motion="drift"),
}


def manifest_dir(name: str) -> Path:
    return REPO / "results" / "ci" / "testsets" if name.startswith("fixture") else MANIFEST_DIR


def cache_dir(name: str) -> Path:
    return corpus.CACHE / "testsets" / name


def specs_for(name: str) -> list[dict[str, Any]]:
    d = SET_DEFS[name]
    specs = []
    i = 0
    for kind in scene.SCENE_TYPES:
        for snr in SNRS:
            for _ in range(d["per_cell"]):
                sp = scene.sample_spec(kind, d["split"], seed=10_000 + i, snr_db=snr,
                                       duration_s=d["duration_s"], head_motion=d["head_motion"])
                sp["id"] = f"{kind}_{int(snr):+03d}_{i:04d}"
                if name.startswith("fixture"):
                    sp["room"]["max_order_cap"] = 6
                specs.append(sp)
                i += 1
    return specs


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


def render_one(args: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    name, sp = args
    out = cache_dir(name) / f"{sp['id']}.npz"
    r = scene.render(sp, enroll_seconds=10.0 if not name.startswith("fixture") else 2.0)
    arrays = dict(mix=r.mix, target_early=r.target_early, target_reverb=r.target_reverb,
                  interference=r.interference, yaw=r.yaw, enrollment=r.enrollment)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)  # type: ignore[arg-type]
    return dict(id=sp["id"], kind=sp["kind"], snr_db=sp["snr_db"], meta=r.meta, spec=sp,
                sha256={k: _sha(v) for k, v in arrays.items()})


def load(name: str, item_id: str) -> dict[str, np.ndarray]:
    with np.load(cache_dir(name) / f"{item_id}.npz") as z:
        return {k: z[k] for k in z.files}


def manifest(name: str) -> dict[str, Any]:
    return json.loads((manifest_dir(name) / f"{name}.json").read_text())


def verify(name: str, n_check: int | None = None) -> list[str]:
    """Re-render items and compare checksums; returns list of mismatching ids."""
    man = manifest(name)
    bad = []
    for it in man["items"][:n_check]:
        r = render_one((name + "_verify", it["spec"]))
        if r["sha256"] != it["sha256"]:
            bad.append(it["id"])
    return bad

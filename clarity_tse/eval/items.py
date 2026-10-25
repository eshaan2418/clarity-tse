"""Access to frozen evaluation items with oracle side information and cached enrollment embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from clarity_tse.synth import scene, testsets


def oracle_theta(item_meta: dict[str, Any], yaw: np.ndarray) -> np.ndarray:
    """Per-sample oracle target azimuth relative to the head (deg)."""
    return scene.relative_azimuth(item_meta["target_world_az"], yaw).astype(np.float32)


def frame_theta(theta_samples: np.ndarray, hop: int, n_frames: int) -> np.ndarray:
    """Sample-rate azimuth -> one value per STFT frame (value at the frame's last sample)."""
    idx = np.minimum((np.arange(n_frames) + 1) * hop - 1, len(theta_samples) - 1)
    return theta_samples[idx]


def enrollment_embeddings(set_name: str) -> dict[str, np.ndarray]:
    from clarity_tse.embed import speaker

    path = testsets.cache_dir(set_name) / f"emb_{speaker._which()}.npz"
    if path.exists():
        z = np.load(path)
        return dict(zip(z["ids"].tolist(), z["emb"], strict=True))
    man = testsets.manifest(set_name)
    ids, embs = [], []
    for it in man["items"]:
        a = testsets.load(set_name, it["id"])
        ids.append(it["id"])
        embs.append(speaker.embed(a["enrollment"]))
    np.savez(path, ids=np.array(ids), emb=np.stack(embs))
    return dict(zip(ids, embs, strict=True))


def iter_items(set_name: str, ids: list[str] | None = None):
    man = testsets.manifest(set_name)
    for it in man["items"]:
        if ids is not None and it["id"] not in ids:
            continue
        a = testsets.load(set_name, it["id"])
        a["theta"] = oracle_theta(it["meta"], a["yaw"])
        yield it, a


def stratified_ids(set_name: str, per_cell: int) -> list[str]:
    man = testsets.manifest(set_name)
    seen: dict[tuple[str, float], int] = {}
    out = []
    for it in man["items"]:
        k = (it["kind"], it["snr_db"])
        if seen.get(k, 0) < per_cell:
            out.append(it["id"])
            seen[k] = seen.get(k, 0) + 1
    return out

"""Seeded multi-talker scene simulator with listener head rotation.

A scene is a JSON-serialisable *spec* (room, listener, yaw trajectory, sources) produced by
``sample_spec`` and rendered deterministically by ``render``. Rendering uses pyroomacoustics
image-source RIRs for a binaural earbud pair (see dsp/array.py for the geometry). Head rotation
is rendered by computing RIRs on a yaw grid (all grid mics share one room/image computation) and
crossfading the per-grid-point signals according to the yaw trajectory.

Calibration: the reverberant target at the left ear has RMS ``TARGET_DBFS`` (taken as 65 dB SPL,
i.e. ``SPL_AT_0DBFS`` dB SPL corresponds to RMS 1.0).
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

from clarity_tse.dsp.array import EAR_SPACING
from clarity_tse.synth import corpus

SR = corpus.SR
TARGET_DBFS = -26.0
SPEECH_SPL = 65.0
SPL_AT_0DBFS = SPEECH_SPL - TARGET_DBFS  # 91 dB SPL <-> RMS 1
EARLY_MS = 50.0
YAW_GRID_DEG = 5.0
RT60_CALIB = 0.65  # pyroomacoustics ISM overshoots Sabine RT60; measured RT60 stored in meta

SCENE_TYPES: dict[str, dict[str, Any]] = {
    # room dims ranges (m), nominal RT60 range (s), #competing talkers, noise categories, babble
    "restaurant": dict(dims=((6, 12), (5, 9), (2.8, 3.5)), rt60=(0.4, 0.8), talkers=(2, 4),
                       noise=["clatter", "kitchen"], babble=(6, 10), music=False),
    "party": dict(dims=((5, 10), (4, 8), (2.6, 3.2)), rt60=(0.4, 0.9), talkers=(3, 5),
                  noise=["party"], babble=(8, 12), music=True),
    "street": dict(dims=((25, 40), (15, 25), (6, 10)), rt60=(0.15, 0.35), talkers=(1, 2),
                   noise=["street"], babble=(0, 4), music=False),
    "kitchen": dict(dims=((3, 5), (3, 4.5), (2.4, 2.8)), rt60=(0.3, 0.6), talkers=(1, 2),
                    noise=["kitchen", "clatter"], babble=(0, 0), music=False),
}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


@dataclass
class Rendered:
    mix: np.ndarray  # (2, T) binaural mixture
    target_early: np.ndarray  # (T,) direct+early target at left ear (reference for metrics)
    target_reverb: np.ndarray  # (2, T) full reverberant target
    interference: np.ndarray  # (2, T) everything else
    yaw: np.ndarray  # (T,) head yaw (deg)
    enrollment: np.ndarray  # (Te,) dry enrollment speech of the target speaker
    meta: dict[str, Any]
    sources_early: list[np.ndarray] | None = None  # optional per-source early images (M4)


def yaw_trajectory(keyframes: list[tuple[float, float]], n: int) -> np.ndarray:
    """Piecewise-linear yaw (deg) from (time_s, yaw_deg) keyframes, smoothed with a raised cosine."""
    t = np.arange(n) / SR
    kt = np.array([k[0] for k in keyframes])
    ky = np.array([k[1] for k in keyframes])
    if len(kt) == 1:
        return np.full(n, ky[0])
    yaw = np.interp(t, kt, ky)
    # cosine easing within each segment for physiologically plausible acceleration
    seg = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2)
    frac = np.clip((t - kt[seg]) / np.maximum(kt[seg + 1] - kt[seg], 1e-9), 0, 1)
    eased = 0.5 - 0.5 * np.cos(np.pi * frac)
    yaw = ky[seg] + (ky[seg + 1] - ky[seg]) * eased
    yaw[t >= kt[-1]] = ky[-1]
    yaw[t < kt[0]] = ky[0]
    return yaw


def _place(listener: np.ndarray, az_deg: float, dist: float, height: float, dims: np.ndarray) -> list[float]:
    d = dist
    for _ in range(20):
        p = listener[:2] + d * np.array([np.cos(np.deg2rad(az_deg)), np.sin(np.deg2rad(az_deg))])
        if np.all(p > 0.3) and np.all(p < dims[:2] - 0.3):
            return [float(p[0]), float(p[1]), float(height)]
        d *= 0.85
    p = np.clip(p, 0.3, dims[:2] - 0.3)
    return [float(p[0]), float(p[1]), float(height)]


def render_sources(spec: dict[str, Any], signals: list[np.ndarray], yaw: np.ndarray,
                   early_for: set[int] | None = None) -> tuple[list[np.ndarray], list[np.ndarray | None], float]:
    """Convolve each source with (yaw-interpolated) binaural RIRs.

    Returns per-source reverberant images (2, T), early images at left ear (T,) for indices in
    ``early_for`` and the measured RT60 of the first RIR.
    """
    dims = np.array(spec["room"]["dims"])
    try:
        e_abs, max_order = pra.inverse_sabine(spec["room"]["rt60"] * RT60_CALIB, dims)
    except ValueError:  # large, very absorbent space (street): near-free-field with few reflections
        e_abs, max_order = 0.9, 3
    max_order = int(min(max_order, spec["room"].get("max_order_cap", 60)))
    room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(e_abs), max_order=max_order,
                       air_absorption=False)
    for s in spec["sources"]:
        room.add_source(s["pos"])
    lp = np.array(spec["listener"]["pos"])
    lo = float(np.floor(yaw.min() / YAW_GRID_DEG) * YAW_GRID_DEG)
    hi = float(np.ceil(yaw.max() / YAW_GRID_DEG) * YAW_GRID_DEG)
    grid = np.arange(lo, hi + YAW_GRID_DEG / 2, YAW_GRID_DEG)
    mics = []
    for g in grid:
        for side in (+1, -1):  # left ear at yaw+90, right ear at yaw-90
            ang = np.deg2rad(g + side * 90)
            mics.append(lp + np.array([np.cos(ang), np.sin(ang), 0.0]) * EAR_SPACING / 2)
    room.add_microphone_array(np.array(mics).T)
    room.compute_rir()
    n = len(yaw)
    # interpolation weights per grid point
    pos = (yaw - grid[0]) / YAW_GRID_DEG
    i0 = np.clip(np.floor(pos).astype(int), 0, len(grid) - 1)
    frac = np.clip(pos - i0, 0, 1)
    weights = np.zeros((len(grid), n))
    weights[i0, np.arange(n)] += 1 - frac
    i1 = np.clip(i0 + 1, 0, len(grid) - 1)
    weights[i1, np.arange(n)] += frac
    used = np.where(weights.max(axis=1) > 0)[0]
    early_n = int(EARLY_MS * 1e-3 * SR)
    images: list[np.ndarray] = []
    earlies: list[np.ndarray | None] = []
    for si, sig in enumerate(signals):
        img = np.zeros((2, n))
        early = np.zeros(n) if early_for and si in early_for else None
        for gi in used:
            for ch in range(2):
                h = np.asarray(room.rir[2 * gi + ch][si])
                conv = fftconvolve(sig, h)[:n]
                img[ch] += weights[gi] * conv
                if early is not None and ch == 0:
                    d0 = int(np.argmax(np.abs(h)))
                    he = h[: d0 + early_n].copy()
                    # 5 ms half-cosine taper at the early/late boundary
                    tap = min(80, len(he))
                    he[-tap:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, tap))
                    early += weights[gi] * fftconvolve(sig, he)[:n]
        images.append(img)
        earlies.append(early)
    rt = float(pra.experimental.measure_rt60(np.asarray(room.rir[0][0]), SR))
    return images, earlies, rt


def _speech_track(utts: list[corpus.Utterance], n: int, rng: np.random.Generator,
                  start_offset: int = 0) -> tuple[np.ndarray, list[str]]:
    out = np.zeros(n, dtype=np.float32)
    used = []
    pos = start_offset
    order = rng.permutation(len(utts))
    for idx in order:
        if pos >= n:
            break
        x = corpus.load_utt(utts[idx])
        x = x / (_rms(x) + 1e-9) * 0.05
        m = min(n - pos, len(x))
        out[pos : pos + m] = x[:m]
        used.append(utts[idx].path)
        pos += m + int(rng.uniform(0.1, 0.4) * SR)
    return out, used


def sample_spec(kind: str, split: str, seed: int, snr_db: float | None = None, duration_s: float = 6.0,
                head_motion: str = "static", n_talkers: int | None = None) -> dict[str, Any]:
    """Sample a scene spec. ``head_motion`` in {static, drift, turn}."""
    cfg = SCENE_TYPES[kind]
    rng = np.random.default_rng([seed, zlib.crc32(kind.encode()), {"train": 0, "val": 1, "test": 2}[split]])
    dims = [float(rng.uniform(*r)) for r in cfg["dims"]]
    dims_a = np.array(dims)
    listener = [float(rng.uniform(1.0, dims[0] - 1.0)), float(rng.uniform(1.0, dims[1] - 1.0)), 1.2]
    yaw0 = float(rng.uniform(-180, 180))
    if head_motion == "static":
        keys = [(0.0, yaw0)]
    elif head_motion == "drift":
        keys = [(0.0, yaw0), (duration_s, yaw0 + float(rng.uniform(-15, 15)))]
    elif head_motion == "turn":
        t0 = float(rng.uniform(1.0, duration_s - 2.0))
        keys = [(0.0, yaw0), (t0, yaw0), (t0 + 0.6, yaw0 + float(rng.choice([-1, 1]) * rng.uniform(20, 45)))]
    else:
        raise ValueError(head_motion)
    spk_index = corpus.librispeech_index(split)
    speakers = sorted(spk_index)
    if n_talkers is None:
        n_talkers = int(rng.integers(cfg["talkers"][0], cfg["talkers"][1] + 1))
    n_babble = int(rng.integers(cfg["babble"][0], cfg["babble"][1] + 1))
    n_talkers = min(n_talkers, len(speakers) - 1)
    n_babble = min(n_babble, len(speakers) - 1 - n_talkers)  # only binds on tiny fixture corpora
    chosen = rng.choice(len(speakers), size=1 + n_talkers + n_babble, replace=False)
    target_spk = speakers[chosen[0]]
    sources: list[dict[str, Any]] = []
    # target roughly in front (people face who they talk to), +-25 deg, 0.8-2 m
    t_rel = float(rng.uniform(-25, 25))
    t_az = yaw0 + t_rel
    sources.append(dict(role="target", speaker=target_spk, az=t_az, dist=float(rng.uniform(0.8, 2.0)),
                        height=float(rng.uniform(1.15, 1.75)), gain_db=0.0))
    for j in range(n_talkers):
        # competing talkers anywhere at least 20 deg away from target
        while True:
            az = float(rng.uniform(-180, 180))
            if abs((az - t_az + 180) % 360 - 180) > 20:
                break
        sources.append(dict(role="interferer", speaker=speakers[chosen[1 + j]], az=az,
                            dist=float(rng.uniform(1.0, 3.5)), height=float(rng.uniform(1.15, 1.75)),
                            gain_db=float(rng.uniform(-4, 2))))
    babble_spk = [speakers[c] for c in chosen[1 + n_talkers :]]
    n_groups = min(4, n_babble)
    for j in range(n_groups):
        # each babble source is a sum of several distant talkers (keeps RIR cost bounded)
        sources.append(dict(role="babble", speakers=babble_spk[j::n_groups],
                            az=float(rng.uniform(-180, 180)), dist=float(rng.uniform(3.0, 8.0)),
                            height=float(rng.uniform(1.1, 1.8)), gain_db=float(rng.uniform(-10, -4))))
    esc = corpus.esc50_index(split)
    for cat_group in cfg["noise"]:
        cats = [c for c in corpus.NOISE_CATEGORIES[cat_group] if c in esc]
        if cats:
            c = cats[int(rng.integers(len(cats)))]
            f = esc[c][int(rng.integers(len(esc[c])))]
            src = dict(kind="esc50", file=f, category=c)
        else:
            src = dict(kind="synthetic", synth="clatter" if cat_group != "street" else "pink")
        sources.append(dict(role="noise", noise=src, az=float(rng.uniform(-180, 180)),
                            dist=float(rng.uniform(1.5, 5.0)), height=float(rng.uniform(0.8, 2.0)),
                            gain_db=float(rng.uniform(-6, 0))))
    if cfg["music"]:
        sources.append(dict(role="noise", noise=dict(kind="synthetic", synth="music"),
                            az=float(rng.uniform(-180, 180)), dist=float(rng.uniform(2.5, 6.0)),
                            height=2.0, gain_db=float(rng.uniform(-6, 0))))
    if kind == "street":
        sources.append(dict(role="noise", noise=dict(kind="synthetic", synth="pink"),
                            az=float(rng.uniform(-180, 180)), dist=float(rng.uniform(4.0, 10.0)),
                            height=1.0, gain_db=float(rng.uniform(-4, 0))))
    for s in sources:
        s["pos"] = _place(np.array(listener), s["az"], s["dist"], s["height"], dims_a)
    if snr_db is None:
        snr_db = float(rng.uniform(-5, 15))
    return dict(kind=kind, split=split, seed=seed, sr=SR, duration_s=duration_s, snr_db=float(snr_db),
                room=dict(dims=dims, rt60=float(rng.uniform(*cfg["rt60"]))),
                listener=dict(pos=listener), yaw_keys=keys, sources=sources,
                sub_seed=int(rng.integers(2**31)))


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def _noise_signal(src: dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    if src["kind"] == "esc50":
        x = corpus.read_audio(corpus.EXTERNAL / src["file"])
        # loop/trim ESC-50 5 s clip with random offset
        reps = int(np.ceil((n + len(x)) / max(1, len(x))))
        x = np.tile(x, reps)
        off = int(rng.integers(0, len(x) - n))
        x = x[off : off + n]
    else:
        x = corpus.synthetic_noise(src["synth"], n, rng)
    return (x / (_rms(x) + 1e-9) * 0.05).astype(np.float32)


def enrollment_for(speaker: str, split: str, exclude: set[str], rng: np.random.Generator,
                   seconds: float = 10.0) -> np.ndarray:
    utts = [u for u in corpus.librispeech_index(split)[speaker] if u.path not in exclude]
    n = int(seconds * SR)
    out, _ = _speech_track(utts, n, rng)
    return out


def render(spec: dict[str, Any], enroll_seconds: float = 10.0, with_enrollment: bool = True) -> Rendered:
    n = int(spec["duration_s"] * SR)
    rng = np.random.default_rng(spec["sub_seed"])
    spk_index = corpus.librispeech_index(spec["split"])
    signals: list[np.ndarray] = []
    used_paths: set[str] = set()
    for s in spec["sources"]:
        if s["role"] == "target" and "target_utt" in spec:
            # single complete sentence (listening-study stimuli: transcript is known), onset 0.5 s
            x = corpus.load_utt(corpus.Utterance(s["speaker"], spec["target_utt"], 0))
            x = x / (_rms(x) + 1e-9) * 0.05
            sig = np.zeros(n, dtype=np.float32)
            m = min(len(x), n - SR // 2)
            sig[SR // 2 : SR // 2 + m] = x[:m]
            used_paths.add(spec["target_utt"])
        elif "speaker" in s:
            utts = spk_index[s["speaker"]]
            off = 0 if s["role"] == "target" else int(rng.integers(0, int(0.5 * SR)))
            sig, used = _speech_track(utts, n, rng, start_offset=off)
            if s["role"] == "target":
                used_paths.update(used)
        elif "speakers" in s:
            sig = np.zeros(n, dtype=np.float32)
            for spk in s["speakers"]:
                sig += _speech_track(spk_index[spk], n, rng, start_offset=int(rng.integers(0, SR)))[0]
            sig /= np.sqrt(max(1, len(s["speakers"])))
        else:
            sig = _noise_signal(s["noise"], n, rng)
        signals.append(sig * 10 ** (s["gain_db"] / 20))
    yaw = yaw_trajectory([tuple(k) for k in spec["yaw_keys"]], n)
    images, earlies, rt = render_sources(spec, signals, yaw, early_for={0})
    target_img, target_early = images[0], earlies[0]
    assert target_early is not None
    interf = np.sum(images[1:], axis=0) if len(images) > 1 else np.zeros((2, n))
    g_t = 10 ** (TARGET_DBFS / 20) / _rms(target_img[0])
    target_img *= g_t
    target_early = target_early * g_t
    g_i = _rms(target_img[0]) / (_rms(interf[0]) + 1e-12) * 10 ** (-spec["snr_db"] / 20)
    interf *= g_i
    sensor = np.random.default_rng(spec["sub_seed"] + 1).standard_normal((2, n)) * 10 ** (-75 / 20)
    mix = target_img + interf + sensor
    enroll = (enrollment_for(spec["sources"][0]["speaker"], spec["split"], used_paths, rng, enroll_seconds)
              if with_enrollment else np.zeros(0, dtype=np.float32))
    lp = np.array(spec["listener"]["pos"])
    tp = np.array(spec["sources"][0]["pos"])
    t_world = float(np.rad2deg(np.arctan2(tp[1] - lp[1], tp[0] - lp[0])))
    meta = dict(spec_hash=spec_hash(spec), rt60_measured=rt, target_world_az=t_world,
                target_rel_az_start=float((t_world - yaw[0] + 180) % 360 - 180),
                n_sources=len(spec["sources"]))
    return Rendered(mix=mix.astype(np.float32), target_early=target_early.astype(np.float32),
                    target_reverb=target_img.astype(np.float32), interference=(interf + sensor).astype(np.float32),
                    yaw=yaw.astype(np.float32), enrollment=enroll.astype(np.float32), meta=meta)


def relative_azimuth(world_az: float, yaw: np.ndarray) -> np.ndarray:
    return (world_az - yaw + 180) % 360 - 180

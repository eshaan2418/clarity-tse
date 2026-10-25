"""M4 target-selection scenarios (test-split speakers only).

turn    : partners A (-35 deg) and B (+35 deg) both talking; listener faces A, turns to B at ~4 s, back at ~8 s
similar : target A ahead; distractor D with the most similar enrolled-voice embedding at +-45 deg
moving  : A walks from 0 to 50 deg; listener's head follows with lag; B static at -50 deg
pause   : A ahead talks, pauses 4-7 s, resumes; B at 40 deg talks throughout

All scenes also contain a background babble source and an ESC-50 noise. The *attended* talker
(ground truth) is the one the listener faces (turn/moving) or A (similar/pause).
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from clarity_tse.synth import corpus, scene

SR = corpus.SR
KINDS = ("turn", "similar", "moving", "pause")


@functools.lru_cache(maxsize=1)
def _speaker_means(split: str) -> tuple[list[str], np.ndarray]:
    from clarity_tse.embed import speaker

    tab = speaker.utterance_embedding_table(split, device="cpu")
    idx = corpus.librispeech_index(split)
    spks = sorted(idx)
    M = []
    for s in spks:
        e = np.mean([tab[u.path] for u in idx[s] if u.path in tab], axis=0)
        M.append(e / np.linalg.norm(e))
    return spks, np.stack(M)


def most_similar_speaker(spk: str, split: str, exclude: set[str]) -> str:
    spks, M = _speaker_means(split)
    i = spks.index(spk)
    sims = M @ M[i]
    for j in np.argsort(-sims):
        if spks[j] != spk and spks[j] not in exclude:
            return spks[j]
    raise RuntimeError("no distractor")


def _track(spk: str, n: int, rng: np.random.Generator, used: set[str], split: str) -> np.ndarray:
    utts = [u for u in corpus.librispeech_index(split)[spk] if u.path not in used]
    x, paths = scene._speech_track(utts, n, rng)
    used.update(paths)
    return x


def make(kind: str, seed: int, split: str = "test", duration_s: float = 12.0) -> dict[str, Any]:
    rng = np.random.default_rng([seed, KINDS.index(kind), 777])
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    spks = sorted(corpus.librispeech_index(split))
    pick = [spks[i] for i in rng.choice(len(spks), size=8, replace=False)]
    A, B, C = pick[0], pick[1], pick[2]
    babble_spk = pick[3:6]
    if kind == "similar":
        B = most_similar_speaker(A, split, set(pick[1:]))
    dims = [float(rng.uniform(6, 10)), float(rng.uniform(5, 8)), 3.0]
    listener = [dims[0] / 2 + float(rng.uniform(-0.5, 0.5)), dims[1] / 2 + float(rng.uniform(-0.5, 0.5)), 1.2]
    base = float(rng.uniform(-180, 180))
    used: set[str] = set()
    talkers: list[dict[str, Any]] = []  # name, speaker, world az (const or array), signal
    sA = _track(A, n, rng, used, split)
    sB = _track(B, n, rng, used, split)
    if kind == "turn":
        t1, t2 = float(rng.uniform(3.5, 4.5)), float(rng.uniform(7.5, 8.5))
        keys = [(0, base - 35), (t1, base - 35), (t1 + 0.6, base + 35), (t2, base + 35), (t2 + 0.6, base - 35)]
        talkers += [dict(name="A", speaker=A, az=base - 35, sig=sA), dict(name="B", speaker=B, az=base + 35, sig=sB)]
        att = np.where((t >= t1 + 0.3) & (t < t2 + 0.3), 1, 0)
        turn_times = [(t1 + 0.6, 1), (t2 + 0.6, 0)]
    elif kind == "similar":
        side = float(rng.choice([-1, 1]))
        keys = [(0, base), (duration_s, base + float(rng.uniform(-8, 8)))]
        talkers += [dict(name="A", speaker=A, az=base, sig=sA), dict(name="B", speaker=B, az=base + side * 45, sig=sB)]
        att = np.zeros(n, dtype=int)
        turn_times = []
    elif kind == "moving":
        az_a = base + np.interp(t, [0, 2, 10, duration_s], [0, 0, 50, 50])
        lag = int(0.8 * SR)
        yaw_follow = np.concatenate([np.full(lag, az_a[0]), az_a[:-lag]])
        keys = None
        yaw = base + 0.85 * (yaw_follow - base)
        talkers += [dict(name="A", speaker=A, az=az_a, sig=sA), dict(name="B", speaker=B, az=base - 50, sig=sB)]
        att = np.zeros(n, dtype=int)
        turn_times = []
    elif kind == "pause":
        p0, p1 = float(rng.uniform(3.5, 4.5)), float(rng.uniform(6.5, 7.5))
        sA = sA * ((t < p0) | (t >= p1))
        keys = [(0, base), (duration_s, base + float(rng.uniform(-5, 5)))]
        talkers += [dict(name="A", speaker=A, az=base, sig=sA), dict(name="B", speaker=B, az=base + 40, sig=sB)]
        att = np.zeros(n, dtype=int)
        turn_times = []
    else:
        raise ValueError(kind)
    if keys is not None:
        yaw = scene.yaw_trajectory(keys, n)
    # third talker behind / to the side, quieter
    talkers.append(dict(name="C", speaker=C, az=base + float(rng.uniform(110, 250)),
                        sig=_track(C, n, rng, used, split) * 10 ** (-3 / 20)))
    # sources: moving talkers are split into position segments crossfaded in time
    sources, signals, owner = [], [], []
    for ti, tk in enumerate(talkers):
        az = np.asarray(tk["az"], dtype=float)
        if az.ndim == 0:
            sources.append(dict(pos=scene._place(np.array(listener), float(az), 1.3, 1.5, np.array(dims))))
            signals.append(tk["sig"])
            owner.append(ti)
        else:
            nodes = np.linspace(az.min(), az.max(), 8)
            pos = np.clip((az - nodes[0]) / max(nodes[1] - nodes[0], 1e-6), 0, len(nodes) - 1)
            for k, a in enumerate(nodes):
                w = np.clip(1 - np.abs(pos - k), 0, 1)
                sources.append(dict(pos=scene._place(np.array(listener), float(a), 1.3, 1.5, np.array(dims))))
                signals.append(tk["sig"] * w)
                owner.append(ti)
    # background babble and noise
    bab = sum(_track(s, n, rng, used, split) for s in babble_spk) / np.sqrt(len(babble_spk))
    sources.append(dict(pos=scene._place(np.array(listener), base + float(rng.uniform(60, 300)), 4.0, 1.6, np.array(dims))))
    signals.append(bab)
    owner.append(-1)
    esc = corpus.esc50_index(split)
    cats = [c for c in corpus.NOISE_CATEGORIES["clatter"] if c in esc]
    nz = scene._noise_signal(dict(kind="esc50", file=esc[cats[0]][int(rng.integers(len(esc[cats[0]])))]) if cats
                             else dict(kind="synthetic", synth="clatter"), n, rng)
    sources.append(dict(pos=scene._place(np.array(listener), base + float(rng.uniform(-180, 180)), 3.0, 1.0, np.array(dims))))
    signals.append(nz)
    owner.append(-1)
    spec = dict(room=dict(dims=dims, rt60=float(rng.uniform(0.35, 0.6)), max_order_cap=40), listener=dict(pos=listener),
                sources=sources)
    images, earlies, rt = scene.render_sources(spec, signals, yaw, early_for=set(range(len(signals))))
    T = len(talkers)
    tk_img = [np.zeros((2, n)) for _ in range(T)]
    tk_early = [np.zeros(n) for _ in range(T)]
    bg = np.zeros((2, n))
    for si, o in enumerate(owner):
        if o >= 0:
            tk_img[o] += images[si]
            e = earlies[si]
            assert e is not None
            tk_early[o] += e
        else:
            bg += images[si]
    # calibrate: each partner at -26 dBFS (when active), babble+noise at 3 dB below partners
    g = 10 ** (scene.TARGET_DBFS / 20) / (np.sqrt(np.mean(tk_img[0][0] ** 2)) + 1e-9)
    tk_img = [x * g for x in tk_img]
    tk_early = [x * g for x in tk_early]
    gb = np.sqrt(np.mean(tk_img[0][0] ** 2)) / (np.sqrt(np.mean(bg[0] ** 2)) + 1e-9) * 10 ** (-float(rng.uniform(0, 6)) / 20)
    bg *= gb
    mix = sum(tk_img) + bg
    att_ref = np.where(att == 0, tk_early[0], tk_early[1])
    world_az = [np.broadcast_to(np.asarray(tk["az"], float), (n,)) for tk in talkers]
    enroll = {}
    for tk in talkers[:2]:
        enroll[tk["name"]] = scene.enrollment_for(tk["speaker"], split, used, rng, 10.0)
    return dict(kind=kind, seed=seed, mix=mix.astype(np.float32), yaw=yaw.astype(np.float32), attended=att,
                attended_ref=att_ref.astype(np.float32), talker_early=[x.astype(np.float32) for x in tk_early],
                talker_names=[tk["name"] for tk in talkers], speakers=[tk["speaker"] for tk in talkers],
                world_az=world_az, enroll=enroll, turn_times=turn_times, rt60=rt)

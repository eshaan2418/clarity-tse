"""Training data: a bank of precomputed binaural RIRs + on-the-fly mixing.

Each bank entry is a static-head scene geometry from ``scene.sample_spec`` (train or val split),
with RIRs (S sources x 2 ears) truncated to ``RIR_LEN``. At training time we keep the geometry and
source *roles* but re-draw speech (train speakers only), noise (ESC-50 train folds / synthetic),
gains and SNR, so every example is new.

Augmentations implementing the "fail gracefully" / fallback behaviours:
  * p_absent  : enrolled target not talking -> reference is silence (model must attenuate)
  * p_no_voice: no enrollment embedding       (direction-only mode)
  * p_no_doa  : no direction (theta = NaN)    (voice-only mode)
  * theta noise ~ N(0, doa_noise_deg) mimics DOA/IMU error from the selection policy
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pyroomacoustics as pra
import torch
from scipy.signal import fftconvolve

from clarity_tse.synth import corpus, scene

RIR_LEN = 16000
SR = corpus.SR


def bank_dir(split: str) -> Path:
    return corpus.CACHE / "rir_bank" / split


def _bank_entry(args: tuple[str, int, int]) -> str:
    split, i, max_order_cap = args
    out = bank_dir(split) / f"{i:05d}.npz"
    if out.exists():
        return str(out)
    kind = list(scene.SCENE_TYPES)[i % len(scene.SCENE_TYPES)]
    spec = scene.sample_spec(kind, split, seed=50_000 + i, head_motion="static")
    spec["room"]["max_order_cap"] = max_order_cap
    dims = np.array(spec["room"]["dims"])
    try:
        e_abs, mo = pra.inverse_sabine(spec["room"]["rt60"] * scene.RT60_CALIB, dims)
    except ValueError:
        e_abs, mo = 0.9, 3
    room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(e_abs), max_order=int(min(mo, max_order_cap)))
    for s in spec["sources"]:
        room.add_source(s["pos"])
    lp = np.array(spec["listener"]["pos"])
    yaw = spec["yaw_keys"][0][1]
    mics = [lp + np.array([np.cos(np.deg2rad(yaw + sd * 90)), np.sin(np.deg2rad(yaw + sd * 90)), 0]) * 0.09
            for sd in (+1, -1)]
    room.add_microphone_array(np.array(mics).T)
    room.compute_rir()
    S = len(spec["sources"])
    rirs = np.zeros((S, 2, RIR_LEN), dtype=np.float16)
    for si in range(S):
        for ch in range(2):
            h = np.asarray(room.rir[ch][si])[:RIR_LEN]
            rirs[si, ch, : len(h)] = h
    rel = []
    for s in spec["sources"]:
        p = np.array(s["pos"])
        world = np.rad2deg(np.arctan2(p[1] - lp[1], p[0] - lp[0]))
        rel.append(float((world - yaw + 180) % 360 - 180))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rirs=rirs, rel_az=np.array(rel, dtype=np.float32), spec=json.dumps(spec))
    return str(out)


def build_bank(split: str, n: int, jobs: int = 8, max_order_cap: int = 40) -> list[str]:
    with Pool(jobs) as p:
        return p.map(_bank_entry, [(split, i, max_order_cap) for i in range(n)], chunksize=4)


@dataclass
class MixConfig:
    clip_s: float = 4.0
    snr_range: tuple[float, float] = (-5.0, 15.0)
    p_absent: float = 0.08
    p_no_voice: float = 0.15
    p_no_doa: float = 0.15
    doa_noise_deg: float = 5.0
    enroll_from_table: bool = True


class OnlineMixDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, size: int, cfg: MixConfig, seed: int = 0, bank_size: int | None = None):
        self.split, self.size, self.cfg, self.seed = split, size, cfg, seed
        files = sorted(bank_dir(split).glob("*.npz"))
        if bank_size:
            files = files[:bank_size]
        if not files:
            raise FileNotFoundError(f"empty RIR bank for {split}; run bench/build_rir_bank")
        self.files = files
        self.epoch = 0
        self._emb: dict[str, np.ndarray] | None = None
        self._spk: dict[str, list[corpus.Utterance]] | None = None

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return self.size

    def _lazy_light(self) -> None:
        if self._spk is None:
            self._spk = corpus.librispeech_index(self.split)
            self._speakers = sorted(self._spk)
            self._esc = corpus.esc50_index(self.split)

    def _lazy(self) -> None:
        self._lazy_light()
        if self._emb is None:
            from clarity_tse.embed import speaker

            self._emb = speaker.utterance_embedding_table(self.split, device="cpu")

    def _speech(self, spk: str, n: int, rng: np.random.Generator, exclude: str | None = None) -> tuple[np.ndarray, str]:
        assert self._spk is not None
        utts = [u for u in self._spk[spk] if u.path != exclude]
        u = utts[int(rng.integers(len(utts)))]
        x = corpus.load_utt(u)
        if len(x) > n:
            off = int(rng.integers(0, len(x) - n))
            x = x[off : off + n]
        out = np.zeros(n, dtype=np.float32)
        start = int(rng.integers(0, max(1, n - len(x) + 1)))
        out[start : start + len(x)] = x
        return out / (np.sqrt(np.mean(out**2)) + 1e-9) * 0.05, u.path

    def _noise(self, role_src: dict[str, Any], n: int, rng: np.random.Generator, kind: str) -> np.ndarray:
        groups = scene.SCENE_TYPES[kind]["noise"]
        cats = [c for g in groups for c in corpus.NOISE_CATEGORIES[g] if c in self._esc]
        if role_src.get("kind") == "synthetic" or not cats or rng.random() < 0.15:
            x = corpus.synthetic_noise(str(rng.choice(["music", "pink", "clatter"])), n, rng)
        else:
            c = cats[int(rng.integers(len(cats)))]
            f = self._esc[c][int(rng.integers(len(self._esc[c])))]
            y = corpus.read_audio(corpus.EXTERNAL / f)
            y = np.tile(y, int(np.ceil(n / len(y))) + 1)
            off = int(rng.integers(0, len(y) - n))
            x = y[off : off + n]
        return (x / (np.sqrt(np.mean(x**2)) + 1e-9) * 0.05).astype(np.float32)

    def render_raw(self, idx: int) -> dict[str, Any]:
        """Render one mixture without any conditioning augmentation."""
        self._lazy_light()
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        return self._render(rng)

    def _render(self, rng: np.random.Generator) -> dict[str, Any]:
        cfg = self.cfg
        n = int(cfg.clip_s * SR)
        f = self.files[int(rng.integers(len(self.files)))]
        with np.load(f) as z:
            rirs = z["rirs"].astype(np.float32)
            rel = z["rel_az"]
            spec = json.loads(str(z["spec"]))
        roles = [s["role"] for s in spec["sources"]]
        kind = spec["kind"]
        n_spk_needed = sum(1 for r in roles if r in ("target", "interferer")) + 2 * roles.count("babble")
        spks = rng.choice(len(self._speakers), size=n_spk_needed + 1, replace=False)
        spk_iter = iter(self._speakers[int(i)] for i in spks)
        target_spk = next(spk_iter)
        imgs = []
        target_img = np.zeros((2, n))
        target_early = np.zeros(n)
        target_path = None
        for si, role in enumerate(roles):
            if role == "target":
                sig, target_path = self._speech(target_spk, n, rng)
            elif role == "interferer":
                sig = self._speech(next(spk_iter), n, rng)[0] * 10 ** (rng.uniform(-4, 2) / 20)
            elif role == "babble":
                sig = (self._speech(next(spk_iter), n, rng)[0] + self._speech(next(spk_iter), n, rng)[0]) / np.sqrt(2)
                sig = sig * 10 ** (rng.uniform(-10, -4) / 20)
            else:
                sig = self._noise(spec["sources"][si].get("noise", {}), n, rng, kind) * 10 ** (rng.uniform(-6, 0) / 20)
            img = np.stack([fftconvolve(sig, rirs[si, ch])[:n] for ch in range(2)])
            if role == "target":
                h = rirs[si, 0]
                d0 = int(np.argmax(np.abs(h)))
                he = h[: d0 + int(scene.EARLY_MS * 1e-3 * SR)].copy()
                he[-80:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, 80))
                target_early = fftconvolve(sig, he)[:n]
                target_img = img
            else:
                imgs.append(img)
        interf = np.sum(imgs, axis=0) if imgs else np.zeros((2, n))
        g_t = 10 ** (scene.TARGET_DBFS / 20) / (np.sqrt(np.mean(target_img[0] ** 2)) + 1e-9)
        target_img, target_early = target_img * g_t, target_early * g_t
        snr = rng.uniform(*cfg.snr_range)
        g_i = np.sqrt(np.mean(target_img[0] ** 2)) / (np.sqrt(np.mean(interf[0] ** 2)) + 1e-9) * 10 ** (-snr / 20)
        interf = interf * g_i
        mix = target_img + interf + rng.standard_normal((2, n)) * 10 ** (-75 / 20)
        t_idx = roles.index("target")
        return dict(mix=mix, ref=target_early, theta=float(rel[t_idx]), target_spk=target_spk, target_utt=target_path,
                    speakers=[self._speakers[int(i)] for i in spks[:n_spk_needed + 1]])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._lazy()
        assert self._emb is not None and self._spk is not None
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        cfg = self.cfg
        d = self._render(rng)
        mix, ref = d["mix"], d["ref"]
        absent = rng.random() < cfg.p_absent
        theta = float(d["theta"] + rng.normal(0, cfg.doa_noise_deg))
        if absent:  # enrolled person not in the scene
            others = [s for s in self._speakers if s not in set(d["speakers"])]
            spk = others[int(rng.integers(len(others)))]
            cands = [u.path for u in self._spk[spk]]
            ref, theta = ref * 0, float("nan")
        else:
            cands = [u.path for u in self._spk[d["target_spk"]] if u.path != d["target_utt"]]
        emb = self._emb[cands[int(rng.integers(len(cands)))]].copy()
        if not absent and rng.random() < cfg.p_no_voice:
            emb[:] = 0
        if not absent and rng.random() < cfg.p_no_doa:
            theta = float("nan")
        lvl = 10 ** (rng.uniform(-10, 10) / 20)
        return dict(mix=torch.tensor(mix * lvl, dtype=torch.float32), ref=torch.tensor(ref * lvl, dtype=torch.float32),
                    emb=torch.tensor(emb, dtype=torch.float32), theta=torch.tensor(theta, dtype=torch.float32),
                    absent=torch.tensor(float(absent)))


# ---------------------------------------------------------------------------------------------
# Pre-rendered pool (DECISIONS D11): rendering on the fly starves the GPU on a contended host,
# so mixtures are rendered once into fp16 shards and re-used by all runs. Enrollment embeddings,
# "enrolled-but-absent" examples, voice/direction dropout and DOA noise are applied at load time.
# ---------------------------------------------------------------------------------------------
POOL_SHARD = 500


def pool_dir(split: str) -> Path:
    return corpus.CACHE / "train_pool" / split


def _render_shard(args: tuple[str, int, float]) -> str:
    split, shard, clip_s = args
    out = pool_dir(split) / f"shard_{shard:04d}"
    if (out / "done").exists():
        return str(out)
    ds = OnlineMixDataset(split, POOL_SHARD, MixConfig(clip_s=clip_s, p_absent=0.0, p_no_voice=0.0, p_no_doa=0.0,
                                                       doa_noise_deg=0.0), seed=shard)
    ds._lazy_light()
    n = int(clip_s * SR)
    mix = np.zeros((POOL_SHARD, 2, n), np.float16)
    ref = np.zeros((POOL_SHARD, n), np.float16)
    theta = np.zeros(POOL_SHARD, np.float32)
    tspk, tutt, allspk = [], [], []
    for i in range(POOL_SHARD):
        d = ds.render_raw(i)
        sc = 1.0 / max(1.0, float(np.abs(d["mix"]).max()) / 60000.0)
        mix[i], ref[i], theta[i] = d["mix"] * sc, d["ref"] * sc, d["theta"]
        tspk.append(d["target_spk"])
        tutt.append(d["target_utt"])
        allspk.append(",".join(d["speakers"]))
    out.mkdir(parents=True, exist_ok=True)  # separate .npy files so workers can memory-map
    np.save(out / "mix.npy", mix)
    np.save(out / "ref.npy", ref)
    np.savez(out / "meta.npz", theta=theta, target_spk=np.array(tspk), target_utt=np.array(tutt),
             speakers=np.array(allspk))
    (out / "done").touch()
    return str(out)


def build_pool(split: str, n_shards: int, jobs: int, clip_s: float = 4.0) -> list[str]:
    with Pool(jobs) as p:
        return p.map(_render_shard, [(split, s, clip_s) for s in range(n_shards)], chunksize=1)


class PoolDataset(torch.utils.data.Dataset):
    """Loads pre-rendered mixtures and applies conditioning augmentations."""

    def __init__(self, split: str, cfg: MixConfig, seed: int = 0, n_shards: int | None = None, crop_s: float | None = None):
        files = sorted(d for d in pool_dir(split).glob("shard_*") if (d / "done").exists())
        if n_shards:
            files = files[:n_shards]
        if not files:
            raise FileNotFoundError(f"no training pool for {split}; run `python -m bench.build_train_pool`")
        self.files, self.cfg, self.seed, self.split = files, cfg, seed, split
        self.crop = int(crop_s * SR) if crop_s else None
        self.epoch = 0
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        self._emb: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.files) * POOL_SHARD

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def _shard(self, k: int) -> dict[str, np.ndarray]:
        if k not in self._cache:
            d = self.files[k]
            with np.load(d / "meta.npz") as z:
                entry = {key: z[key] for key in z.files}
            entry["mix"] = np.load(d / "mix.npy", mmap_mode="r")
            entry["ref"] = np.load(d / "ref.npy", mmap_mode="r")
            self._cache[k] = entry
        return self._cache[k]

    def _lazy(self) -> None:
        if self._emb is None:
            from clarity_tse.embed import speaker

            self._emb = speaker.utterance_embedding_table(self.split, device="cpu")
            idx = corpus.librispeech_index(self.split)
            self._utts = {s: [u.path for u in v if u.path in self._emb] for s, v in idx.items()}
            self._speakers = sorted(self._utts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._lazy()
        assert self._emb is not None
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        z = self._shard(idx // POOL_SHARD)
        j = idx % POOL_SHARD
        mix = z["mix"][j].astype(np.float32)
        ref = z["ref"][j].astype(np.float32)
        if self.crop and self.crop < mix.shape[-1]:
            o = int(rng.integers(0, mix.shape[-1] - self.crop))
            mix, ref = mix[:, o : o + self.crop], ref[o : o + self.crop]
        lvl = 10 ** (rng.uniform(-10, 10) / 20)
        mix, ref = mix * lvl, ref * lvl
        tspk, tutt = str(z["target_spk"][j]), str(z["target_utt"][j])
        theta = float(z["theta"][j] + rng.normal(0, self.cfg.doa_noise_deg))
        absent = rng.random() < self.cfg.p_absent
        if absent:
            # enrolled person is not in the scene: embedding of a speaker absent from the mixture
            present = set(str(z["speakers"][j]).split(","))
            while True:
                other = self._speakers[int(rng.integers(len(self._speakers)))]
                if other not in present:
                    break
            cands = self._utts[other]
            ref = ref * 0
            theta = float("nan")
        else:
            cands = [u for u in self._utts[tspk] if u != tutt]
        emb = self._emb[cands[int(rng.integers(len(cands)))]].copy()
        if not absent and rng.random() < self.cfg.p_no_voice:
            emb[:] = 0
        if not absent and rng.random() < self.cfg.p_no_doa:
            theta = float("nan")
        return dict(mix=torch.tensor(mix), ref=torch.tensor(ref), emb=torch.tensor(emb, dtype=torch.float32),
                    theta=torch.tensor(theta, dtype=torch.float32), absent=torch.tensor(float(absent)))

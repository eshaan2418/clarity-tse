"""Speaker embeddings for enrollment and voice similarity.

Uses the pretrained SpeechBrain ECAPA-TDNN (VoxCeleb, Apache-2.0) as a frozen embedder
(DECISIONS D10). On CI/fixtures, or when ``CLARITY_EMBEDDER=mfcc``, falls back to a tiny
deterministic MFCC-statistics embedder so the pipeline runs without downloads.
"""

from __future__ import annotations

import functools
import os
import numpy as np
import torch

from clarity_tse.synth import corpus

EMB_DIM = 192


def _which() -> str:
    return os.environ.get("CLARITY_EMBEDDER", "ecapa")


@functools.lru_cache(maxsize=2)
def _ecapa(device: str = "cpu"):
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                          savedir=str(corpus.CACHE / "ecapa"), run_opts={"device": device})


def _mfcc_embed(x: np.ndarray) -> np.ndarray:
    import torchaudio

    m = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=48, melkwargs=dict(n_fft=512, hop_length=160, n_mels=64))
    c = m(torch.tensor(x, dtype=torch.float32)[None])[0]  # (48, T)
    e = torch.cat([c.mean(1), c.std(1), c[:, 1:].sub(c[:, :-1]).std(1)])[:EMB_DIM]
    e = torch.nn.functional.pad(e, (0, EMB_DIM - len(e)))
    return (e / (e.norm() + 1e-8)).numpy()


def embed(x: np.ndarray, device: str = "cpu") -> np.ndarray:
    """(T,) 16 kHz -> unit-norm (192,) embedding."""
    if _which() == "mfcc":
        return _mfcc_embed(x)
    with torch.no_grad():
        e = _ecapa(device).encode_batch(torch.tensor(x, dtype=torch.float32)[None].to(device))[0, 0].cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-8)


def embed_batch(xs: list[np.ndarray], device: str = "cpu") -> np.ndarray:
    if _which() == "mfcc":
        return np.stack([_mfcc_embed(x) for x in xs])
    lens = np.array([len(x) for x in xs])
    T = lens.max()
    batch = np.zeros((len(xs), T), dtype=np.float32)
    for i, x in enumerate(xs):
        batch[i, : len(x)] = x
    with torch.no_grad():
        e = _ecapa(device).encode_batch(torch.tensor(batch).to(device), torch.tensor(lens / T, dtype=torch.float32).to(device))
    e = e[:, 0].cpu().numpy()
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def _load_trunc(u: corpus.Utterance) -> np.ndarray:
    return corpus.load_utt(u)[: 16000 * 10]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


MAX_PER_SPEAKER = 16


def utterance_embedding_table(split: str, device: str = "mps") -> dict[str, np.ndarray]:
    """Precompute (and cache) embeddings for up to MAX_PER_SPEAKER of each speaker's longest
    utterances (truncated to 10 s, matching test-time enrollment length): path -> (192,)."""
    tag = _which()
    path = corpus.CACHE / f"utt_emb_{tag}_{split}_{_ext_tag()}_k{MAX_PER_SPEAKER}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return dict(zip(z["paths"].tolist(), z["emb"], strict=True))
    idx = corpus.librispeech_index(split)
    utts = [u for v in idx.values() for u in sorted(v, key=lambda u: (-u.n_samples, u.path))[:MAX_PER_SPEAKER]]
    utts.sort(key=lambda u: (u.n_samples, u.path))
    embs = []
    bs = 32
    from multiprocessing import get_context

    with get_context("spawn").Pool(8) as pool:
        for i in range(0, len(utts), bs):
            chunk = utts[i : i + bs]
            audio = pool.map(_load_trunc, chunk)
            embs.append(embed_batch(audio, device=device))
            if (i // bs) % 50 == 0:
                print(f"[emb {split}] {i}/{len(utts)}", flush=True)
    E = np.concatenate(embs).astype(np.float32)
    paths = np.array([u.path for u in utts])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, paths=paths, emb=E)
    return dict(zip(paths.tolist(), E, strict=True))


def _ext_tag() -> str:
    import zlib

    return f"{zlib.crc32(str(corpus.EXTERNAL).encode()):08x}"

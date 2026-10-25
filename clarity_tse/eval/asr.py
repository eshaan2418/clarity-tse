"""Offline ASR word error rate as an intelligibility proxy (M9, and validation on simulated items).

Uses torchaudio's WAV2VEC2_ASR_BASE_960H (fine-tuned on LibriSpeech 960 h *train* subsets; the
test-clean speakers used for evaluation are not in it) with greedy CTC decoding.
"""

from __future__ import annotations

import functools

import numpy as np
import torch


@functools.lru_cache(maxsize=1)
def _model():
    import torchaudio

    b = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    return b.get_model().eval(), b.get_labels(), b.sample_rate


def transcribe(x: np.ndarray, sr: int = 16000) -> str:
    model, labels, msr = _model()
    assert sr == msr
    x = x / (np.abs(x).max() + 1e-9) * 0.5  # level-normalise: ASR input only, not what the listener hears
    with torch.inference_mode():
        em, _ = model(torch.tensor(x, dtype=torch.float32)[None])
    idx = em[0].argmax(-1).tolist()
    out, prev = [], None
    for i in idx:
        if i != prev and labels[i] != "-":
            out.append(labels[i])
        prev = i
    return "".join(out).replace("|", " ").strip()


def _norm(s: str) -> list[str]:
    return "".join(c if c.isalpha() or c in " '" else " " for c in s.upper()).replace("'", "").split()


def wer(hyp: str, ref: str) -> tuple[float, int]:
    """(word error rate, #reference words) by Levenshtein distance on normalised words."""
    h, r = _norm(hyp), _norm(ref)
    d = np.arange(len(h) + 1)
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return float(d[len(h)] / max(1, len(r))), len(r)

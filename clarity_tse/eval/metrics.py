"""Metric suite. Thin, tested wrappers around reference implementations.

- SI-SDR / SI-SDRi: own implementation, validated against torchmetrics in tests.
- PESQ (wide-band, ITU-T P.862.2): ``pesq`` package (ITU reference C code).
- STOI / ESTOI: ``pystoi`` (Taal et al. 2011; Jensen & Taal 2016).
- HASPI v2 / HASQI v2: ``pyclarity`` port of Kates & Arehart's MATLAB code, with audiogram.
- DNSMOS P.835 (SIG/BAK/OVRL): Microsoft DNS-Challenge ``sig_bak_ovr.onnx`` + official polyfit.
"""

from __future__ import annotations

import functools
import warnings
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from clarity_tse.synth.scene import SPL_AT_0DBFS

AUDIOGRAM_FREQS = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])
DNSMOS_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "dnsmos" / "sig_bak_ovr.onnx"


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    s = alpha * ref
    e = est - s
    return float(10 * np.log10((np.dot(s, s) + eps) / (np.dot(e, e) + eps)))


def snr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    e = est - ref
    return float(10 * np.log10((np.dot(ref, ref) + eps) / (np.dot(e, e) + eps)))


def pesq_wb(est: np.ndarray, ref: np.ndarray, sr: int = 16000) -> float:
    from pesq import PesqError, pesq

    try:
        return float(pesq(sr, ref.astype(np.float64), est.astype(np.float64), "wb"))
    except PesqError:
        return float("nan")


def stoi(est: np.ndarray, ref: np.ndarray, sr: int = 16000, extended: bool = False) -> float:
    from pystoi import stoi as _stoi

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # spurious matmul FP warnings from Accelerate BLAS
        return float(_stoi(ref, est, sr, extended=extended))


def _audiogram(levels: np.ndarray | None):
    from clarity.utils.audiogram import Audiogram

    if levels is None:
        levels = np.zeros(len(AUDIOGRAM_FREQS))
    return Audiogram(levels=np.asarray(levels, dtype=float), frequencies=AUDIOGRAM_FREQS)


def haspi(est: np.ndarray, ref: np.ndarray, sr: int = 16000, audiogram: np.ndarray | None = None) -> float:
    """HASPI v2 with signals in our dBFS calibration (RMS 1 <-> SPL_AT_0DBFS dB SPL).

    ``ref`` must be the unamplified clean reference; ``est`` is what reaches the ear (including
    any hearing-aid gain).
    """
    from clarity.evaluator.haspi import haspi_v2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score, _ = haspi_v2(ref, sr, est, sr, _audiogram(audiogram), level1=SPL_AT_0DBFS)
    return float(score)


def hasqi(est: np.ndarray, ref: np.ndarray, sr: int = 16000, audiogram: np.ndarray | None = None) -> float:
    from clarity.evaluator.hasqi import hasqi_v2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score, _, _, _ = hasqi_v2(ref, sr, est, sr, _audiogram(audiogram), equalisation=1,
                                  level1=SPL_AT_0DBFS)
    return float(score)


@functools.lru_cache(maxsize=1)
def _dnsmos_session():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(DNSMOS_PATH), so, providers=["CPUExecutionProvider"])


def dnsmos(x: np.ndarray, sr: int = 16000) -> dict[str, float]:
    """Non-intrusive DNSMOS P.835 (non-personalised polyfit), mirroring dnsmos_local.py."""
    if sr != 16000:
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g)
    input_len = 9.01
    n = int(input_len * 16000)
    audio = np.asarray(x, dtype=np.float32)
    while len(audio) < n:
        audio = np.append(audio, audio)
    num_hops = int(np.floor(len(audio) / 16000) - input_len) + 1
    sess = _dnsmos_session()
    sig, bak, ovr = [], [], []
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    for i in range(max(1, num_hops)):
        seg = audio[i * 16000 : i * 16000 + n]
        if len(seg) < n:
            continue
        s, b, o = sess.run(None, {"input_1": seg[None, :]})[0][0]
        sig.append(p_sig(s))
        bak.append(p_bak(b))
        ovr.append(p_ovr(o))
    return dict(SIG=float(np.mean(sig)), BAK=float(np.mean(bak)), OVRL=float(np.mean(ovr)))


def intrusive_suite(est: np.ndarray, ref: np.ndarray, mix_ref: np.ndarray | None = None, sr: int = 16000,
                    audiogram: np.ndarray | None = None, fast: bool = False) -> dict[str, float]:
    """Compute intrusive metrics; ``mix_ref`` (unprocessed reference-mic mixture) enables SI-SDRi."""
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    out = dict(si_sdr=si_sdr(est, ref), estoi=stoi(est, ref, sr, extended=True), stoi=stoi(est, ref, sr))
    if mix_ref is not None:
        out["si_sdri"] = out["si_sdr"] - si_sdr(mix_ref[:n], ref)
    if not fast:
        out["pesq"] = pesq_wb(est, ref, sr)
        out["haspi"] = haspi(est, ref, sr, audiogram)
    return out

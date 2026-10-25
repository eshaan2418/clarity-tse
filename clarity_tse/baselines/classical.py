"""Baselines: unprocessed, RNNoise (single-channel NS), delay-and-sum, MVDR/MPDR, NAL-R.

All take the binaural mixture (2, T) at 16 kHz and return a mono estimate time-aligned with the
reference mic. Algorithmic latency of each baseline is reported in ``LATENCY_MS``; any constant
processing delay is compensated before metrics (and reported) so that comparisons are about
quality, while latency is accounted for separately.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from clarity_tse.dsp import array
from clarity_tse.dsp.stft import STFT, LatencyConfig

REPO = Path(__file__).resolve().parents[2]
BF_CFG = LatencyConfig(n_fft=512, hop=80)  # 10 ms algorithmic latency, same STFT family as Clarity
LATENCY_MS = {"unprocessed": 0.0, "rnnoise": 20.0, "ds": BF_CFG.algorithmic_latency_ms,
              "mpdr": BF_CFG.algorithmic_latency_ms, "mvdr_oracle": BF_CFG.algorithmic_latency_ms}


def unprocessed(mix: np.ndarray, **_) -> np.ndarray:
    return mix[0].copy()


def _stft():
    return STFT(BF_CFG)


def _theta_frames(theta: np.ndarray | float, K: int) -> np.ndarray:
    th = np.asarray(theta, dtype=float)
    if th.ndim == 0:
        return np.full(K, float(th))
    idx = np.minimum((np.arange(K) + 1) * BF_CFG.hop - 1, len(th) - 1)
    return th[idx]


def delay_and_sum(mix: np.ndarray, theta: np.ndarray | float, **_) -> np.ndarray:
    st = _stft()
    X = st.analysis(torch.tensor(mix, dtype=torch.float64)).numpy()
    f = np.fft.rfftfreq(BF_CFG.n_fft, 1 / BF_CFG.sr)
    Y = array.delay_and_sum(X, _theta_frames(theta, X.shape[-1]), f)
    return st.synthesis(torch.tensor(Y), mix.shape[1]).numpy().astype(np.float32)


def mpdr(mix: np.ndarray, theta: np.ndarray | float, noise: np.ndarray | None = None, **_) -> np.ndarray:
    st = _stft()
    X = st.analysis(torch.tensor(mix, dtype=torch.float64)).numpy()
    N = st.analysis(torch.tensor(noise, dtype=torch.float64)).numpy() if noise is not None else None
    f = np.fft.rfftfreq(BF_CFG.n_fft, 1 / BF_CFG.sr)
    Y = array.mvdr(X, _theta_frames(theta, X.shape[-1]), f, noise=N)
    return st.synthesis(torch.tensor(Y), mix.shape[1]).numpy().astype(np.float32)


def mvdr_oracle(mix: np.ndarray, theta: np.ndarray | float, interference: np.ndarray, **_) -> np.ndarray:
    return mpdr(mix, theta, noise=interference)


@functools.lru_cache(maxsize=1)
def _rnnoise_lib():
    ext = "dylib" if (REPO / "data/external/src/rnnoise/librnnoise.dylib").exists() else "so"
    lib = ctypes.CDLL(str(REPO / f"data/external/src/rnnoise/librnnoise.{ext}"))
    lib.rnnoise_create.restype = ctypes.c_void_p
    lib.rnnoise_create.argtypes = [ctypes.c_void_p]
    lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    lib.rnnoise_process_frame.restype = ctypes.c_float
    lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    return lib


RNNOISE_DELAY_48K = 960  # measured: output lags input by 20 ms (bench: tests/test_baselines.py)


def rnnoise(mix: np.ndarray, **_) -> np.ndarray:
    lib = _rnnoise_lib()
    x = resample_poly(mix[0].astype(np.float64), 3, 1) * 32768.0
    n = len(x)
    pad = (-n) % 480 + RNNOISE_DELAY_48K + 480
    x = np.concatenate([x, np.zeros(pad)]).astype(np.float32)
    y = np.zeros_like(x)
    st = lib.rnnoise_create(None)
    buf_in = (ctypes.c_float * 480)()
    buf_out = (ctypes.c_float * 480)()
    for i in range(0, len(x), 480):
        ctypes.memmove(buf_in, x[i : i + 480].ctypes.data, 480 * 4)
        lib.rnnoise_process_frame(st, buf_out, buf_in)
        y[i : i + 480] = np.frombuffer(buf_out, dtype=np.float32)
    lib.rnnoise_destroy(st)
    y = y[RNNOISE_DELAY_48K : RNNOISE_DELAY_48K + n] / 32768.0
    return resample_poly(y, 1, 3).astype(np.float32)[: mix.shape[1]]


BASELINES = {"unprocessed": unprocessed, "rnnoise": rnnoise, "ds": delay_and_sum, "mpdr": mpdr,
             "mvdr_oracle": mvdr_oracle}

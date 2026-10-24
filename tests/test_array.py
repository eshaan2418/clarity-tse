import numpy as np
import torch

from clarity_tse.dsp.array import delay_and_sum, gcc_phat_doa, mvdr, srp_phat, steering_vector
from clarity_tse.dsp.stft import STFT, LatencyConfig


def _plane_wave(theta, n=16000, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n)
    S = np.fft.rfft(s)
    f = np.fft.rfftfreq(n, 1 / sr)
    a = steering_vector(theta, f)
    return np.stack([np.fft.irfft(S * a[c], n=n) for c in range(2)]), s


def test_gcc_phat_recovers_lateral_angle():
    for theta in [-60, -20, 0, 30, 70]:
        x, _ = _plane_wave(theta)
        assert abs(gcc_phat_doa(x[0], x[1], 16000) - theta) < 4


def test_srp_peak_and_ds_distortionless():
    theta = 35.0
    x, _ = _plane_wave(theta)
    st = STFT(LatencyConfig(512, 128))
    X = st.analysis(torch.tensor(x)).numpy()
    f = np.fft.rfftfreq(512, 1 / 16000)
    grid = np.arange(-90, 91, 1.0)
    p = srp_phat(X, f, grid).mean(axis=1)
    assert abs(grid[np.argmax(p)] - theta) <= 3
    Y = delay_and_sum(X, theta, f)
    y = st.synthesis(torch.tensor(Y), x.shape[1]).numpy()
    # fractional inter-mic delays are only approximately frame-separable: allow -30 dB error
    assert np.mean((y[600:-600] - x[0, 600:-600]) ** 2) / np.mean(x[0] ** 2) < 1e-3
    Ym = mvdr(X, theta, f)
    ym = st.synthesis(torch.tensor(Ym), x.shape[1]).numpy()
    err = np.mean((ym[4000:] - x[0, 4000:]) ** 2) / np.mean(x[0] ** 2)
    assert err < 1e-2

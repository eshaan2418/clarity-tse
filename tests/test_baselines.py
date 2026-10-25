import numpy as np
import pytest

from clarity_tse.baselines import classical as C


def _plane(theta, n=32000, seed=0):
    from clarity_tse.dsp.array import steering_vector
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n) * 0.05
    f = np.fft.rfftfreq(n, 1 / 16000)
    a = steering_vector(theta, f)
    return np.stack([np.fft.irfft(np.fft.rfft(s) * a[c], n=n) for c in range(2)])


def test_beamformers_pass_target_direction():
    x = _plane(30.0)
    for name in ("ds", "mpdr"):
        y = C.BASELINES[name](x, theta=30.0)
        err = np.mean((y[4000:] - x[0, 4000:]) ** 2) / np.mean(x[0] ** 2)
        assert err < 0.05, name


@pytest.mark.skipif(not (C.REPO / "data/external/src/rnnoise").exists(), reason="rnnoise not built")
def test_rnnoise_alignment():
    import sys

    from clarity_tse.synth.corpus import synthetic_noise  # noqa: F401
    sys.path.insert(0, str(C.REPO / "scripts"))
    from make_fixtures import synth_speaker_utt
    x = synth_speaker_utt(np.random.default_rng(0), 120, [600, 1400, 2500], 3.0)
    y = C.rnnoise(np.stack([x, x]))
    lags = range(-50, 50)
    best = max(lags, key=lambda D: np.dot(np.roll(y, -D)[4000:-4000], x[4000:-4000]))
    assert abs(best) <= 2

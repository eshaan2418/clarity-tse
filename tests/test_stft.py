import numpy as np
import pytest
import torch

from clarity_tse.dsp.stft import STFT, LatencyConfig, StreamingSTFT, asymmetric_windows


@pytest.mark.parametrize("n_fft,hop", [(512, 16), (512, 40), (512, 80), (512, 128), (512, 256), (640, 320)])
def test_perfect_reconstruction_and_streaming_equivalence(n_fft, hop):
    cfg = LatencyConfig(n_fft=n_fft, hop=hop)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(hop * 50).astype(np.float32)
    st = STFT(cfg)
    X = st.analysis(torch.tensor(x))
    y = st.synthesis(X, len(x)).numpy()
    assert np.max(np.abs(y - x)) < 1e-4
    # streaming: output of hop k is samples [(k-1)H, kH)
    ss = StreamingSTFT(cfg)
    outs = []
    for k in range(len(x) // hop):
        spec = ss.push(x[None, k * hop : (k + 1) * hop])[0]
        np.testing.assert_allclose(spec, X[:, k].numpy(), atol=1e-3)
        outs.append(ss.pull(spec))
    ys = np.concatenate(outs)[hop:]
    assert np.max(np.abs(ys - x[: len(ys)])) < 1e-4


def test_window_product_is_short_hann():
    wa, ws = asymmetric_windows(512, 40)
    p = wa * ws
    assert np.all(p[: 512 - 80] == 0)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(80) / 80)
    np.testing.assert_allclose(p[-80:], hann, atol=1e-9)


def test_latency_formula():
    assert LatencyConfig.for_budget_ms(2).algorithmic_latency_ms == pytest.approx(2.0)
    assert LatencyConfig.for_budget_ms(10).algorithmic_latency_ms == pytest.approx(10.0)
    assert LatencyConfig(n_fft=512, hop=80, lookahead=1).algorithmic_latency_ms == pytest.approx(15.0)
    assert LatencyConfig.for_budget_ms(40).n_fft == 640

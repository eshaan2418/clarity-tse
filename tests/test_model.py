import numpy as np
import pytest
import torch

from clarity_tse.models.tse import StreamingTSE, TSEConfig, TSENet


def _rand_net(**kw):
    torch.manual_seed(0)
    net = TSENet(TSEConfig(**kw)).eval()
    with torch.no_grad():  # FiLM is zero-initialised; perturb so conditioning matters in the test
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return net


@pytest.mark.parametrize("hop,lookahead,channels", [(16, 0, 2), (80, 0, 2), (80, 1, 2), (40, 0, 1)])
def test_streaming_matches_batch(hop, lookahead, channels):
    net = _rand_net(hop=hop, lookahead=lookahead, channels=channels, blocks=3, repeats=2,
                    bottleneck=32, hidden=48)
    T = hop * 60
    mix = torch.randn(1, 2, T) * 0.1
    emb = torch.nn.functional.normalize(torch.randn(1, 192), dim=1)
    K = net.frames_for(T)
    theta = torch.full((1, K), 20.0)
    with torch.no_grad():
        y_batch = net(mix, emb, theta if channels == 2 else None)[0].numpy()
        s = StreamingTSE(net)
        states = s.init_states()
        film = net.condition(emb, 1, mix.device)
        outs = []
        for k in range(T // hop):
            res = s(mix[:, :channels, k * hop : (k + 1) * hop], torch.tensor([20.0]), film, *states)
            outs.append(res[0][0].numpy())
            states = list(res[1:])
    y_stream = np.concatenate(outs)
    delay = hop * (1 + lookahead)
    n = len(y_stream) - delay
    np.testing.assert_allclose(y_stream[delay:], y_batch[:n], atol=2e-4, rtol=0)


def test_causality():
    net = _rand_net(hop=80, blocks=3, repeats=1, bottleneck=32, hidden=48)
    T = 80 * 40
    a = torch.randn(1, 2, T)
    b = a.clone()
    b[..., 2000:] = torch.randn(1, 2, T - 2000)
    with torch.no_grad():
        ya, yb = net(a)[0], net(b)[0]
    # outputs before (change point - algorithmic latency) must be identical
    cut = 2000 - 2 * 80
    assert torch.allclose(ya[:cut], yb[:cut], atol=1e-5)
    assert not torch.allclose(ya[2000:], yb[2000:])


def test_bounded_weights_attenuate():
    net = _rand_net(hop=80, blocks=2, repeats=1, bottleneck=16, hidden=16)
    x = torch.randn(1, 2, 8000)
    with torch.no_grad():
        y = net(x)[0]
    assert y.square().mean() <= 4 * x[:, 0].square().mean() + 1e-6


def test_nan_theta_equals_no_direction():
    net = _rand_net(hop=80, blocks=2, repeats=1, bottleneck=16, hidden=16)
    x = torch.randn(1, 2, 8000)
    K = net.frames_for(8000)
    with torch.no_grad():
        a = net(x, None, torch.full((1, K), float("nan")))
        b = net(x, None, None)
    assert torch.allclose(a, b, atol=1e-6)


def test_per_frame_embedding_matches_constant():
    net = _rand_net(hop=80, blocks=2, repeats=1, bottleneck=16, hidden=16)
    x = torch.randn(1, 2, 8000)
    K = net.frames_for(8000)
    e = torch.nn.functional.normalize(torch.randn(1, 192), dim=1)
    with torch.no_grad():
        a = net(x, e)
        b = net(x, e[:, None, :].expand(1, K, 192))
    assert torch.allclose(a, b, atol=1e-5)

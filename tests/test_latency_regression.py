"""Latency regression tests: algorithmic latency of every sweep config matches its budget name,
the measured end-to-end delay of the STFT path never exceeds the declared algorithmic latency,
and per-hop compute of the operating-point architecture stays below real time."""
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from clarity_tse.dsp.stft import STFT, LatencyConfig
from clarity_tse.eval.latency import measure_per_hop
from clarity_tse.models.tse import EMB_DIM, StreamingTSE, TSEConfig, TSENet

GEN = Path(__file__).resolve().parents[1] / "configs" / "m3" / "gen"


@pytest.mark.parametrize("path", sorted(GEN.glob("b*.yaml")), ids=lambda p: p.stem)
def test_config_budget_matches_name(path):
    cfg = yaml.safe_load(path.read_text())
    budget_ms = float(re.match(r"b(\d+)", cfg["name"]).group(1))
    m = TSEConfig(**cfg["model"])
    assert m.causal
    assert m.latency.algorithmic_latency_ms == pytest.approx(budget_ms, abs=0.01)


@pytest.mark.parametrize("hop", [16, 40, 80, 128, 160])
def test_measured_delay_within_algorithmic(hop):
    cfg = LatencyConfig(512, hop)
    st = STFT(cfg)
    T = hop * 64
    x = torch.zeros(1, T)
    x[0, T // 2] = 1.0
    # identity processing: the STFT path itself is time-aligned (zero added delay in batch mode);
    # the streaming wrapper adds exactly one hop + lookahead (tests/test_model.py). Both <= 2*hop.
    y = st.synthesis(st.analysis(x), T)[0].numpy()
    assert int(np.argmax(np.abs(y))) - T // 2 <= cfg.algorithmic_latency_samples
    np.testing.assert_allclose(y, x[0].numpy(), atol=1e-5)


def test_operating_point_compute_is_real_time(tmp_path):
    """ONNX Runtime (the engines' inference path), 1 thread, operating-point architecture."""
    import onnxruntime as ort

    from bench.export_onnx import export

    net = TSENet(TSEConfig(hop=80)).eval()  # configs/m3/base.yaml architecture
    export(net, tmp_path, "op")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(tmp_path / "op.onnx"), so, providers=["CPUExecutionProvider"])
    feed = {"hop": np.zeros((1, 2, 80), np.float32), "theta": np.zeros(1, np.float32),
            "film": net.condition(torch.zeros(1, EMB_DIM), 1, torch.device("cpu")).detach().numpy()}
    feed.update({f"state_{i}": z.numpy() for i, z in enumerate(StreamingTSE(net).init_states())})

    def step():
        res = sess.run(None, feed)
        feed.update({f"state_{i}": v for i, v in enumerate(res[1:])})

    stats = measure_per_hop(step, hop_ms=1000 * 80 / 16000, n_hops=300, warmup=30)
    # median, not mean: the CI/dev host is shared (BLOCKERS B8); tails are measured in results/m6
    assert stats.p50_ms < stats.hop_ms, stats

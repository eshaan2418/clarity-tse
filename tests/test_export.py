import torch

from bench.export_onnx import export, verify
from clarity_tse.models.tse import TSEConfig, TSENet


def test_onnx_export_matches_batch(tmp_path):
    for la in (0, 1):
        torch.manual_seed(0)
        net = TSENet(TSEConfig(hop=80, lookahead=la, blocks=3, repeats=1, bottleneck=32, hidden=48)).eval()
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.05 * torch.randn_like(p))
        export(net, tmp_path, f"t{la}")
        assert verify(net, tmp_path / f"t{la}.onnx", n_hops=60) < 5e-4

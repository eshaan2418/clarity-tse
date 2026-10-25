"""Export the frozen SpeechBrain ECAPA-TDNN embedder (features + normalisation + encoder) to ONNX
for on-device enrollment in the Android app, and check it against the PyTorch embedder.

    uv run python -m bench.export_ecapa --out data/cache/export/ecapa.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from clarity_tse.embed import speaker


class MatmulSTFT(torch.nn.Module):
    """Drop-in for speechbrain's STFT (center=True, constant pad, onesided) using real matmuls,
    so the graph exports to ONNX. Output (B, frames, F, 2) like speechbrain's."""

    def __init__(self, sb_stft):
        super().__init__()
        n, self.hop = sb_stft.n_fft, sb_stft.hop_length
        assert sb_stft.win_length == n and sb_stft.center and sb_stft.pad_mode == "constant"
        k = torch.arange(n, dtype=torch.float64)[:, None]
        f = torch.arange(n // 2 + 1, dtype=torch.float64)[None]
        w = sb_stft.window.double()[:, None]
        ang = 2 * torch.pi * k * f / n
        kern = torch.cat([w * torch.cos(ang), -w * torch.sin(ang)], dim=1).T  # (2F, N)
        self.register_buffer("kern", kern.float()[:, None, :])
        self.n, self.F = n, n // 2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x[:, None], (self.n // 2, self.n // 2))
        y = torch.nn.functional.conv1d(x, self.kern, stride=self.hop)  # (B, 2F, K) strided DFT
        return torch.stack([y[:, : self.F], y[:, self.F :]], dim=-1).transpose(1, 2)


class EcapaWrapper(torch.nn.Module):
    def __init__(self, clf):
        super().__init__()
        self.compute_features = clf.mods.compute_features
        self.compute_features.compute_STFT = MatmulSTFT(self.compute_features.compute_STFT)
        self.mean_var_norm = clf.mods.mean_var_norm
        self.embedding_model = clf.mods.embedding_model

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (1, T) -> (1, 192)
        feats = self.compute_features(wav)
        feats = self.mean_var_norm(feats, torch.ones(1))
        return self.embedding_model(feats, torch.ones(1))[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cache/export/ecapa.onnx")
    a = ap.parse_args()
    import copy

    clf = copy.deepcopy(speaker._ecapa("cpu"))  # keep the cached embedder unpatched for the check
    m = EcapaWrapper(clf).eval()
    x = torch.randn(1, 16000 * 15) * 0.05
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(m, (x,), str(out), input_names=["wav"], output_names=["emb"], opset_version=17,
                          dynamic_axes={"wav": {1: "T"}}, dynamo=False)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for n in (16000 * 15, 16000 * 10):
        w = (rng.standard_normal(n) * 0.05).astype(np.float32)
        e_onnx = sess.run(None, {"wav": w[None]})[0][0]
        e_onnx = e_onnx / np.linalg.norm(e_onnx)
        e_pt = speaker.embed(w)
        print(f"T={n}: cosine(onnx, torch) = {float(np.dot(e_onnx, e_pt)):.6f}")


if __name__ == "__main__":
    main()

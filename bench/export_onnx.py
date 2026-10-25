"""Export a trained streaming checkpoint to ONNX (hop-in/hop-out with explicit state) and verify
ONNX Runtime == PyTorch streaming == PyTorch batch on random input.

Also exports a FiLM-conditioning model (embedding -> FiLM params) so engines can enroll once.

    uv run python -m bench.export_onnx --ckpt b10_s0 --out results/m3/export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bench.train import load_model, sha256_file
from clarity_tse.models.tse import EMB_DIM, StreamingTSE, TSENet
from clarity_tse.synth import corpus


class FilmModel(torch.nn.Module):
    def __init__(self, net: TSENet):
        super().__init__()
        self.net = net

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net.condition(emb, 1, emb.device)


def export(net: TSENet, out_dir: Path, name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = StreamingTSE(net).eval()
    cfg = net.cfg
    states = s.init_states()
    film = net.condition(torch.zeros(1, EMB_DIM), 1, torch.device("cpu"))
    hop = torch.zeros(1, cfg.channels, cfg.hop)
    theta = torch.zeros(1)
    state_names = [f"state_{i}" for i in range(len(states))]
    path = out_dir / f"{name}.onnx"
    torch.onnx.export(s, (hop, theta, film, *states), str(path), input_names=["hop", "theta", "film", *state_names],
                      output_names=["out", *[f"new_{n}" for n in state_names]], opset_version=17, dynamo=False)
    fpath = out_dir / f"{name}_film.onnx"
    torch.onnx.export(FilmModel(net).eval(), (torch.zeros(1, EMB_DIM),), str(fpath), input_names=["emb"],
                      output_names=["film"], opset_version=17, dynamo=False)
    meta = dict(name=name, model_cfg=cfg.to_dict(), state_shapes=[list(x.shape) for x in states],
                film_shape=list(film.shape), algorithmic_latency_ms=cfg.latency.algorithmic_latency_ms,
                output_delay_samples=cfg.hop * (1 + cfg.lookahead), onnx_sha256=sha256_file(path),
                n_params=net.n_params(), macs_per_frame=net.macs_per_frame())
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1))
    # flat key=value config for the C++ / Android engines (no JSON dependency)
    lines = [f"hop={cfg.hop}", f"channels={cfg.channels}", f"sr={cfg.sr}", f"lookahead={cfg.lookahead}",
             f"n_states={len(states)}", "film_shape=" + ",".join(str(d) for d in film.shape)]
    lines += [f"state_{i}=" + ",".join(str(d) for d in st.shape) for i, st in enumerate(states)]
    (out_dir / f"{name}.cfg").write_text("\n".join(lines) + "\n")
    return meta


def verify(net: TSENet, path: Path, n_hops: int = 200, seed: int = 0) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    cfg = net.cfg
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((1, cfg.channels, cfg.hop * n_hops)) * 0.05).astype(np.float32)
    emb = torch.nn.functional.normalize(torch.randn(1, EMB_DIM), dim=1)
    film = net.condition(emb, 1, torch.device("cpu")).detach().numpy()
    st = [z.numpy() for z in StreamingTSE(net).init_states()]
    outs = []
    for k in range(n_hops):
        feed = {"hop": x[:, :, k * cfg.hop : (k + 1) * cfg.hop], "theta": np.array([15.0], np.float32), "film": film}
        feed.update({f"state_{i}": v for i, v in enumerate(st)})
        res = sess.run(None, feed)
        outs.append(res[0][0])
        st = res[1:]
    y_onnx = np.concatenate(outs)
    with torch.no_grad():
        mix = torch.tensor(x)
        if cfg.channels == 1:
            mix = torch.cat([mix, mix], 1)
        K = net.frames_for(mix.shape[-1])
        y_b = net(mix, emb, torch.full((1, K), 15.0))[0].numpy()
    d = cfg.hop * (1 + cfg.lookahead)
    return float(np.max(np.abs(y_onnx[d:] - y_b[: len(y_onnx) - d])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", default=str(corpus.CACHE / "export"))
    a = ap.parse_args()
    for c in a.ckpt:
        net = load_model(corpus.CACHE / "ckpt" / f"{c}.pt")
        meta = export(net, Path(a.out), c)
        err = verify(net, Path(a.out) / f"{c}.onnx")
        meta["onnx_vs_batch_max_abs_err"] = err
        (Path(a.out) / f"{c}.json").write_text(json.dumps(meta, indent=1))
        print(c, "exported; max |onnx - batch| =", err)


if __name__ == "__main__":
    main()

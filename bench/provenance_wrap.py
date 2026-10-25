"""Attach provenance (git, hardware, load, checkpoint/ONNX hashes) to a raw JSON written by the
C++ tools, and derive RTF = mean compute / hop duration.

    uv run python -m bench.provenance_wrap raw.json out.json --ckpt data/cache/ckpt/x.pt --onnx x.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.train import sha256_file
from clarity_tse.eval.latency import write_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--ckpt")
    ap.add_argument("--onnx")
    ap.add_argument("--stage", default="desktop")
    a = ap.parse_args()
    d = json.loads(Path(a.raw).read_text())
    for key in ("compute", "hop_compute"):
        if key in d and "hop_ms" in d:
            d[key]["rtf"] = d[key]["mean_ms"] / d["hop_ms"]
    if "hop_compute" in d and "hop" in d:
        d["hop_ms"] = d.get("hop_ms", 1000 * d["hop"] / 16000)
        d["hop_compute"]["rtf"] = d["hop_compute"]["mean_ms"] / d["hop_ms"]
    ck = a.ckpt if a.ckpt and Path(a.ckpt).exists() else None
    if a.onnx:
        d["onnx_sha256"] = sha256_file(Path(a.onnx))
    write_result(a.out, d, ckpt_hash=sha256_file(Path(ck)) if ck else None, stage=a.stage,
                 config=dict(tool=Path(a.raw).name))


if __name__ == "__main__":
    main()

"""Scripted end-to-end demo (desktop): a recorded-style scene is injected as input and processed
hop-by-hop by the *exported* artifacts the engines use (ECAPA ONNX enrollment -> FiLM ONNX ->
streaming extractor ONNX), steered by the target-selection policy, followed by the safety limiter.
Then the C++ engine processes the same input on a simulated real-time clock to count xruns.

The scene uses LibriSpeech speech (CC BY 4.0) and *synthetic* noise only, so the audio written to
results/demo/audio/synthetic_*.wav is redistributable.

    uv run python -m bench.demo [--model b16_s0]
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from clarity_tse.dsp.limiter import SafetyLimiter
from clarity_tse.eval import metrics as M
from clarity_tse.eval.latency import write_result
from clarity_tse.synth import corpus, scene

EXPORT = corpus.CACHE / "export"


def demo_scene(seed: int = 3) -> tuple[scene.Rendered, dict]:
    spec = scene.sample_spec("restaurant", "test", seed=90_000 + seed, snr_db=0.0, duration_s=10.0,
                             head_motion="turn")
    spec = copy.deepcopy(spec)
    for s in spec["sources"]:
        if s["role"] == "noise" and s["noise"]["kind"] != "synthetic":
            s["noise"] = dict(kind="synthetic", synth="clatter")  # no ESC-50 audio in published samples
    return scene.render(spec, enroll_seconds=15.0), spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="b16_s0")
    ap.add_argument("--out", default="results/demo")
    a = ap.parse_args()
    import onnxruntime as ort

    from clarity_tse.embed import speaker
    from clarity_tse.select.policy import PolicyConfig, TargetSelector

    r, spec = demo_scene()
    cfg = dict(line.split("=", 1) for line in (EXPORT / f"{a.model}.cfg").read_text().split())
    H, C = int(cfg["hop"]), int(cfg["channels"])
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(EXPORT / f"{a.model}.onnx"), so, providers=["CPUExecutionProvider"])
    ecapa = ort.InferenceSession(str(EXPORT / "ecapa.onnx"), providers=["CPUExecutionProvider"])
    film_s = ort.InferenceSession(str(EXPORT / f"{a.model}_film.onnx"), providers=["CPUExecutionProvider"])
    # 1) enrollment (15 s) -> embedding -> FiLM, exactly as on the phone
    e = ecapa.run(None, {"wav": r.enrollment[None].astype(np.float32)})[0][0]
    e = (e / np.linalg.norm(e)).astype(np.float32)
    film = film_s.run(None, {"emb": e[None]})[0]
    # 2) hop-by-hop processing with the tau policy updating theta every 100 ms ("facing" mode)
    sel = TargetSelector(PolicyConfig(mode="facing"), [e], embed_fn=speaker.embed)
    n_st = int(cfg["n_states"])
    states = {f"state_{i}": np.zeros([int(d) for d in cfg[f"state_{i}"].split(",")], np.float32) for i in range(n_st)}
    lim = SafetyLimiter()
    T = r.mix.shape[1] // H * H
    out = np.zeros(T, np.float32)
    theta = np.float32(np.nan)
    blk = 1600
    t_hop = []
    for k in range(T // H):
        end = (k + 1) * H
        if end % blk == 0:
            th, _, _ = sel.step(r.mix[:, max(0, end - 6400) : end], r.mix[:, max(0, end - 24000) : end]
                                if end >= 12000 else None, float(r.yaw[end - 1]))
            theta = np.float32(th)
        feed = {"hop": r.mix[None, :C, k * H : end], "theta": np.array([theta], np.float32), "film": film, **states}
        t0 = time.perf_counter()
        res = sess.run(None, feed)
        t_hop.append(1e3 * (time.perf_counter() - t0))
        states = {f"state_{i}": v for i, v in enumerate(res[1:])}
        out[k * H : end] = lim.process(res[0][0])
    delay = H * (1 + int(cfg["lookahead"]))
    y = np.concatenate([out[delay:], np.zeros(delay, np.float32)])  # align for scoring only
    ref = r.target_early[:T]
    mix0 = r.mix[0, :T]
    res_d = dict(model=a.model, scene=dict(kind=spec["kind"], snr_db=spec["snr_db"], duration_s=spec["duration_s"],
                                         head_motion="turn", noise="synthetic"),
                 unprocessed=dict(si_sdr=M.si_sdr(mix0, ref), estoi=M.stoi(mix0, ref, extended=True)),
                 clarity=dict(si_sdr=M.si_sdr(y, ref), estoi=M.stoi(y, ref, extended=True)),
                 output_delay_samples=delay, peak_dbfs=float(20 * np.log10(np.abs(out).max() + 1e-9)),
                 python_ort_hop_ms=dict(p50=float(np.percentile(t_hop, 50)), p99=float(np.percentile(t_hop, 99))))
    od = Path(a.out)
    (od / "audio").mkdir(parents=True, exist_ok=True)
    sf.write(od / "audio" / "synthetic_demo_unprocessed.wav", mix0, 16000, subtype="PCM_16")
    sf.write(od / "audio" / "synthetic_demo_clarity.wav", out, 16000, subtype="PCM_16")
    sf.write(od / "audio" / "synthetic_demo_target_reference.wav", ref, 16000, subtype="PCM_16")
    # 3) C++ engine on the same input, simulated real-time clock
    raw = corpus.CACHE / "demo_input.f32"
    np.ascontiguousarray(r.mix[:C].T).astype(np.float32).tofile(raw)
    rt = Path("rt/build/clarity_rt")
    if rt.exists():
        j = od / "rt_simclock.raw.json"
        subprocess.run([str(rt), "--model", str(EXPORT / f"{a.model}.onnx"), "--input", str(raw), "--minutes", "0.5",
                        "--simclock", "--mute", "--json", str(j)], check=True, capture_output=True)
        res_d["cpp_engine_simclock"] = json.loads(j.read_text())
        j.unlink()
    write_result(od / "demo.json", res_d, config=vars(a))
    print(json.dumps({k: v for k, v in res_d.items() if k != "cpp_engine_simclock"}, indent=1))
    if "cpp_engine_simclock" in res_d:
        print("C++ engine xruns:", res_d["cpp_engine_simclock"]["xruns_total"])


if __name__ == "__main__":
    main()

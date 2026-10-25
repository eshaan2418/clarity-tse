"""Scripted end-to-end demo (desktop): a recorded-style scene is injected as input and processed
hop-by-hop by the *exported* artifacts the engines use (ECAPA ONNX enrollment -> FiLM ONNX ->
streaming extractor ONNX), steered by the target-selection policy (val-tuned, "facing" mode),
then device volume and the safety limiter. The C++ engine then processes the same input on the
real CoreAudio clock (muted) to count xruns.

Two scenes, both reported:
  * ``faced``    first seed >= 3 whose partner is within +-8 deg of the nose at the start (the case
                 the "focus on who I'm facing" mode is designed for; selection rule recorded in the output)
  * ``off_axis`` seed 3 (partner ~20 deg off-axis, other talkers nearer the nose): a documented failure case
Each also gets the same model with the *oracle* direction, to separate selection from extraction errors.

Speech is LibriSpeech (CC BY 4.0) and noise is synthetic, so results/demo/audio/synthetic_*.wav are
redistributable.

    uv run python -m bench.demo [--model b16_s0]
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from clarity_tse.dsp.limiter import SafetyLimiter
from clarity_tse.eval import metrics as M
from clarity_tse.eval.latency import write_result
from clarity_tse.synth import corpus, scene

EXPORT = corpus.CACHE / "export"
PLAYBACK_DB = -10.0  # device volume before the limiter (cf. study PRESENT_DB)
BLOCK = 1600  # policy block (100 ms)


def demo_scene(seed: int) -> tuple[scene.Rendered, dict]:
    spec = scene.sample_spec("restaurant", "test", seed=90_000 + seed, snr_db=0.0, duration_s=10.0,
                             head_motion="drift")  # same head-motion model as the test set
    spec = copy.deepcopy(spec)
    for s in spec["sources"]:
        if s["role"] == "noise" and s["noise"]["kind"] != "synthetic":
            s["noise"] = dict(kind="synthetic", synth="clatter")  # no ESC-50 audio in published samples
    return scene.render(spec, enroll_seconds=15.0), spec


def faced_seed(start: int = 3, max_deg: float = 8.0) -> int:
    s = start
    while True:
        spec = scene.sample_spec("restaurant", "test", seed=90_000 + s, snr_db=0.0, duration_s=10.0, head_motion="drift")
        lp, tp = np.array(spec["listener"]["pos"]), np.array(spec["sources"][0]["pos"])
        rel = (np.rad2deg(np.arctan2(tp[1] - lp[1], tp[0] - lp[0])) - spec["yaw_keys"][0][1] + 180) % 360 - 180
        if abs(rel) <= max_deg:
            return s
        s += 1


def run(r: scene.Rendered, model: str, oracle: bool) -> tuple[np.ndarray, np.ndarray, list[float]]:
    import onnxruntime as ort

    from clarity_tse.embed import speaker
    from clarity_tse.select.policy import PolicyConfig, TargetSelector

    cfg = dict(line.split("=", 1) for line in (EXPORT / f"{model}.cfg").read_text().split())
    H, C = int(cfg["hop"]), int(cfg["channels"])
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(EXPORT / f"{model}.onnx"), so, providers=["CPUExecutionProvider"])
    ecapa = ort.InferenceSession(str(EXPORT / "ecapa.onnx"), providers=["CPUExecutionProvider"])
    film_s = ort.InferenceSession(str(EXPORT / f"{model}_film.onnx"), providers=["CPUExecutionProvider"])
    # enrollment (15 s) -> embedding -> FiLM, exactly as on the phone
    e = ecapa.run(None, {"wav": r.enrollment[None].astype(np.float32)})[0][0]
    e = (e / np.linalg.norm(e)).astype(np.float32)
    film = film_s.run(None, {"emb": e[None]})[0]
    tuned = Path("results/m4/tuned_policy.json")
    over = json.loads(tuned.read_text())["best"] if tuned.exists() else {}
    sel = TargetSelector(PolicyConfig(mode="facing", **over), [e], embed_fn=speaker.embed)
    true_rel = scene.relative_azimuth(r.meta["target_world_az"], r.yaw)
    states = {f"state_{i}": np.zeros([int(d) for d in cfg[f"state_{i}"].split(",")], np.float32)
              for i in range(int(cfg["n_states"]))}
    lim = SafetyLimiter()
    g = 10 ** (PLAYBACK_DB / 20)
    T = r.mix.shape[1] // H * H
    out, raw = np.zeros(T, np.float32), np.zeros(T, np.float32)
    theta = np.float32(np.nan)
    t_hop = []
    for k in range(T // H):
        end = (k + 1) * H
        if oracle:
            theta = np.float32(true_rel[end - 1])
        elif end % BLOCK == 0 or (k * H) // BLOCK != end // BLOCK:
            b_end = end // BLOCK * BLOCK
            th, _, _ = sel.step(r.mix[:, max(0, b_end - 6400) : b_end],
                                r.mix[:, max(0, b_end - 24000) : b_end] if b_end >= 12000 else None,
                                float(r.yaw[b_end - 1]), yaw_doa=float(r.yaw[max(0, b_end - 6400) : b_end].mean()),
                                yaw_voice=float(r.yaw[max(0, b_end - 24000) : b_end].mean()))
            theta = np.float32(th)
        feed = {"hop": r.mix[None, :C, k * H : end], "theta": np.array([theta], np.float32), "film": film, **states}
        t0 = time.perf_counter()
        res = sess.run(None, feed)
        t_hop.append(1e3 * (time.perf_counter() - t0))
        states = {f"state_{i}": v for i, v in enumerate(res[1:])}
        raw[k * H : end] = res[0][0]
        out[k * H : end] = lim.process(res[0][0] * g)
    delay = H * (1 + int(cfg["lookahead"]))
    return out, raw, t_hop if not oracle else [float(delay)]


def score(x: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    return dict(si_sdr=M.si_sdr(x, ref), estoi=M.stoi(x, ref, extended=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="b16_s0")
    ap.add_argument("--out", default="results/demo")
    a = ap.parse_args()
    od = Path(a.out)
    (od / "audio").mkdir(parents=True, exist_ok=True)
    res_d: dict = dict(model=a.model, playback_gain_db=PLAYBACK_DB, scenes={})
    cpp_input = None
    for tag, seed in (("faced", faced_seed()), ("off_axis", 3)):
        r, spec = demo_scene(seed)
        out, raw, t_hop = run(r, a.model, oracle=False)
        out_o, raw_o, (delay,) = run(r, a.model, oracle=True)
        d = int(delay)
        T = len(out)

        def al(x, d=d):
            return np.concatenate([x[d:], np.zeros(d, np.float32)])  # time-align for scoring only

        ref, mix0 = r.target_early[:T], r.mix[0, :T]
        res_d["scenes"][tag] = dict(
            seed=seed, selection_rule="first seed >= 3 with |target rel. azimuth| <= 8 deg" if tag == "faced" else "fixed seed 3",
            target_rel_az_start_deg=r.meta["target_rel_az_start"], kind=spec["kind"], snr_db=spec["snr_db"],
            n_sources=len(spec["sources"]), noise="synthetic", head_motion="drift",
            unprocessed=score(mix0, ref), clarity_policy=score(al(raw), ref), clarity_oracle_direction=score(al(raw_o), ref),
            clarity_policy_after_volume_and_limiter=score(al(out), ref),
            fraction_samples_at_ceiling=float(np.mean(np.abs(out) >= 0.999 * 10 ** (-12 / 20))),
            peak_dbfs=float(20 * np.log10(np.abs(out).max() + 1e-9)),
            python_ort_hop_ms=dict(p50=float(np.percentile(t_hop, 50)), p99=float(np.percentile(t_hop, 99))))
        sf.write(od / "audio" / f"synthetic_demo_{tag}_unprocessed.wav", mix0 * 10 ** (PLAYBACK_DB / 20), 16000, subtype="PCM_16")
        sf.write(od / "audio" / f"synthetic_demo_{tag}_clarity.wav", out, 16000, subtype="PCM_16")
        sf.write(od / "audio" / f"synthetic_demo_{tag}_target_reference.wav", ref * 10 ** (PLAYBACK_DB / 20), 16000,
                 subtype="PCM_16")
        if tag == "faced":
            cpp_input = r.mix
    # C++ engine on the same input (faced scene), real CoreAudio clock on macOS
    rt = Path("rt/build/clarity_rt")
    if rt.exists() and cpp_input is not None:
        cfg = dict(line.split("=", 1) for line in (EXPORT / f"{a.model}.cfg").read_text().split())
        raw_in = corpus.CACHE / "demo_input.f32"
        np.ascontiguousarray(cpp_input[: int(cfg["channels"])].T).astype(np.float32).tofile(raw_in)
        j = od / "rt.raw.json"
        clock = [] if sys.platform == "darwin" else ["--simclock"]
        subprocess.run([str(rt), "--model", str(EXPORT / f"{a.model}.onnx"), "--input", str(raw_in), "--minutes", "0.5",
                        *clock, "--mute", "--json", str(j)], check=True, capture_output=True)
        res_d["cpp_engine"] = json.loads(j.read_text())
        j.unlink()
    write_result(od / "demo.json", res_d, config=vars(a))
    print(json.dumps(res_d["scenes"], indent=1))
    if "cpp_engine" in res_d:
        print("C++ engine (", res_d["cpp_engine"]["clock"], ") xruns:", res_d["cpp_engine"]["xruns_total"])


if __name__ == "__main__":
    main()

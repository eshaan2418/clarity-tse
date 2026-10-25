"""M9: evaluation on consented real recordings (no clean reference): DNSMOS + target-talker ASR WER.

Inputs (git-ignored, see HUMAN_TODO.md §3):
  data/personal/raw/scenes/<scene>.wav     binaural (L, R) recording at the listener's ears; mono allowed
  data/personal/raw/scenes/<scene>.json    {"target": "<name>", "transcript": "...", "target_azimuth_deg": 0,
                                            "tags": ["similar_voice", "low_snr", "reverberant", "off_axis"]}
  data/personal/raw/enroll/<name>.wav      10-20 s enrollment of that person (with consent)

Systems: unprocessed | rnnoise | mpdr (steered to target_azimuth_deg, default 0 = straight ahead) |
clarity (operating point; enrollment + direction) | clarity_1ch (single-mic model, enrollment only).
Per-scene rows -> results/m9/per_scene.csv (metrics only, no audio); A/B clips for the listening
test -> data/personal/out/ (git-ignored, stays on this machine).

Without personal data, ``--validate-on-study`` runs the identical pipeline on the simulated
single-sentence study stimuli (transcripts known) and labels the output ``simulated``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from clarity_tse.eval.latency import write_result
from clarity_tse.eval.stats import bootstrap_ci, paired_test
from clarity_tse.synth import corpus

RAW = corpus.REPO / "data" / "personal" / "raw"
OUTP = corpus.REPO / "data" / "personal" / "out"


def load_scene(p: Path) -> tuple[np.ndarray, dict]:
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = np.stack([corpus._resample(x[:, c], sr) for c in range(x.shape[1])]).astype(np.float32)
    meta = json.loads(p.with_suffix(".json").read_text()) if p.with_suffix(".json").exists() else {}
    return x, meta


def process(mix: np.ndarray, emb: np.ndarray | None, theta_deg: float, op: str) -> dict[str, np.ndarray]:
    from bench.evaluate import run_system

    two = mix.shape[0] >= 2
    m2 = mix[:2] if two else np.concatenate([mix, mix])
    th = np.full(m2.shape[1], theta_deg, np.float32)
    a = dict(mix=m2, theta=th, interference=None)
    out = {"unprocessed": m2[0], "rnnoise": run_system("rnnoise", a, None)}
    if two:
        out["mpdr"] = run_system("mpdr", a, None)
        out["clarity"] = run_system(f"model:{op}", a, emb)
    out["clarity_1ch"] = run_system(f"model:{op.split('_')[0]}_1ch_s0", a, emb)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="b16_s0")
    ap.add_argument("--validate-on-study", action="store_true")
    ap.add_argument("--out", default="results/m9")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    from clarity_tse.embed import speaker
    from clarity_tse.eval import asr
    from clarity_tse.eval import metrics as M

    rows = []
    if a.validate_on_study:
        man = json.loads((corpus.REPO / "study" / "stimuli_manifest.json").read_text())
        stim = corpus.CACHE / "study_stimuli"
        g = 10 ** (-man["present_db"] / 20)
        cond_map = {"unprocessed": "unprocessed", "rnnoise": "rnnoise", "mpdr": "mpdr", "clarity": "clarity_lo"}
        for it in man["items"]:
            for sysn, c in cond_map.items():
                y = sf.read(stim / c / f"{it['item']}.wav", dtype="float32")[0] * g
                w, n = asr.wer(asr.transcribe(y), it["transcript"])
                d = M.dnsmos(y)
                rows.append(dict(scene=it["item"], system=sysn, kind=it["kind"], snr_db=it["snr"], wer=w, n_words=n,
                                 dnsmos_sig=d["SIG"], dnsmos_bak=d["BAK"], dnsmos_ovrl=d["OVRL"], tags=""))
        label, tag = "simulated (study stimuli, pipeline validation)", "validation_simulated"
    else:
        scenes = sorted((RAW / "scenes").glob("*.wav")) if (RAW / "scenes").exists() else []
        if not scenes:
            print("no consented recordings in data/personal/raw/scenes (M9 blocked-with-fallback; see HUMAN_TODO.md)")
            return
        OUTP.mkdir(parents=True, exist_ok=True)
        for p in scenes:
            mix, meta = load_scene(p)
            enr = RAW / "enroll" / f"{meta.get('target', '')}.wav"
            emb = speaker.embed(corpus.read_audio(enr)) if enr.exists() else None
            outs = process(mix, emb, float(meta.get("target_azimuth_deg", 0.0)), a.ckpt)
            for sysn, y in outs.items():
                sf.write(OUTP / f"{p.stem}__{sysn}.wav", y, 16000, subtype="FLOAT")  # local A/B clips only
                d = M.dnsmos(y)
                r = dict(scene=p.stem, system=sysn, tags=",".join(meta.get("tags", [])),
                         dnsmos_sig=d["SIG"], dnsmos_bak=d["BAK"], dnsmos_ovrl=d["OVRL"])
                if meta.get("transcript"):
                    r["wer"], r["n_words"] = asr.wer(asr.transcribe(y), meta["transcript"])
                rows.append(r)
        label, tag = "real recordings (consented)", "per_scene"
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{tag}.csv", index=False)
    table, tests = [], []
    for sysn, g_ in df.groupby("system"):
        r = dict(system=sysn, n=len(g_))
        for m in ("wer", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"):
            if m in g_ and g_[m].notna().any():
                r[m], r[f"{m}_lo"], r[f"{m}_hi"] = bootstrap_ci(g_[m].to_numpy())
        table.append(r)
    for m in ("wer", "dnsmos_ovrl"):
        if m in df:
            piv = df.pivot_table(index="scene", columns="system", values=m)
            for b in [s for s in piv.columns if s != "clarity" and "clarity" in piv]:
                tests.append(dict(metric=m, a="clarity", b=b, **paired_test(piv["clarity"].to_numpy(), piv[b].to_numpy())))
    failure = []
    if "tags" in df and df.tags.str.len().gt(0).any():
        for t in sorted({x for s in df.tags for x in s.split(",") if x}):
            g_ = df[df.tags.str.contains(t) & (df.system == "clarity")]
            failure.append(dict(tag=t, n=len(g_), wer=float(g_.wer.mean()) if "wer" in g_ else None,
                                dnsmos_ovrl=float(g_.dnsmos_ovrl.mean())))
    write_result(out / f"summary_{tag}.json", dict(label=label, table=table, paired=tests, failure_cases=failure),
                 config=vars(a), stage="simulated" if a.validate_on_study else "desktop")
    print(pd.DataFrame(table).round(3).to_string())


if __name__ == "__main__":
    main()

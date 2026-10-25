"""Analysis of listening-study responses (real: study/data/*.jsonl; dry run: study/data_simulated).

Word recognition: binomial mixed model (statsmodels BinomialBayesMixedGLM, random intercepts for
participant and item) + robustness LMM on empirical-logit proportions; planned contrasts of
clarity_lo vs every baseline with Holm correction. MUSHRA/effort: LMM with participant intercepts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from clarity_tse.eval.stats import holm


def load(data_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(data_dir.glob("*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["condition"] = pd.Categorical(df["condition"], categories=["unprocessed", "rnnoise", "mpdr", "clarity_lo", "clarity_hi"])
    df["pc"] = df["correct"] / df["words"]
    out: dict = dict(n_participants=int(df.participant.nunique()), n_trials=int(len(df)),
                     simulated=bool(df.get("simulated", pd.Series([False])).any()))
    out["percent_correct"] = (df.groupby("condition", observed=True)["pc"].mean() * 100).round(2).to_dict()
    # empirical-logit LMM (robustness check, gives standard errors + p-values)
    df["elogit"] = np.log((df["correct"] + 0.5) / (df["words"] - df["correct"] + 0.5))
    lmm = smf.mixedlm("elogit ~ C(condition, Treatment('clarity_lo')) + snr", df, groups=df["participant"]).fit()
    contrasts = {}
    for name in ["unprocessed", "rnnoise", "mpdr", "clarity_hi"]:
        key = f"C(condition, Treatment('clarity_lo'))[T.{name}]"
        contrasts[name] = dict(coef=float(-lmm.params[key]), p=float(lmm.pvalues[key]))
    adj = holm([v["p"] for v in contrasts.values()])
    for (k, v), pa in zip(contrasts.items(), adj, strict=True):
        v["p_holm"] = pa
    out["elogit_lmm_clarity_lo_minus_baseline"] = contrasts
    # binomial GLMM (variational Bayes); expand to Bernoulli rows
    long = df.loc[df.index.repeat(df["words"])].copy()
    long["y"] = np.concatenate([np.r_[np.ones(c), np.zeros(w - c)] for c, w in zip(df["correct"], df["words"], strict=True)])
    glmm = BinomialBayesMixedGLM.from_formula("y ~ C(condition) + snr", {"p": "0 + C(participant)", "i": "0 + C(item)"}, long).fit_vb()
    out["glmm_fixed_effects"] = {n: float(m) for n, m in zip(glmm.model.exog_names, glmm.fe_mean, strict=True)}
    for col in ("mushra", "effort"):
        if col in df:
            m = smf.mixedlm(f"{col} ~ C(condition)", df, groups=df["participant"]).fit()
            out[f"{col}_means"] = df.groupby("condition", observed=True)[col].mean().round(2).to_dict()
            out[f"{col}_lmm_params"] = {k: float(v) for k, v in m.params.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="study/data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.data)
    df = load(d)
    if df.empty:
        print(f"no responses in {d} (awaiting-human)")
        return
    res = analyze(df)
    out = Path(a.out) if a.out else Path("results/m10") / ("analysis_simulated.json" if res["simulated"] else "analysis.json")
    from clarity_tse.eval.latency import write_result
    write_result(out, res, config=dict(data=str(d)))
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()

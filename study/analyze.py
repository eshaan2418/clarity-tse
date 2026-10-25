"""Analysis of listening-study responses (real: study/data/*.jsonl; dry run: study/data_simulated).

Rows are exactly what study/server.py stores:
  type=word   : participant, item, condition, snr, typed, effort
  type=mushra : participant, item, condition (5 conditions + reference + anchor), rating
Typed text is scored against the item transcript (study/score.py; manifest next to the data if
present, else study/stimuli_manifest.json).

Word recognition: binomial mixed model (statsmodels BinomialBayesMixedGLM, random intercepts for
participant and item) + robustness LMM on empirical-logit proportions; planned contrasts of
clarity_lo vs every other condition with Holm correction. Effort and MUSHRA: LMM with participant
intercepts. MUSHRA post-screening (BS.1534-style): exclude participants who rate the hidden
reference below 90 on more than 15 % of screens.
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
from study.score import score

CONDS = ["unprocessed", "rnnoise", "mpdr", "clarity_lo", "clarity_hi"]
REF = "clarity_lo"


def load(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    for f in sorted(data_dir.glob("*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    man_path = data_dir / "stimuli_manifest.json"
    if not man_path.exists():
        man_path = Path(__file__).resolve().parent / "stimuli_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"items": []}
    return pd.DataFrame(rows), man


def score_words(word: pd.DataFrame, man: dict) -> pd.DataFrame:
    trans = {it["item"]: it["transcript"] for it in man["items"]}
    word = word.copy()
    sc = [score(t, trans[i]) for t, i in zip(word["typed"], word["item"], strict=True)]
    word["correct"] = [c for c, _ in sc]
    word["words"] = [n for _, n in sc]
    return word[word["words"] > 0]


def _contrasts(fit, prefix: str) -> dict:
    out = {}
    for name in [c for c in CONDS if c != REF]:
        key = f"{prefix}[T.{name}]"
        out[name] = dict(coef=float(-fit.params[key]), se=float(fit.bse[key]), p=float(fit.pvalues[key]))
    for v, pa in zip(out.values(), holm([v["p"] for v in out.values()]), strict=True):
        v["p_holm"] = pa
    return out


def analyze(df: pd.DataFrame, man: dict) -> dict:
    out: dict = dict(simulated=bool(df["simulated"].any()) if "simulated" in df else False)
    word = score_words(df[df["type"] == "word"], man)
    word["condition"] = pd.Categorical(word["condition"], categories=CONDS)
    out.update(n_participants=int(word.participant.nunique()), n_word_trials=int(len(word)))
    word["pc"] = word["correct"] / word["words"]
    out["percent_keywords_correct"] = (word.groupby("condition", observed=True)["pc"].mean() * 100).round(2).to_dict()
    out["percent_keywords_correct_by_snr"] = {f"{c}@{int(s):+d}": float(v) for (c, s), v in
                                              (word.groupby(["condition", "snr"], observed=True)["pc"].mean() * 100).round(2).items()}
    word["elogit"] = np.log((word["correct"] + 0.5) / (word["words"] - word["correct"] + 0.5))
    prefix = f"C(condition, Treatment('{REF}'))"
    lmm = smf.mixedlm(f"elogit ~ {prefix} + snr", word, groups=word["participant"]).fit(method=["lbfgs", "powell"])
    out[f"elogit_lmm_{REF}_minus_condition"] = _contrasts(lmm, prefix)
    long = word.loc[word.index.repeat(word["words"])].copy()
    long["y"] = np.concatenate([np.r_[np.ones(c), np.zeros(w - c)] for c, w in zip(word["correct"], word["words"], strict=True)])
    glmm = BinomialBayesMixedGLM.from_formula(f"y ~ {prefix} + snr", {"p": "0 + C(participant)", "i": "0 + C(item)"},
                                              long).fit_vb()
    out["glmm_fixed_effects_logit"] = {n: dict(mean=float(m), sd=float(s)) for n, m, s in
                                       zip(glmm.model.exog_names, glmm.fe_mean, glmm.fe_sd, strict=True)}
    eff = smf.mixedlm(f"effort ~ {prefix} + snr", word, groups=word["participant"]).fit(method=["lbfgs", "powell"])
    out["effort_means"] = word.groupby("condition", observed=True)["effort"].mean().round(2).to_dict()
    out[f"effort_lmm_{REF}_minus_condition"] = _contrasts(eff, prefix)
    mu = df[df["type"] == "mushra"].copy()
    if len(mu):
        ref = mu[mu.condition == "reference"]
        bad_rate = (ref.rating < 90).groupby(ref.participant).mean()
        excluded = sorted(bad_rate[bad_rate > 0.15].index)
        out["mushra_excluded_participants"] = excluded
        mu = mu[~mu.participant.isin(excluded) & mu.condition.isin(CONDS)]
        if mu.participant.nunique() >= 2:
            mu["condition"] = pd.Categorical(mu["condition"], categories=CONDS)
            out["mushra_means"] = mu.groupby("condition", observed=True)["rating"].mean().round(2).to_dict()
            m = smf.mixedlm(f"rating ~ {prefix}", mu, groups=mu["participant"]).fit(method=["lbfgs", "powell"])
            out[f"mushra_lmm_{REF}_minus_condition"] = _contrasts(m, prefix)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="study/data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.data)
    df, man = load(d)
    if df.empty:
        print(f"no responses in {d} (awaiting-human)")
        return
    res = analyze(df, man)
    out = Path(a.out) if a.out else Path("results/m10") / ("analysis_simulated.json" if res["simulated"] else "analysis.json")
    from clarity_tse.eval.latency import write_result

    write_result(out, res, config=dict(data=str(d)))
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()

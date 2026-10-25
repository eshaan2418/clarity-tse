import numpy as np
import pandas as pd

from study.analyze import analyze
from study.randomize import CONDITIONS, trial_list
from study.score import score
from study.simulate import fake_manifest, simulate


def test_counterbalanced_lists():
    man = fake_manifest(np.random.default_rng(0))
    lists = [trial_list(p, man) for p in range(5)]
    for L in lists:
        items = [t["item"] for t in L["word"]]
        assert len(set(items)) == len(items)  # no repeats within a participant
        for c in CONDITIONS:
            assert sum(t["condition"] == c for t in L["word"]) == 6  # 2 per SNR x 3 SNRs
        assert all(len(m["stimuli"]) == len(CONDITIONS) + 2 for m in L["mushra"])
    # across 5 participants (one Latin-square cycle) every item is heard in every condition
    seen: dict[str, set] = {}
    for L in lists:
        for t in L["word"]:
            seen.setdefault(t["item"], set()).add(t["condition"])
    assert all(v == set(CONDITIONS) for v in seen.values())


def test_keyword_scoring():
    assert score("the cat sat on a mat", "THE CAT SAT ON THE MAT") == (3, 3)
    assert score("Cat, cat!", "THE CAT AND THE CAT") == (2, 2)
    assert score("", "HELLO WORLD") == (0, 2)
    assert score("dog", "CAT") == (0, 1)


def test_analysis_recovers_assumed_ordering():
    rows, man = simulate(12, seed=1)
    res = analyze(pd.DataFrame(rows), man)
    pc = res["percent_keywords_correct"]
    assert pc["clarity_lo"] > pc["unprocessed"]
    assert res["elogit_lmm_clarity_lo_minus_condition"]["unprocessed"]["coef"] > 0
    assert res["simulated"] is True
    assert res["mushra_excluded_participants"] == ["S11"]  # reference post-screening works
    assert res["mushra_means"]["clarity_lo"] > res["mushra_means"]["unprocessed"]

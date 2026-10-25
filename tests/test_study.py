import json

from study.analyze import analyze
from study.randomize import CONDITIONS, trial_list
from study.simulate import simulate


def test_counterbalanced_lists():
    man = json.load(open("results/m1/testsets/test_main.json")) if __import__("os").path.exists("results/m1/testsets/test_main.json") else None
    if man is None:
        return
    a, b = trial_list(0, man), trial_list(1, man)
    assert len({t["item"] for t in a}) == len(a)  # no repeats
    for cond in CONDITIONS:
        assert sum(t["condition"] == cond for t in a) == sum(t["condition"] == cond for t in b)
    assert {(t["item"], t["condition"]) for t in a} != {(t["item"], t["condition"]) for t in b}


def test_analysis_recovers_assumed_ordering():
    import pandas as pd
    res = analyze(pd.DataFrame(simulate(12, seed=1)))
    pc = res["percent_correct"]
    assert pc["clarity_lo"] > pc["unprocessed"]
    assert res["elogit_lmm_clarity_lo_minus_baseline"]["unprocessed"]["coef"] > 0
    assert res["simulated"] is True

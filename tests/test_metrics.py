import numpy as np
import pytest
import torch

from clarity_tse.eval import metrics as M
from clarity_tse.eval.stats import bootstrap_ci, holm, paired_test


def _speechlike(seed=0, n=48000):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from make_fixtures import synth_speaker_utt
    return synth_speaker_utt(np.random.default_rng(seed), 130, [700, 1500, 2600], n / 16000)


def test_si_sdr_matches_torchmetrics():
    from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as tm
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000)
    est = ref + 0.3 * rng.standard_normal(16000)
    ours = M.si_sdr(est, ref)
    theirs = float(tm(torch.tensor(est), torch.tensor(ref), zero_mean=True))
    assert ours == pytest.approx(theirs, abs=1e-3)


def test_intelligibility_metrics_monotonic_in_snr():
    ref = _speechlike()
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(ref)) * np.sqrt(np.mean(ref**2))
    vals = [M.intrusive_suite(ref + noise * 10 ** (-s / 20), ref, fast=True) for s in (-5, 5, 20)]
    for k in ("si_sdr", "estoi", "stoi"):
        assert vals[0][k] < vals[1][k] < vals[2][k], k
    assert M.stoi(ref, ref) > 0.99


@pytest.mark.slow
def test_haspi_identity_and_noise():
    ref = _speechlike(n=32000)
    assert M.haspi(ref, ref) > 0.95
    noisy = ref + np.random.default_rng(2).standard_normal(len(ref)) * np.sqrt(np.mean(ref**2)) * 2
    assert M.haspi(noisy, ref) < M.haspi(ref, ref)
    # severe hearing loss lowers predicted intelligibility of unamplified speech
    loss = np.array([60, 65, 70, 75, 80, 80, 85, 90])
    assert M.haspi(ref, ref, audiogram=loss) < M.haspi(ref, ref)


@pytest.mark.skipif(not M.DNSMOS_PATH.exists(), reason="DNSMOS model not downloaded")
def test_dnsmos_prefers_clean():
    ref = _speechlike()
    noisy = ref + np.random.default_rng(3).standard_normal(len(ref)) * np.sqrt(np.mean(ref**2))
    assert M.dnsmos(ref)["BAK"] > M.dnsmos(noisy)["BAK"]


def test_stats():
    m, lo, hi = bootstrap_ci(np.arange(100.0))
    assert lo < m < hi
    r = paired_test(np.arange(50.0) + 1, np.arange(50.0))
    assert r["mean_diff"] == pytest.approx(1.0) and r["wilcoxon_p"] < 1e-3
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

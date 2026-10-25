import numpy as np
import pytest
import torch

from clarity_tse.personalize.audiogram import PROFILES, validate
from clarity_tse.personalize.rho import (
    MAX_BAND_GAIN_DB,
    AudibilityModel,
    RhoStage,
    intelligibility_proxy,
    nalr_gains_db,
)


def test_gain_caps_enforced_for_extreme_audiograms():
    rho = RhoStage(512, 80, per_listener=True)
    with torch.no_grad():
        rho.offset.fill_(50.0)  # try to force absurd gains
        for p in rho.net.parameters():
            p.add_(torch.randn_like(p) * 5)
    y = torch.randn(1, 257, 50) * 0.01
    for ag in ([120] * 8, [0] * 8, [-10] * 8):
        g = rho.gain_db(y, y, torch.tensor([ag], dtype=torch.float32))
        assert float(g.max()) <= MAX_BAND_GAIN_DB + 1e-4
        assert float(g.min()) >= -1e-4


def test_validate_rejects_bad_input():
    with pytest.raises(ValueError):
        validate([10, 20])
    with pytest.raises(ValueError):
        validate([float("nan")] * 8)
    assert validate([200] * 8).max() == 120


def test_nalr_matches_pyclarity():
    from clarity.enhancer.nalr import NALR
    from clarity.utils.audiogram import Audiogram

    from clarity_tse.personalize.audiogram import FREQS
    ag = np.array(PROFILES["moderate_sloping"], float)
    # our closed-form gains at NAL frequencies vs. pyclarity's FIR magnitude response there
    n = NALR(220, 16000)
    fir, _ = n.build(Audiogram(levels=ag, frequencies=FREQS))
    H = np.abs(np.fft.rfft(fir, 16000))
    for f, g in zip([500, 1000, 2000, 4000], nalr_gains_db(ag)[1:5], strict=True):
        assert abs(20 * np.log10(H[f]) - g) < 1.5


def test_proxy_prefers_audible_signal():
    m = AudibilityModel()
    t = np.arange(32000) / 16000
    clean = torch.tensor((0.05 * np.sin(2 * np.pi * 1000 * t) * (1 + np.sin(2 * np.pi * 4 * t))) [None], dtype=torch.float32)
    loss = torch.tensor([PROFILES["moderate_flat"]], dtype=torch.float32)
    nh = torch.zeros(1, 8)
    ref = m(clean, nh)
    quiet = intelligibility_proxy(m(clean * 0.01, loss), ref)
    loud = intelligibility_proxy(m(clean, loss), ref)
    assert float(loud) > float(quiet)

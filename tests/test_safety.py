"""Hearing-safety tests: the limiter cannot be configured or driven into exceeding its ceiling."""
import numpy as np
import pytest

from clarity_tse.dsp.limiter import ABSOLUTE_MAX_CEILING_DBFS, MAX_USER_GAIN_DB, SafetyLimiter, db_to_lin


@pytest.mark.parametrize("ceiling,gain", [(-12, 0), (0, 100), (6, 60), (float("nan"), float("nan")),
                                          (float("inf"), 20), (-30, -100), (-1, 20)])
def test_ceiling_never_exceeded(ceiling, gain):
    lim = SafetyLimiter(ceiling_dbfs=ceiling, user_gain_db=gain)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(8000) * 10, np.full(100, 1e6), np.full(10, np.nan),
                        np.full(10, np.inf), -np.full(10, np.inf), rng.standard_normal(4000) * 1e-3])
    y = lim.process(x)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= db_to_lin(ABSOLUTE_MAX_CEILING_DBFS) + 1e-7
    assert np.max(np.abs(y)) <= lim.ceiling + 1e-7
    assert lim.user_gain_db <= MAX_USER_GAIN_DB


def test_cannot_bypass_by_mutating_after_init():
    lim = SafetyLimiter()
    lim.ceiling_dbfs = 20.0  # attacker/bug sets an unsafe ceiling after construction
    y = lim.process(np.full(1000, 100.0))
    assert np.max(np.abs(y)) <= db_to_lin(ABSOLUTE_MAX_CEILING_DBFS) + 1e-7


def test_multichannel_and_transparent_below_ceiling():
    lim = SafetyLimiter(ceiling_dbfs=-6)
    x = 0.01 * np.random.default_rng(1).standard_normal((2, 4000))
    y = lim.process(x)
    np.testing.assert_allclose(y, x, atol=1e-6)

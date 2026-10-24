"""Bootstrap confidence intervals and paired significance tests."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1-alpha) CI, ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired comparison of two systems on the same items: mean diff + bootstrap CI + Wilcoxon p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    m, lo, hi = bootstrap_ci(d)
    try:
        p = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return dict(mean_diff=m, ci_lo=lo, ci_hi=hi, wilcoxon_p=p, n=int(ok.sum()))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()

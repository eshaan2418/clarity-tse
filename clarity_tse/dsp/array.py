"""Array geometry, steering vectors, beamformers (DS, MVDR/MPDR) and DOA (GCC-PHAT / SRP-PHAT).

Convention: azimuth in degrees, 0 = straight ahead of the listener's nose, positive = to the
listener's left. ``relative azimuth = world azimuth - head yaw``. The simulated array is a
binaural earbud pair on the interaural axis (spacing ``EAR_SPACING``); a 2-mic line array only
resolves *lateral* angle, so front/back are confused (sin(theta) is what is observable).
"""

from __future__ import annotations

import numpy as np

SPEED_OF_SOUND = 343.0
EAR_SPACING = 0.18


def tdoa_right_minus_left(theta_deg: np.ndarray | float, d: float = EAR_SPACING) -> np.ndarray:
    """Arrival time at right mic minus left mic (s). Positive for sources on the left."""
    return d * np.sin(np.deg2rad(theta_deg)) / SPEED_OF_SOUND


def steering_vector(theta_deg: float, freqs: np.ndarray, d: float = EAR_SPACING) -> np.ndarray:
    """(2, F) relative transfer function w.r.t. the left (reference) mic, free field."""
    tau = tdoa_right_minus_left(theta_deg, d)
    return np.stack([np.ones_like(freqs, dtype=complex), np.exp(-2j * np.pi * freqs * tau)])


def delay_and_sum(X: np.ndarray, theta_deg: float, freqs: np.ndarray) -> np.ndarray:
    """X: (2, F, K) -> (F, K)."""
    a = steering_vector(theta_deg, freqs)
    w = a / a.shape[0]
    return np.einsum("cf,cfk->fk", w.conj(), X)


def mvdr_weights(R: np.ndarray, a: np.ndarray, loading: float = 1e-3) -> np.ndarray:
    """R: (F, C, C), a: (C, F) -> w: (C, F)."""
    C = R.shape[-1]
    tr = np.real(np.trace(R, axis1=1, axis2=2))[:, None, None] / C
    Rl = R + loading * (tr + 1e-10) * np.eye(C)[None]
    Rinv_a = np.linalg.solve(Rl, a.T[..., None])[..., 0]  # (F, C)
    denom = np.einsum("fc,fc->f", a.T.conj(), Rinv_a)
    return (Rinv_a / (denom[:, None] + 1e-12)).T


def mvdr(X: np.ndarray, theta_deg: float, freqs: np.ndarray, noise: np.ndarray | None = None,
         alpha: float = 0.98, loading: float = 1e-2) -> np.ndarray:
    """Causal MVDR. If ``noise`` (2,F,K) is given, uses an oracle recursive noise covariance
    (upper-bound baseline); otherwise MPDR on the mixture covariance (practical, causal)."""
    a = steering_vector(theta_deg, freqs)
    src = X if noise is None else noise
    C, F, K = X.shape
    R = np.tile(np.eye(C, dtype=complex)[None] * 1e-6, (F, 1, 1))
    Y = np.empty((F, K), dtype=complex)
    for k in range(K):
        v = src[:, :, k].T  # (F, C)
        R = alpha * R + (1 - alpha) * np.einsum("fc,fd->fcd", v, v.conj())
        w = mvdr_weights(R, a, loading)
        Y[:, k] = np.einsum("cf,cf->f", w.conj(), X[:, :, k])
    return Y


def srp_phat(X: np.ndarray, freqs: np.ndarray, grid_deg: np.ndarray, fmin: float = 200.0,
             fmax: float = 4000.0) -> np.ndarray:
    """Steered-response power with PHAT weighting per frame. X: (2,F,K) -> (G, K)."""
    band = (freqs >= fmin) & (freqs <= fmax)
    cross = X[0, band] * X[1, band].conj()
    cross = cross / (np.abs(cross) + 1e-12)
    tau = tdoa_right_minus_left(grid_deg)  # (G,)
    # E[X0 X1*] for a source at theta = exp(+j 2 pi f tau)
    steer = np.exp(-2j * np.pi * freqs[band][None, :] * tau[:, None])  # (G, Fb)
    return np.real(np.einsum("gf,fk->gk", steer, cross)) / max(1, band.sum())


def gcc_phat_doa(x_left: np.ndarray, x_right: np.ndarray, sr: int, d: float = EAR_SPACING,
                 interp: int = 8) -> float:
    """Single-source broadband DOA (deg, lateral) from a block of two-channel audio."""
    n = len(x_left) + len(x_right)
    XL, XR = np.fft.rfft(x_left, n=n), np.fft.rfft(x_right, n=n)
    G = XR * XL.conj()
    G /= np.abs(G) + 1e-12
    cc = np.fft.irfft(G, n=n * interp)
    max_shift = int(interp * sr * d / SPEED_OF_SOUND) + 1
    cc = np.concatenate([cc[-max_shift:], cc[: max_shift + 1]])
    lag = (np.argmax(np.abs(cc)) - max_shift) / float(interp * sr)
    s = np.clip(lag * SPEED_OF_SOUND / d, -1.0, 1.0)
    return float(np.rad2deg(np.arcsin(s)))

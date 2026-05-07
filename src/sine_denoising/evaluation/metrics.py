"""Evaluation metrics for the denoising task.

All inputs are numpy arrays. See `docs/PRD_models.md` §2.2 for the metric
definitions used in the lab report.
"""
from __future__ import annotations

import math

import numpy as np


def mse_overall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error across all entries (mean over batch and time)."""
    return float(np.mean((y_pred - y_true) ** 2))


def mse_per_freq(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    freq_idx: np.ndarray,
    num_freqs: int,
) -> dict[int, float]:
    """MSE grouped by `freq_idx`. Returns one entry per non-empty group.

    Raises `ValueError` on `num_freqs <= 0` or any `freq_idx` outside
    `[0, num_freqs)`.
    """
    if num_freqs <= 0:
        raise ValueError(f"num_freqs must be positive; got {num_freqs}")
    fi = np.asarray(freq_idx).ravel()
    if fi.size and (int(fi.min()) < 0 or int(fi.max()) >= num_freqs):
        raise ValueError(
            f"freq_idx out of range [0, {num_freqs}); "
            f"got [{int(fi.min())}, {int(fi.max())}]"
        )
    result: dict[int, float] = {}
    for k in range(num_freqs):
        mask = fi == k
        if mask.any():
            result[int(k)] = float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
    return result


def mse_per_sigma(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sigma: np.ndarray,
) -> dict[float, float]:
    """MSE grouped by sigma value. `sigma` may be `(N,)` or `(N, 1)`."""
    sf = np.asarray(sigma).ravel()
    result: dict[float, float] = {}
    for s in np.unique(sf):
        mask = sf == s
        result[float(s)] = float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
    return result


def snr_improvement_db(
    clean: np.ndarray,
    noisy: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Signal-to-noise improvement in dB.

    `improvement_dB = 10 * log10(MSE(clean, noisy) / MSE(clean, reconstructed))`.

    Edge cases:
        - returns `inf` when `MSE(clean, reconstructed) == 0` (perfect),
        - returns `0.0` when `MSE(clean, noisy) == 0` (input was already clean).
    """
    noisy_err = float(np.mean((clean - noisy) ** 2))
    rec_err = float(np.mean((clean - reconstructed) ** 2))
    if rec_err == 0.0:
        return float("inf")
    if noisy_err == 0.0:
        return 0.0
    return 10.0 * math.log10(noisy_err / rec_err)

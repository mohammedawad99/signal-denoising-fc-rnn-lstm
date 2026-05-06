"""Pure-numpy generation of clean and noisy sine signals.

Mirrors §2 of `docs/PRD_dataset.md`:
    clean(t) = A * sin(2*pi*f*t + phase)
    noisy(t) = clean(t) + epsilon,   epsilon ~ N(0, sigma * A)
"""
from __future__ import annotations

import numpy as np


def make_signal(
    frequency: float,
    sigma: float,
    fs: float,
    duration: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean sinusoid and its additive-Gaussian-noisy counterpart.

    Parameters
    ----------
    frequency : signal frequency in Hz; must be positive.
    sigma : noise standard deviation as a fraction of `amplitude`; must be >= 0.
    fs : sampling rate in Hz; must satisfy `fs > 2 * frequency` (Nyquist).
    duration : signal length in seconds; must be positive.
    amplitude : signal amplitude `A` (default 1.0).
    phase : initial phase in radians (default 0.0).
    rng : numpy `Generator` for the noise draw. If `None`, a fresh
          `np.random.default_rng()` is created — non-reproducible.

    Returns
    -------
    (clean, noisy) : a pair of `float32` arrays, each of shape
        `(round(fs * duration),)`. When `sigma == 0`, `noisy == clean`.
    """
    if frequency <= 0:
        raise ValueError(f"frequency must be positive; got {frequency}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative; got {sigma}")
    if fs <= 0:
        raise ValueError(f"fs must be positive; got {fs}")
    if duration <= 0:
        raise ValueError(f"duration must be positive; got {duration}")
    if fs <= 2 * frequency:
        raise ValueError(
            f"fs={fs} violates Nyquist for frequency={frequency} Hz; "
            "need fs > 2 * frequency."
        )

    n_samples = int(round(fs * duration))
    t = np.arange(n_samples, dtype=np.float64) / fs
    clean = amplitude * np.sin(2.0 * np.pi * frequency * t + phase)

    if sigma == 0.0:
        noisy = clean.copy()
    else:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(loc=0.0, scale=sigma * amplitude, size=n_samples)
        noisy = clean + noise

    return clean.astype(np.float32), noisy.astype(np.float32)

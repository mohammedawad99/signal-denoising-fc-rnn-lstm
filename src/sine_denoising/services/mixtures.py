"""Build a noisy mixture realisation and the matching clean component windows.

For one realisation we draw an independent random phase per frequency, generate
a clean component plus an independent noise trace per frequency, and sum the
noisy components into one combined mixture. Both the mixture and every clean
component are returned as non-overlapping windows of length `window_size`.
"""
from __future__ import annotations

import numpy as np

from sine_denoising.services.encoding import non_overlapping_windows
from sine_denoising.services.signal import make_signal


def make_mixture_realisation(
    frequencies: list[float],
    sigma: float,
    fs: float,
    duration: float,
    window_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one mixture realisation and its component windows.

    Returns
    -------
    mixture_windows : `(num_windows, window_size)` float32
        Non-overlapping windows of the noisy mixture (sum of noisy components).
    clean_windows : `(K, num_windows, window_size)` float32
        For each frequency `k`, the matching clean (noise-free) component cut
        into the same windows.
    """
    component_clean: list[np.ndarray] = []
    component_noisy: list[np.ndarray] = []
    for f in frequencies:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        clean, noisy = make_signal(
            frequency=f,
            sigma=sigma,
            fs=fs,
            duration=duration,
            phase=phase,
            rng=rng,
        )
        component_clean.append(clean)
        component_noisy.append(noisy)

    mixture = np.sum(np.stack(component_noisy, axis=0), axis=0)
    mixture_windows = non_overlapping_windows(mixture, window_size).astype(np.float32)
    clean_windows = np.stack(
        [non_overlapping_windows(c, window_size) for c in component_clean],
        axis=0,
    ).astype(np.float32)
    return mixture_windows, clean_windows

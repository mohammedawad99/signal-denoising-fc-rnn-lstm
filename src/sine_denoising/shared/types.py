"""Lightweight record types for split-array bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitArrays:
    """Arrays for a single split (train, val, or test).

    Shapes (N is the number of windows in the split, K = num frequencies,
    T = window length):
        C:              (N, K)    one-hot frequency
        sigma:          (N, 1)    noise level for each window
        x_noisy:        (N, T)    noisy input window
        y_clean:        (N, T)    clean target window
        freq_idx:       (N,)      integer index of the frequency (bookkeeping)
        realisation_id: (N,)      integer id of the source realisation
    """

    C: np.ndarray
    sigma: np.ndarray
    x_noisy: np.ndarray
    y_clean: np.ndarray
    freq_idx: np.ndarray
    realisation_id: np.ndarray

    def __len__(self) -> int:
        return int(self.x_noisy.shape[0])


@dataclass(frozen=True)
class Splits:
    """Container of the three splits produced by the dataset builder."""

    train: SplitArrays
    val: SplitArrays
    test: SplitArrays

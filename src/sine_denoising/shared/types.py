"""Lightweight record types for split-array bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitArrays:
    """Arrays for a single split (train, val, or test).

    Shapes (N is the number of records in the split, K = num frequencies,
    T = window length):
        C:              (N, K)    one-hot query: which component to reconstruct
        sigma:          (N, 1)    noise level used per component for the mixture
        x_noisy:        (N, T)    window of the noisy mixture
        y_clean:        (N, T)    window of the clean component selected by C
        freq_idx:       (N,)      query frequency index (= argmax C); bookkeeping
        realisation_id: (N,)      integer id of the source mixture realisation
        window_idx:     (N,)      index of the non-overlapping window inside the
                                  mixture realisation (bookkeeping)
    """

    C: np.ndarray
    sigma: np.ndarray
    x_noisy: np.ndarray
    y_clean: np.ndarray
    freq_idx: np.ndarray
    realisation_id: np.ndarray
    window_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.x_noisy.shape[0])


@dataclass(frozen=True)
class Splits:
    """Container of the three splits produced by the dataset builder."""

    train: SplitArrays
    val: SplitArrays
    test: SplitArrays

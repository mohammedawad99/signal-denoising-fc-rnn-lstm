"""Tests for the SplitArrays / Splits record types."""
from __future__ import annotations

import numpy as np

from sine_denoising.shared.types import Splits, SplitArrays


def _make(n: int, k: int = 4, t: int = 10) -> SplitArrays:
    return SplitArrays(
        C=np.zeros((n, k), dtype=np.float32),
        sigma=np.zeros((n, 1), dtype=np.float32),
        x_noisy=np.zeros((n, t), dtype=np.float32),
        y_clean=np.zeros((n, t), dtype=np.float32),
        freq_idx=np.zeros((n,), dtype=np.int8),
        realisation_id=np.zeros((n,), dtype=np.int32),
    )


def test_split_arrays_len_matches_window_count() -> None:
    s = _make(5)
    assert len(s) == 5


def test_splits_holds_three_partitions() -> None:
    splits = Splits(train=_make(7), val=_make(2), test=_make(2))
    assert len(splits.train) == 7
    assert len(splits.val) == 2
    assert len(splits.test) == 2

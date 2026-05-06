"""Tests for `services.encoding`."""
from __future__ import annotations

import numpy as np
import pytest

from sine_denoising.services.encoding import non_overlapping_windows, one_hot


def test_one_hot_shape_dtype_and_values() -> None:
    v = one_hot(2, 4)
    assert v.shape == (4,)
    assert v.dtype == np.float32
    assert v.tolist() == [0.0, 0.0, 1.0, 0.0]


def test_one_hot_sums_to_one_for_every_index() -> None:
    for i in range(5):
        v = one_hot(i, 5)
        assert v.sum() == 1.0
        assert int(v.argmax()) == i


def test_one_hot_negative_index_raises() -> None:
    with pytest.raises(ValueError, match="index"):
        one_hot(-1, 4)


def test_one_hot_index_too_large_raises() -> None:
    with pytest.raises(ValueError, match="index"):
        one_hot(4, 4)


def test_one_hot_zero_size_raises() -> None:
    with pytest.raises(ValueError, match="size"):
        one_hot(0, 0)


def test_non_overlapping_windows_shape() -> None:
    arr = np.arange(50)
    w = non_overlapping_windows(arr, 10)
    assert w.shape == (5, 10)


def test_non_overlapping_windows_content() -> None:
    arr = np.arange(20)
    w = non_overlapping_windows(arr, 5)
    assert w[0].tolist() == [0, 1, 2, 3, 4]
    assert w[3].tolist() == [15, 16, 17, 18, 19]


def test_non_overlapping_windows_discards_leftover() -> None:
    arr = np.arange(23)
    w = non_overlapping_windows(arr, 10)
    assert w.shape == (2, 10)
    # Last sample retained is index 19; samples 20..22 are dropped.
    assert int(w[-1, -1]) == 19


def test_non_overlapping_windows_empty_when_too_short() -> None:
    arr = np.arange(3)
    w = non_overlapping_windows(arr, 10)
    assert w.shape == (0, 10)
    assert w.dtype == arr.dtype


def test_non_overlapping_windows_preserves_dtype() -> None:
    arr = np.arange(20, dtype=np.float32)
    w = non_overlapping_windows(arr, 5)
    assert w.dtype == np.float32


def test_non_overlapping_windows_rejects_2d() -> None:
    arr = np.zeros((5, 5))
    with pytest.raises(ValueError, match="1-D"):
        non_overlapping_windows(arr, 5)


def test_non_overlapping_windows_rejects_zero_window() -> None:
    arr = np.arange(10)
    with pytest.raises(ValueError, match="window_size"):
        non_overlapping_windows(arr, 0)

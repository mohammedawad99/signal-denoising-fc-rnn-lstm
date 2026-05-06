"""One-hot encoding and non-overlapping window slicing helpers."""
from __future__ import annotations

import numpy as np


def one_hot(index: int, size: int) -> np.ndarray:
    """Return a `(size,)` one-hot vector with `1.0` at position `index`.

    Raises `ValueError` when `size <= 0` or `index` is outside `[0, size)`.
    The output dtype is `float32`.
    """
    if size <= 0:
        raise ValueError(f"size must be positive; got {size}")
    if not 0 <= index < size:
        raise ValueError(f"index must be in [0, {size}); got {index}")
    v = np.zeros(size, dtype=np.float32)
    v[index] = 1.0
    return v


def non_overlapping_windows(array: np.ndarray, window_size: int) -> np.ndarray:
    """Slice a 1-D array into non-overlapping windows of length `window_size`.

    Leftover samples that do not fill a full window are discarded.
    Returns shape `(num_windows, window_size)`. When the input is shorter
    than one window, returns an empty array of shape `(0, window_size)`.
    """
    if array.ndim != 1:
        raise ValueError(f"expected 1-D array; got ndim={array.ndim}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive; got {window_size}")
    n = int(array.shape[0])
    num_windows = n // window_size
    if num_windows == 0:
        return np.empty((0, window_size), dtype=array.dtype)
    trimmed = array[: num_windows * window_size]
    return trimmed.reshape(num_windows, window_size)

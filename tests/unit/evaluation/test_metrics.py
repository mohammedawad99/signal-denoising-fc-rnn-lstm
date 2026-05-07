"""Tests for `evaluation.metrics`."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sine_denoising.evaluation.metrics import (
    mse_overall,
    mse_per_freq,
    mse_per_sigma,
    snr_improvement_db,
)


def test_mse_overall_known_value() -> None:
    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_true = np.array([[1.0, 2.0], [3.0, 5.0]])
    # Errors squared: 0, 0, 0, 1 -> MSE = 0.25.
    assert math.isclose(mse_overall(y_pred, y_true), 0.25)


def test_mse_overall_zero_when_arrays_equal() -> None:
    y = np.zeros((3, 4))
    assert mse_overall(y, y) == 0.0


def test_mse_per_freq_groups_by_index() -> None:
    y_pred = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_true = np.array([[2.0], [2.0], [3.0], [5.0]])
    freq_idx = np.array([0, 0, 1, 1])
    # Group 0: errors 1, 0 -> MSE 0.5.  Group 1: errors 0, 1 -> MSE 0.5.
    assert mse_per_freq(y_pred, y_true, freq_idx, num_freqs=2) == {0: 0.5, 1: 0.5}


def test_mse_per_freq_skips_empty_groups() -> None:
    y_pred = np.zeros((2, 3))
    y_true = np.zeros((2, 3))
    freq_idx = np.array([0, 0])
    assert mse_per_freq(y_pred, y_true, freq_idx, num_freqs=4) == {0: 0.0}


def test_mse_per_freq_rejects_non_positive_num_freqs() -> None:
    with pytest.raises(ValueError, match="num_freqs"):
        mse_per_freq(np.zeros((1, 1)), np.zeros((1, 1)), np.array([0]), num_freqs=0)


def test_mse_per_freq_rejects_out_of_range_index() -> None:
    y_pred = np.zeros((2, 3))
    y_true = np.zeros((2, 3))
    freq_idx = np.array([0, 5])
    with pytest.raises(ValueError, match="out of range"):
        mse_per_freq(y_pred, y_true, freq_idx, num_freqs=4)


def test_mse_per_sigma_handles_2d_sigma() -> None:
    y_pred = np.array([[1.0], [2.0], [3.0]])
    y_true = np.array([[2.0], [2.0], [4.0]])
    sigma_2d = np.array([[0.1], [0.2], [0.1]])
    result = mse_per_sigma(y_pred, y_true, sigma_2d)
    # sigma=0.1: errors 1, 1 -> MSE 1.0.  sigma=0.2: error 0 -> MSE 0.0.
    assert math.isclose(result[0.1], 1.0)
    assert math.isclose(result[0.2], 0.0)


def test_mse_per_sigma_handles_1d_sigma() -> None:
    y_pred = np.array([0.0, 1.0])
    y_true = np.array([0.0, 0.0])
    sigma_1d = np.array([0.1, 0.1])
    assert math.isclose(mse_per_sigma(y_pred, y_true, sigma_1d)[0.1], 0.5)


def test_snr_improvement_positive_when_reconstruction_better() -> None:
    clean = np.array([[1.0, 2.0]])
    noisy = np.array([[2.0, 3.0]])           # error of 1 each
    reconstructed = np.array([[1.1, 2.1]])   # error of 0.1 each
    # noisy_err=1.0, rec_err=0.01, ratio=100, 10*log10(100)=20 dB.
    assert math.isclose(
        snr_improvement_db(clean, noisy, reconstructed), 20.0, abs_tol=1e-6
    )


def test_snr_improvement_negative_when_reconstruction_worse() -> None:
    clean = np.zeros(3)
    noisy = np.array([0.1, 0.1, 0.1])
    reconstructed = np.array([1.0, 1.0, 1.0])
    assert snr_improvement_db(clean, noisy, reconstructed) < 0


def test_snr_improvement_zero_when_input_was_already_clean() -> None:
    clean = np.zeros(3)
    noisy = np.zeros(3)
    rec = np.array([0.0, 0.0, 1.0])
    assert snr_improvement_db(clean, noisy, rec) == 0.0


def test_snr_improvement_inf_on_perfect_reconstruction() -> None:
    clean = np.array([1.0, 2.0])
    noisy = np.array([1.5, 2.5])
    reconstructed = np.array([1.0, 2.0])
    assert snr_improvement_db(clean, noisy, reconstructed) == float("inf")

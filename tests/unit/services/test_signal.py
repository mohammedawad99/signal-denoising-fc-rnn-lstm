"""Tests for `services.signal.make_signal`."""
from __future__ import annotations

import numpy as np
import pytest

from sine_denoising.services.signal import make_signal


def test_shapes_match_duration_and_fs() -> None:
    clean, noisy = make_signal(
        frequency=1.0, sigma=0.1, fs=50.0, duration=10.0,
        rng=np.random.default_rng(0),
    )
    assert clean.shape == (500,)
    assert noisy.shape == (500,)
    assert clean.dtype == np.float32
    assert noisy.dtype == np.float32


def test_sigma_zero_is_noiseless() -> None:
    clean, noisy = make_signal(
        frequency=2.0, sigma=0.0, fs=50.0, duration=2.0,
    )
    assert np.array_equal(clean, noisy)


def test_deterministic_with_fixed_rng() -> None:
    a_clean, a_noisy = make_signal(
        frequency=5.0, sigma=0.2, fs=50.0, duration=4.0,
        rng=np.random.default_rng(42),
    )
    b_clean, b_noisy = make_signal(
        frequency=5.0, sigma=0.2, fs=50.0, duration=4.0,
        rng=np.random.default_rng(42),
    )
    assert np.array_equal(a_clean, b_clean)
    assert np.array_equal(a_noisy, b_noisy)


def test_noise_std_approximately_matches_sigma() -> None:
    sigma = 0.2
    amplitude = 1.0
    clean, noisy = make_signal(
        frequency=1.0, sigma=sigma, fs=200.0, duration=50.0,
        amplitude=amplitude, rng=np.random.default_rng(1),
    )
    residual = (noisy - clean).astype(np.float64)
    assert np.isclose(residual.std(), sigma * amplitude, rtol=0.05)


def test_amplitude_scales_signal() -> None:
    clean, _ = make_signal(
        frequency=1.0, sigma=0.0, fs=100.0, duration=1.0, amplitude=2.5,
    )
    assert np.isclose(clean.max(), 2.5, atol=1e-3)
    assert np.isclose(clean.min(), -2.5, atol=1e-3)


def test_phase_shift_changes_signal() -> None:
    a, _ = make_signal(
        frequency=1.0, sigma=0.0, fs=100.0, duration=1.0, phase=0.0,
    )
    b, _ = make_signal(
        frequency=1.0, sigma=0.0, fs=100.0, duration=1.0, phase=np.pi / 2,
    )
    assert not np.array_equal(a, b)


def test_invalid_frequency_raises() -> None:
    with pytest.raises(ValueError, match="frequency"):
        make_signal(frequency=0.0, sigma=0.1, fs=50.0, duration=1.0)


def test_negative_sigma_raises() -> None:
    with pytest.raises(ValueError, match="sigma"):
        make_signal(frequency=1.0, sigma=-0.1, fs=50.0, duration=1.0)


def test_sub_nyquist_raises() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        make_signal(frequency=10.0, sigma=0.1, fs=15.0, duration=1.0)


def test_invalid_fs_raises() -> None:
    with pytest.raises(ValueError, match="fs"):
        make_signal(frequency=1.0, sigma=0.1, fs=0.0, duration=1.0)


def test_invalid_duration_raises() -> None:
    with pytest.raises(ValueError, match="duration"):
        make_signal(frequency=1.0, sigma=0.1, fs=50.0, duration=0.0)


def test_default_rng_path_runs() -> None:
    # Exercises the branch where the caller passes no rng.
    clean, noisy = make_signal(
        frequency=1.0, sigma=0.1, fs=50.0, duration=1.0,
    )
    assert clean.shape == (50,)
    assert noisy.shape == (50,)

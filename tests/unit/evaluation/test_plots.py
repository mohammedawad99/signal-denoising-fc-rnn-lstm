"""Tests for `evaluation.plots.plot_reconstruction`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sine_denoising.evaluation.plots import plot_reconstruction


def _signals(T: int = 10) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(T)
    clean = np.sin(0.5 * t)
    noisy = clean + 0.1 * np.cos(0.7 * t)
    return clean, noisy


def test_plot_reconstruction_writes_non_empty_png(tmp_path: Path) -> None:
    clean, noisy = _signals()
    pred = clean + 0.05
    out = tmp_path / "out.png"
    written = plot_reconstruction(clean, noisy, {"fc": pred}, out)
    assert written == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_reconstruction_creates_parent_dirs(tmp_path: Path) -> None:
    clean, noisy = _signals()
    out = tmp_path / "nested" / "deep" / "fig.png"
    plot_reconstruction(clean, noisy, {"lstm": clean}, out)
    assert out.exists()


def test_plot_reconstruction_supports_optional_title(tmp_path: Path) -> None:
    clean, noisy = _signals()
    out = tmp_path / "titled.png"
    written = plot_reconstruction(
        clean, noisy, {"rnn": clean}, out, title="Test plot",
    )
    assert written.exists()


def test_plot_reconstruction_supports_multiple_models(tmp_path: Path) -> None:
    clean, noisy = _signals()
    out = tmp_path / "multi.png"
    plot_reconstruction(
        clean,
        noisy,
        {"fc": clean, "rnn": noisy, "lstm": clean + 0.01},
        out,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_reconstruction_supports_zero_models(tmp_path: Path) -> None:
    clean, noisy = _signals()
    out = tmp_path / "no_models.png"
    plot_reconstruction(clean, noisy, {}, out)
    assert out.exists()


def test_plot_reconstruction_rejects_non_1d_clean(tmp_path: Path) -> None:
    out = tmp_path / "x.png"
    bad_clean = np.zeros((2, 10))
    good = np.zeros(10)
    with pytest.raises(ValueError, match="clean must be 1-D"):
        plot_reconstruction(bad_clean, good, {"fc": good}, out)


def test_plot_reconstruction_rejects_clean_noisy_shape_mismatch(
    tmp_path: Path,
) -> None:
    out = tmp_path / "x.png"
    clean = np.zeros(10)
    noisy = np.zeros(8)
    with pytest.raises(ValueError, match="noisy shape"):
        plot_reconstruction(clean, noisy, {"fc": clean}, out)


def test_plot_reconstruction_rejects_prediction_shape_mismatch(
    tmp_path: Path,
) -> None:
    out = tmp_path / "x.png"
    clean = np.zeros(10)
    noisy = np.zeros(10)
    bad = np.zeros(8)
    with pytest.raises(ValueError, match="prediction 'fc'"):
        plot_reconstruction(clean, noisy, {"fc": bad}, out)

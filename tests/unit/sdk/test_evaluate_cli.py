"""Tests for the `sdk.evaluate` CLI."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sine_denoising.models.fc import FCDenoiser
from sine_denoising.models.lstm import LSTMDenoiser
from sine_denoising.models.rnn import RNNDenoiser
from sine_denoising.sdk.evaluate import main
from sine_denoising.services.dataset_builder import build_dataset
from sine_denoising.shared.config import DatasetConfig


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build a tiny dataset and three randomly-initialised checkpoints."""
    cfg = DatasetConfig(
        frequencies=[1.0, 2.0],
        sigmas=[0.1, 0.2],
        fs=20.0,
        duration=1.0,
        window_size=10,
        n_realisations=10,
        seed=0,
    )
    data_dir = tmp_path / "data"
    build_dataset(cfg, output_dir=data_dir)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    K, T = 2, 10
    torch.manual_seed(0)
    torch.save(
        FCDenoiser(window_size=T, num_classes=K).state_dict(),
        ckpt_dir / "fc_best.pt",
    )
    torch.save(
        RNNDenoiser(num_classes=K).state_dict(), ckpt_dir / "rnn_best.pt",
    )
    torch.save(
        LSTMDenoiser(num_classes=K).state_dict(), ckpt_dir / "lstm_best.pt",
    )
    summary_path = tmp_path / "results" / "summary.json"
    plot_path = tmp_path / "assets" / "reconstruction_example.png"
    return data_dir / "dataset.npz", ckpt_dir, summary_path, plot_path


def test_evaluate_cli_with_default_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds_path, ckpt_dir, summary_path, plot_path = _setup(tmp_path)
    rc = main([
        "--dataset-path", str(ds_path),
        "--checkpoint-dir", str(ckpt_dir),
        "--summary-path", str(summary_path),
        "--plot-path", str(plot_path),
    ])
    assert rc == 0
    assert summary_path.exists()
    assert plot_path.exists()
    out = capsys.readouterr().out
    assert str(summary_path) in out
    assert str(plot_path) in out
    assert "fc: mse_overall" in out
    assert "rnn: mse_overall" in out
    assert "lstm: mse_overall" in out


def test_evaluate_cli_subset_with_fc_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds_path, ckpt_dir, summary_path, plot_path = _setup(tmp_path)
    rc = main([
        "--dataset-path", str(ds_path),
        "--checkpoint-dir", str(ckpt_dir),
        "--summary-path", str(summary_path),
        "--plot-path", str(plot_path),
        "--models", "fc",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fc: mse_overall" in out
    assert "Models evaluated: fc" in out
    assert "rnn: mse_overall" not in out
    assert "lstm: mse_overall" not in out


def test_evaluate_cli_rejects_unknown_model(tmp_path: Path) -> None:
    ds_path, ckpt_dir, summary_path, plot_path = _setup(tmp_path)
    with pytest.raises(SystemExit):
        main([
            "--dataset-path", str(ds_path),
            "--checkpoint-dir", str(ckpt_dir),
            "--summary-path", str(summary_path),
            "--plot-path", str(plot_path),
            "--models", "transformer",
        ])

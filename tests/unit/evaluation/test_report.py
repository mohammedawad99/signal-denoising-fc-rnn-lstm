"""Tests for `evaluation.report.generate_report`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sine_denoising.evaluation.report import generate_report
from sine_denoising.models.fc import FCDenoiser
from sine_denoising.models.lstm import LSTMDenoiser
from sine_denoising.models.rnn import RNNDenoiser
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


def test_generate_report_writes_summary_and_plot(tmp_path: Path) -> None:
    ds_path, ckpt_dir, summary_path, plot_path = _setup(tmp_path)
    summary = generate_report(ds_path, ckpt_dir, summary_path, plot_path)
    assert set(summary.keys()) == {"fc", "rnn", "lstm"}
    assert summary_path.exists()
    on_disk = json.loads(summary_path.read_text())
    assert set(on_disk.keys()) == {"fc", "rnn", "lstm"}
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_each_summary_entry_has_required_fields(tmp_path: Path) -> None:
    ds_path, ckpt_dir, summary_path, plot_path = _setup(tmp_path)
    summary = generate_report(ds_path, ckpt_dir, summary_path, plot_path)
    required = {
        "mse_overall",
        "mse_per_freq",
        "mse_per_sigma",
        "snr_improvement_db",
        "parameters",
    }
    for name in ("fc", "rnn", "lstm"):
        assert required.issubset(summary[name].keys())
        assert summary[name]["mse_overall"] >= 0.0
        assert summary[name]["parameters"] > 0


def test_missing_checkpoint_raises_file_not_found(tmp_path: Path) -> None:
    ds_path, ckpt_dir, summary_path, plot_path = _setup(tmp_path)
    (ckpt_dir / "rnn_best.pt").unlink()
    with pytest.raises(FileNotFoundError, match="rnn_best.pt"):
        generate_report(ds_path, ckpt_dir, summary_path, plot_path)


def test_can_subset_models(tmp_path: Path) -> None:
    ds_path, ckpt_dir, summary_path, plot_path = _setup(tmp_path)
    summary = generate_report(
        ds_path,
        ckpt_dir,
        summary_path,
        plot_path,
        model_names=("fc",),
    )
    assert set(summary.keys()) == {"fc"}
    assert plot_path.exists()

"""Tests for the `sdk.train` CLI."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sine_denoising.sdk.train import main
from sine_denoising.services.dataset_builder import build_dataset
from sine_denoising.shared.config import DatasetConfig


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a tiny dataset.npz and a tiny training YAML; return useful paths."""
    ds_cfg = DatasetConfig(
        frequencies=[1.0, 2.0],
        sigmas=[0.1, 0.2],
        fs=20.0,
        duration=1.0,
        window_size=10,
        n_realisations=10,
        seed=0,
    )
    data_dir = tmp_path / "data"
    build_dataset(ds_cfg, output_dir=data_dir)
    training_yaml = {
        "model": "fc",
        "batch_size": 8,
        "learning_rate": 0.001,
        "epochs": 1,
        "early_stopping_patience": 5,
        "grad_clip_norm": 1.0,
        "hidden_size": 16,
        "seed": 0,
        "device": "cpu",
        "results_dir": str(tmp_path / "results"),
    }
    cfg_path = tmp_path / "training.yaml"
    cfg_path.write_text(yaml.safe_dump(training_yaml))
    return cfg_path, data_dir / "dataset.npz", tmp_path / "checkpoints"


@pytest.mark.parametrize("model_name", ["fc", "rnn", "lstm"])
def test_train_cli_runs_one_epoch_per_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    model_name: str,
) -> None:
    cfg_path, ds_path, ckpt_dir = _setup(tmp_path)
    rc = main([
        "--model", model_name,
        "--training-config", str(cfg_path),
        "--dataset-path", str(ds_path),
        "--checkpoint-dir", str(ckpt_dir),
        "--epochs", "1",
    ])
    assert rc == 0
    ckpt = ckpt_dir / f"{model_name}_best.pt"
    assert ckpt.exists()
    out = capsys.readouterr().out
    assert model_name in out
    assert "Best val_loss" in out
    assert str(ckpt) in out


def test_train_cli_rejects_unknown_model(tmp_path: Path) -> None:
    cfg_path, ds_path, ckpt_dir = _setup(tmp_path)
    with pytest.raises(SystemExit):
        main([
            "--model", "transformer",
            "--training-config", str(cfg_path),
            "--dataset-path", str(ds_path),
            "--checkpoint-dir", str(ckpt_dir),
        ])


def test_train_cli_uses_config_epochs_when_no_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path, ds_path, ckpt_dir = _setup(tmp_path)
    rc = main([
        "--model", "fc",
        "--training-config", str(cfg_path),
        "--dataset-path", str(ds_path),
        "--checkpoint-dir", str(ckpt_dir),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # Tiny config sets epochs=1, so without --epochs override we still see 1.
    assert "Epochs run: 1" in out

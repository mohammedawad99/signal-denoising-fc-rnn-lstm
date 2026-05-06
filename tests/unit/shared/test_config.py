"""Tests for the pydantic config loaders."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sine_denoising.shared.config import DatasetConfig, TrainingConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_dataset_config_loads_from_yaml() -> None:
    cfg = DatasetConfig.from_yaml(REPO_ROOT / "config" / "dataset.yaml")
    assert cfg.frequencies == [1.0, 2.0, 5.0, 10.0]
    assert cfg.sigmas == [0.05, 0.10, 0.20, 0.30]
    assert cfg.fs == 50.0
    assert cfg.window_size == 10
    assert cfg.n_realisations == 25
    assert cfg.seed == 42
    assert cfg.train_ratio == 0.70
    assert cfg.val_ratio == 0.15
    assert cfg.test_ratio == 0.15


def test_dataset_config_rejects_bad_ratio_sum() -> None:
    with pytest.raises(ValidationError, match="split ratios"):
        DatasetConfig(
            frequencies=[1.0],
            sigmas=[0.1],
            fs=50.0,
            duration=10.0,
            window_size=10,
            n_realisations=1,
            seed=0,
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.3,
        )


def test_training_config_loads_from_yaml() -> None:
    cfg = TrainingConfig.from_yaml(REPO_ROOT / "config" / "training.yaml")
    assert cfg.model in {"fc", "rnn", "lstm"}
    assert cfg.batch_size > 0
    assert cfg.learning_rate > 0
    assert cfg.epochs > 0


def test_dataset_config_rejects_sub_nyquist(tmp_path: Path) -> None:
    bad = {
        "frequencies": [10.0],
        "sigmas": [0.1],
        "fs": 15.0,
        "duration": 10.0,
        "window_size": 10,
        "n_realisations": 1,
        "seed": 0,
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError, match="Nyquist"):
        DatasetConfig.from_yaml(p)


def test_dataset_config_rejects_indivisible_window(tmp_path: Path) -> None:
    bad = {
        "frequencies": [1.0],
        "sigmas": [0.1],
        "fs": 50.0,
        "duration": 10.0,
        "window_size": 7,  # 500 % 7 != 0
        "n_realisations": 1,
        "seed": 0,
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError, match="multiple"):
        DatasetConfig.from_yaml(p)


def test_dataset_config_rejects_negative_sigma() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        DatasetConfig(
            frequencies=[1.0, 2.0],
            sigmas=[0.1, -0.05],
            fs=50.0,
            duration=10.0,
            window_size=10,
            n_realisations=1,
            seed=0,
        )


def test_training_config_rejects_unknown_model() -> None:
    with pytest.raises(ValidationError, match="model must be one of"):
        TrainingConfig(
            model="transformer",
            batch_size=64,
            learning_rate=1e-3,
            epochs=1,
            early_stopping_patience=0,
            grad_clip_norm=1.0,
            hidden_size=32,
            seed=0,
        )


def test_dataset_config_is_frozen() -> None:
    cfg = DatasetConfig.from_yaml(REPO_ROOT / "config" / "dataset.yaml")
    with pytest.raises(ValidationError):
        cfg.fs = 100.0  # type: ignore[misc]

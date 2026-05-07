"""Tests for `training.trainer.Trainer`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sine_denoising.models.fc import FCDenoiser
from sine_denoising.services.dataset_loader import WindowDataset
from sine_denoising.shared.types import SplitArrays
from sine_denoising.training.trainer import Trainer, TrainingResult


def _make_loader(
    n: int = 8, seed: int = 0, batch_size: int = 4,
) -> DataLoader[dict[str, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    K, T = 4, 10
    split = SplitArrays(
        C=np.eye(K, dtype=np.float32)[rng.integers(0, K, size=n)],
        sigma=rng.uniform(0.05, 0.3, size=(n, 1)).astype(np.float32),
        x_noisy=rng.normal(size=(n, T)).astype(np.float32),
        y_clean=rng.normal(size=(n, T)).astype(np.float32),
        freq_idx=np.zeros(n, dtype=np.int8),
        realisation_id=np.arange(n, dtype=np.int32),
        window_idx=np.zeros(n, dtype=np.int16),
    )
    return DataLoader(WindowDataset(split), batch_size=batch_size, shuffle=False)


def _fresh_trainer(
    tmp_path: Path, epochs: int = 6, patience: int = 3, lr: float = 1e-2,
) -> Trainer:
    torch.manual_seed(0)
    model = FCDenoiser()
    return Trainer(
        model=model,
        train_loader=_make_loader(seed=0),
        val_loader=_make_loader(seed=100),
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        loss_fn=nn.MSELoss(),
        checkpoint_path=tmp_path / "ckpt.pt",
        epochs=epochs,
        early_stopping_patience=patience,
    )


def test_fit_returns_training_result(tmp_path: Path) -> None:
    result = _fresh_trainer(tmp_path).fit()
    assert isinstance(result, TrainingResult)
    assert result.epochs_run >= 1


def test_history_records_both_curves_per_epoch(tmp_path: Path) -> None:
    result = _fresh_trainer(tmp_path).fit()
    assert "train_loss" in result.history
    assert "val_loss" in result.history
    assert len(result.history["train_loss"]) == result.epochs_run
    assert len(result.history["val_loss"]) == result.epochs_run


def test_checkpoint_file_is_created(tmp_path: Path) -> None:
    result = _fresh_trainer(tmp_path).fit()
    assert result.checkpoint_path.exists()
    assert result.checkpoint_path.stat().st_size > 0


def test_load_best_checkpoint_restores_state(tmp_path: Path) -> None:
    trainer = _fresh_trainer(tmp_path)
    trainer.fit()
    # Build a reference model from the saved state.
    ref = FCDenoiser()
    ref.load_state_dict(
        torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=True)
    )
    x = torch.zeros(2, 10)
    C = torch.zeros(2, 4)
    sigma = torch.zeros(2, 1)
    with torch.no_grad():
        ref_out = ref(x, C, sigma)
    # Perturb the trainer's model, then restore from checkpoint.
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.add_(1.0)
    trainer.load_best_checkpoint()
    with torch.no_grad():
        restored_out = trainer.model(x, C, sigma)
    assert torch.allclose(ref_out, restored_out)


def test_early_stopping_triggers_when_val_diverges(tmp_path: Path) -> None:
    # Huge lr drives val loss up after the first step; early stopping fires.
    result = _fresh_trainer(tmp_path, epochs=20, patience=2, lr=1.0).fit()
    assert result.stopped_early is True
    assert result.epochs_run < 20


def test_best_epoch_within_run_range(tmp_path: Path) -> None:
    result = _fresh_trainer(tmp_path).fit()
    assert 1 <= result.best_epoch <= result.epochs_run
    assert result.best_val_loss < float("inf")

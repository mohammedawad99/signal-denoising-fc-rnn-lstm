"""Trainer with early stopping and best-checkpoint saving."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from sine_denoising.training.loops import (
    Device,
    LossFn,
    evaluate,
    train_one_epoch,
)


@dataclass
class TrainingResult:
    """Outcome of a `Trainer.fit()` run."""

    history: dict[str, list[float]]
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    checkpoint_path: Path


@dataclass
class Trainer:
    """Train a model with early stopping and best-checkpoint saving.

    The checkpoint stores `model.state_dict()` only (not the full module).
    Use `load_best_checkpoint()` to restore the best weights into `self.model`.
    """

    model: nn.Module
    train_loader: DataLoader[dict[str, torch.Tensor]]
    val_loader: DataLoader[dict[str, torch.Tensor]]
    optimizer: torch.optim.Optimizer
    loss_fn: LossFn
    checkpoint_path: Path
    device: Device = "cpu"
    epochs: int = 30
    early_stopping_patience: int = 5

    def fit(self) -> TrainingResult:
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_epoch = 0
        epochs_since_improvement = 0
        stopped_early = False
        self.model.to(self.device)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        epoch = 0
        for epoch in range(1, self.epochs + 1):
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.loss_fn,
                self.device,
            )
            val_loss = evaluate(
                self.model,
                self.val_loader,
                self.loss_fn,
                self.device,
            )
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                epochs_since_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= self.early_stopping_patience:
                    stopped_early = True
                    break
        return TrainingResult(
            history=history,
            best_val_loss=best_val,
            best_epoch=best_epoch,
            epochs_run=epoch,
            stopped_early=stopped_early,
            checkpoint_path=self.checkpoint_path,
        )

    def load_best_checkpoint(self) -> None:
        state = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state)

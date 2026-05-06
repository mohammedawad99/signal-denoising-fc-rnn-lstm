"""CLI entry point: train one of the FC/RNN/LSTM denoisers.

Run with:
    uv run python -m sine_denoising.sdk.train --model {fc,rnn,lstm}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from sine_denoising.models.base import DenoiserBase
from sine_denoising.models.fc import FCDenoiser
from sine_denoising.models.lstm import LSTMDenoiser
from sine_denoising.models.rnn import RNNDenoiser
from sine_denoising.services.dataset_loader import WindowDataset, load_dataset
from sine_denoising.shared.config import TrainingConfig
from sine_denoising.shared.seeding import set_seed
from sine_denoising.training.trainer import Trainer

DEFAULT_TRAINING_CONFIG = "config/training.yaml"
DEFAULT_DATASET_PATH = "data/generated/dataset.npz"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"
MODEL_CHOICES = ("fc", "rnn", "lstm")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Train one of the FC/RNN/LSTM denoisers.",
    )
    parser.add_argument(
        "--model", choices=MODEL_CHOICES, required=True,
        help="Which model to train.",
    )
    parser.add_argument(
        "--training-config", default=DEFAULT_TRAINING_CONFIG,
        help="Path to the training YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-path", default=DEFAULT_DATASET_PATH,
        help="Path to dataset.npz (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
        help="Where to save best checkpoints (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the epoch count from the config.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _build_model(
    name: str, hidden_size: int, num_classes: int, window_size: int,
) -> DenoiserBase:
    """Per PRD §3.2/§4.2/§5.2: FC keeps its default hidden_size (64);
    `cfg.hidden_size` (default 32) feeds the recurrent models."""
    if name == "fc":
        return FCDenoiser(window_size=window_size, num_classes=num_classes)
    if name == "rnn":
        return RNNDenoiser(num_classes=num_classes, hidden_size=hidden_size)
    if name == "lstm":
        return LSTMDenoiser(num_classes=num_classes, hidden_size=hidden_size)
    raise ValueError(f"unknown model name: {name!r}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = TrainingConfig.from_yaml(args.training_config)
    update: dict[str, Any] = {"model": args.model, "device": args.device}
    if args.epochs is not None:
        update["epochs"] = args.epochs
    cfg = cfg.model_copy(update=update)
    set_seed(cfg.seed)

    splits = load_dataset(args.dataset_path)
    num_classes = int(splits.train.C.shape[1])
    window_size = int(splits.train.x_noisy.shape[1])
    train_loader = DataLoader(
        WindowDataset(splits.train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WindowDataset(splits.val),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = _build_model(cfg.model, cfg.hidden_size, num_classes, window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    ckpt_path = Path(args.checkpoint_dir) / f"{cfg.model}_best.pt"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        checkpoint_path=ckpt_path,
        device=cfg.device,
        epochs=cfg.epochs,
        early_stopping_patience=cfg.early_stopping_patience,
    )
    result = trainer.fit()

    print(f"Model: {cfg.model}")
    print(f"Epochs run: {result.epochs_run}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best val_loss: {result.best_val_loss:.6f}")
    print(f"Checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

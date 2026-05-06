"""Training loop primitives: `train_one_epoch` and `evaluate`.

Both consume batches shaped as the dict produced by `WindowDataset`:
    x_noisy: (B, T)   y_clean: (B, T)
    C:       (B, K)   sigma:   (B, 1)
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Device = torch.device | str


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    device: Device = "cpu",
) -> float:
    """Run one training epoch; return the example-weighted average loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        x = batch["x_noisy"].to(device)
        C = batch["C"].to(device)
        sigma = batch["sigma"].to(device)
        y = batch["y_clean"].to(device)
        optimizer.zero_grad()
        y_pred = model(x, C, sigma)
        loss = loss_fn(y_pred, y)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        b = int(x.size(0))
        total_loss += float(loss.item()) * b
        total_examples += b
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    loss_fn: LossFn,
    device: Device = "cpu",
) -> float:
    """Compute the example-weighted average loss without gradient updates."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        x = batch["x_noisy"].to(device)
        C = batch["C"].to(device)
        sigma = batch["sigma"].to(device)
        y = batch["y_clean"].to(device)
        y_pred = model(x, C, sigma)
        loss = loss_fn(y_pred, y)
        b = int(x.size(0))
        total_loss += float(loss.item()) * b
        total_examples += b
    return total_loss / max(total_examples, 1)

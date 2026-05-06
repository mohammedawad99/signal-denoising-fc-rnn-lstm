"""Tests for `training.loops.train_one_epoch` and `training.loops.evaluate`."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sine_denoising.models.fc import FCDenoiser
from sine_denoising.services.dataset_loader import WindowDataset
from sine_denoising.shared.types import SplitArrays
from sine_denoising.training.loops import evaluate, train_one_epoch


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
    )
    return DataLoader(WindowDataset(split), batch_size=batch_size, shuffle=False)


def test_train_one_epoch_returns_positive_float() -> None:
    torch.manual_seed(0)
    model = FCDenoiser()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _make_loader()
    loss = train_one_epoch(model, loader, optim, nn.MSELoss())
    assert isinstance(loss, float)
    assert loss > 0


def test_evaluate_returns_positive_float() -> None:
    torch.manual_seed(0)
    model = FCDenoiser()
    loader = _make_loader()
    loss = evaluate(model, loader, nn.MSELoss())
    assert isinstance(loss, float)
    assert loss > 0


def test_train_updates_at_least_one_parameter() -> None:
    torch.manual_seed(0)
    model = FCDenoiser()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = _make_loader()
    before = [p.detach().clone() for p in model.parameters()]
    train_one_epoch(model, loader, optim, nn.MSELoss())
    after = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_evaluate_does_not_update_parameters() -> None:
    torch.manual_seed(0)
    model = FCDenoiser()
    loader = _make_loader()
    before = [p.detach().clone() for p in model.parameters()]
    evaluate(model, loader, nn.MSELoss())
    after = [p.detach().clone() for p in model.parameters()]
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a)


def test_overfitting_drives_loss_down_on_tiny_batch() -> None:
    torch.manual_seed(0)
    model = FCDenoiser()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = _make_loader(n=4, batch_size=4)
    losses: list[float] = []
    for _ in range(30):
        losses.append(train_one_epoch(model, loader, optim, nn.MSELoss()))
    assert losses[-1] < losses[0]
    # Final loss should be at least 50% lower than the starting loss.
    assert losses[-1] < 0.5 * losses[0]


def test_train_and_evaluate_on_empty_loader_return_zero() -> None:
    # Edge case: empty dataloader. Should return 0 instead of dividing by zero.
    torch.manual_seed(0)
    model = FCDenoiser()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    empty_loader = _make_loader(n=0, batch_size=4)
    assert train_one_epoch(model, empty_loader, optim, nn.MSELoss()) == 0.0
    assert evaluate(model, empty_loader, nn.MSELoss()) == 0.0

"""Tests specific to `FCDenoiser`."""
from __future__ import annotations

import torch
from torch import nn

from sine_denoising.models.fc import FCDenoiser


def test_fc_first_linear_in_features_equals_T_plus_K_plus_1() -> None:
    model = FCDenoiser(window_size=10, num_classes=4, hidden_size=8)
    first = next(m for m in model.net if isinstance(m, nn.Linear))
    assert first.in_features == 10 + 4 + 1


def test_fc_last_linear_out_features_equals_T() -> None:
    model = FCDenoiser(window_size=10, num_classes=4, hidden_size=8)
    last = [m for m in model.net if isinstance(m, nn.Linear)][-1]
    assert last.out_features == 10


def test_fc_param_count_matches_prd_estimate() -> None:
    # PRD §3.2 estimate: ~5,834 for the default 64-wide hidden layers.
    model = FCDenoiser()
    n = model.parameter_count()
    assert 5_500 <= n <= 6_500


def test_fc_smoke_forward() -> None:
    model = FCDenoiser()
    y = model(torch.zeros(2, 10), torch.zeros(2, 4), torch.zeros(2, 1))
    assert y.shape == (2, 10)

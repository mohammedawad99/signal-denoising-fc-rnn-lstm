"""Fully Connected denoiser. See `docs/PRD_models.md` §3."""
from __future__ import annotations

import torch
from torch import nn

from sine_denoising.models.base import DenoiserBase


class FCDenoiser(DenoiserBase):
    """MLP that maps `concat(x_noisy, C, sigma) -> y_clean`.

    Default architecture (from PRD §3.2):
        Linear(15, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 10)
    """

    def __init__(
        self,
        window_size: int = 10,
        num_classes: int = 4,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        in_dim = window_size + num_classes + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, window_size),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        C: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.cat([x_noisy, C, sigma], dim=-1)
        out: torch.Tensor = self.net(z)
        return out

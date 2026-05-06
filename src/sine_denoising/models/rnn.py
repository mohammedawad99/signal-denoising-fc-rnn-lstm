"""Vanilla RNN denoiser. See `docs/PRD_models.md` §4."""
from __future__ import annotations

import torch
from torch import nn

from sine_denoising.models.base import DenoiserBase, stack_per_timestep


class RNNDenoiser(DenoiserBase):
    """Elman RNN over per-timestep `[x_t, C, sigma]`, with a linear head.

    Default sizing (PRD §4.2):
        nn.RNN(input_size=6, hidden_size=32, num_layers=1, nonlinearity='tanh')
        Linear(32, 1) applied at every timestep.
    """

    def __init__(
        self,
        num_classes: int = 4,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        in_dim = 1 + num_classes + 1
        self.rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x_noisy: torch.Tensor,
        C: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        u = stack_per_timestep(x_noisy, C, sigma)
        out, _ = self.rnn(u)
        y: torch.Tensor = self.head(out)
        return y.squeeze(-1)

"""Common base class and helpers for FC/RNN/LSTM denoisers."""
from __future__ import annotations

import torch
from torch import nn


def parameter_count(model: nn.Module) -> int:
    """Total number of parameters (trainable + non-trainable) in `model`."""
    return sum(p.numel() for p in model.parameters())


def stack_per_timestep(
    x_noisy: torch.Tensor,
    C: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Build a `(B, T, 1 + K + 1)` tensor for sequence models.

    At timestep `t`, the per-step feature is `[x_noisy[t], C, sigma]` —
    `C` and `sigma` are broadcast across all `T` steps.

    Shapes
    ------
    x_noisy: (B, T)
    C:       (B, K)
    sigma:   (B, 1)
    return:  (B, T, 1 + K + 1)
    """
    if x_noisy.ndim != 2:
        raise ValueError(f"x_noisy must be 2-D (B, T); got shape {x_noisy.shape}")
    B, T = x_noisy.shape
    x_seq = x_noisy.unsqueeze(-1)
    C_seq = C.unsqueeze(1).expand(B, T, C.size(-1))
    sigma_seq = sigma.unsqueeze(1).expand(B, T, sigma.size(-1))
    return torch.cat([x_seq, C_seq, sigma_seq], dim=-1)


class DenoiserBase(nn.Module):
    """Common base for FC/RNN/LSTM denoising models.

    Subclasses must implement
        forward(x_noisy, C, sigma) -> y_pred
    with shapes
        x_noisy=(B, T), C=(B, K), sigma=(B, 1), y_pred=(B, T).
    """

    def parameter_count(self) -> int:
        return parameter_count(self)

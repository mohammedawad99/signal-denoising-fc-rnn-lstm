"""Cross-model invariants (forward shape, dtype, parameter_count)."""
from __future__ import annotations

import pytest
import torch

from sine_denoising.models import (
    DenoiserBase,
    FCDenoiser,
    LSTMDenoiser,
    RNNDenoiser,
    parameter_count,
)

MODEL_CLASSES: list[type[DenoiserBase]] = [FCDenoiser, RNNDenoiser, LSTMDenoiser]


@pytest.fixture
def dummy_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, K = 4, 10, 4
    return torch.zeros(B, T), torch.zeros(B, K), torch.zeros(B, 1)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_forward_output_shape_b4(
    model_cls: type[DenoiserBase],
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    model = model_cls()
    x, C, sigma = dummy_inputs
    y = model(x, C, sigma)
    assert y.shape == (4, 10)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_forward_output_dtype_float32(
    model_cls: type[DenoiserBase],
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    model = model_cls()
    x, C, sigma = dummy_inputs
    y = model(x, C, sigma)
    assert y.dtype == torch.float32


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_forward_supports_batch_one(
    model_cls: type[DenoiserBase],
) -> None:
    model = model_cls()
    x = torch.zeros(1, 10)
    C = torch.zeros(1, 4)
    sigma = torch.zeros(1, 1)
    y = model(x, C, sigma)
    assert y.shape == (1, 10)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_parameter_count_positive_and_consistent(
    model_cls: type[DenoiserBase],
) -> None:
    model = model_cls()
    n = model.parameter_count()
    assert n > 0
    assert parameter_count(model) == n


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_random_inputs_produce_finite_outputs(
    model_cls: type[DenoiserBase],
) -> None:
    torch.manual_seed(0)
    model = model_cls()
    x = torch.randn(3, 10)
    C = torch.zeros(3, 4)
    C[:, 0] = 1.0
    sigma = torch.full((3, 1), 0.1)
    y = model(x, C, sigma)
    assert torch.isfinite(y).all()

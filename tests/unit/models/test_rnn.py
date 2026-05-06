"""Tests specific to `RNNDenoiser`."""
from __future__ import annotations

import torch
from torch import nn

from sine_denoising.models.rnn import RNNDenoiser


def test_rnn_uses_nn_rnn_with_tanh() -> None:
    model = RNNDenoiser()
    assert isinstance(model.rnn, nn.RNN)
    assert model.rnn.nonlinearity == "tanh"
    assert model.rnn.batch_first is True


def test_rnn_input_size_is_one_plus_K_plus_one() -> None:
    model = RNNDenoiser(num_classes=4, hidden_size=32)
    assert model.rnn.input_size == 1 + 4 + 1
    assert model.rnn.hidden_size == 32


def test_rnn_param_count_in_expected_ballpark() -> None:
    # PRD §4.2 estimate: ~1,281; nn.RNN actually has two biases per layer
    # (bias_ih + bias_hh), so the real count sits a bit above that.
    model = RNNDenoiser()
    n = model.parameter_count()
    assert 1_000 <= n <= 2_000


def test_rnn_smoke_forward() -> None:
    model = RNNDenoiser()
    y = model(torch.zeros(2, 10), torch.zeros(2, 4), torch.zeros(2, 1))
    assert y.shape == (2, 10)

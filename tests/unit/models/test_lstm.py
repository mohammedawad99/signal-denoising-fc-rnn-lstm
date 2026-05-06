"""Tests specific to `LSTMDenoiser`."""
from __future__ import annotations

import torch
from torch import nn

from sine_denoising.models.lstm import LSTMDenoiser


def test_lstm_uses_nn_lstm() -> None:
    model = LSTMDenoiser()
    assert isinstance(model.lstm, nn.LSTM)
    assert model.lstm.batch_first is True


def test_lstm_input_size_is_one_plus_K_plus_one() -> None:
    model = LSTMDenoiser(num_classes=4, hidden_size=32)
    assert model.lstm.input_size == 1 + 4 + 1
    assert model.lstm.hidden_size == 32


def test_lstm_param_count_in_expected_ballpark() -> None:
    # PRD §5.2 estimate: ~5,025. nn.LSTM has 4× the gates of a vanilla RNN.
    model = LSTMDenoiser()
    n = model.parameter_count()
    assert 4_000 <= n <= 6_500


def test_lstm_smoke_forward() -> None:
    model = LSTMDenoiser()
    y = model(torch.zeros(2, 10), torch.zeros(2, 4), torch.zeros(2, 1))
    assert y.shape == (2, 10)

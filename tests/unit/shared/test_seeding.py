"""Tests for the reproducible seeding helper."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sine_denoising.shared.seeding import set_seed


def test_seed_makes_numpy_deterministic() -> None:
    set_seed(123)
    a = np.random.rand(10)
    set_seed(123)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_seed_makes_torch_deterministic() -> None:
    set_seed(7)
    a = torch.randn(8)
    set_seed(7)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_different_seeds_diverge() -> None:
    set_seed(1)
    a = np.random.rand(8)
    set_seed(2)
    b = np.random.rand(8)
    assert not np.array_equal(a, b)


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)

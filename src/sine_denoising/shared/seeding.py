"""Reproducible RNG seeding for numpy, Python random, and torch."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every source of randomness used in this project.

    Seeds Python's `random`, NumPy, and PyTorch (CPU + CUDA when available),
    and exports `PYTHONHASHSEED` for completeness.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative; got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""Load the persisted dataset from `.npz` and offer a torch `Dataset` wrapper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sine_denoising.shared.types import SplitArrays, Splits

SPLIT_NAMES = ("train", "val", "test")


def load_dataset(path: str | Path) -> Splits:
    """Load a `dataset.npz` produced by `build_dataset` into a `Splits`."""
    data = np.load(Path(path))
    split = data["split"]
    out: dict[str, SplitArrays] = {}
    for name in SPLIT_NAMES:
        mask = split == name
        out[name] = SplitArrays(
            C=data["C"][mask],
            sigma=data["sigma"][mask],
            x_noisy=data["x_noisy"][mask],
            y_clean=data["y_clean"][mask],
            freq_idx=data["freq_idx"][mask],
            realisation_id=data["realisation_id"][mask],
        )
    return Splits(train=out["train"], val=out["val"], test=out["test"])


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch `Dataset` wrapper over a single `SplitArrays`.

    Each item is a dict with keys `x_noisy`, `C`, `sigma`, `y_clean`. Shapes
    per item: `x_noisy` and `y_clean` are `(T,)`, `C` is `(K,)`, `sigma`
    is `(1,)`.
    """

    def __init__(self, split: SplitArrays) -> None:
        self._split = split

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_noisy": torch.from_numpy(self._split.x_noisy[idx]),
            "C": torch.from_numpy(self._split.C[idx]),
            "sigma": torch.from_numpy(self._split.sigma[idx]),
            "y_clean": torch.from_numpy(self._split.y_clean[idx]),
        }

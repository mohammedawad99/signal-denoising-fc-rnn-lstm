"""Tests for `services.dataset_loader`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sine_denoising.services.dataset_builder import build_dataset
from sine_denoising.services.dataset_loader import WindowDataset, load_dataset
from sine_denoising.shared.config import DatasetConfig


@pytest.fixture
def small_cfg() -> DatasetConfig:
    return DatasetConfig(
        frequencies=[1.0, 2.0],
        sigmas=[0.1, 0.2],
        fs=20.0,
        duration=1.0,
        window_size=10,
        n_realisations=20,
        seed=0,
    )


def test_load_dataset_round_trip(
    tmp_path: Path, small_cfg: DatasetConfig
) -> None:
    in_memory = build_dataset(small_cfg, output_dir=tmp_path)
    on_disk = load_dataset(tmp_path / "dataset.npz")
    assert len(in_memory.train) == len(on_disk.train)
    assert len(in_memory.val) == len(on_disk.val)
    assert len(in_memory.test) == len(on_disk.test)
    np.testing.assert_array_equal(in_memory.train.x_noisy, on_disk.train.x_noisy)
    np.testing.assert_array_equal(in_memory.train.y_clean, on_disk.train.y_clean)
    np.testing.assert_array_equal(in_memory.train.C, on_disk.train.C)
    np.testing.assert_array_equal(
        in_memory.train.realisation_id, on_disk.train.realisation_id
    )


def test_load_dataset_shapes(
    tmp_path: Path, small_cfg: DatasetConfig
) -> None:
    build_dataset(small_cfg, output_dir=tmp_path)
    splits = load_dataset(tmp_path / "dataset.npz")
    K = len(small_cfg.frequencies)
    T = small_cfg.window_size
    n = len(splits.train)
    assert splits.train.x_noisy.shape == (n, T)
    assert splits.train.y_clean.shape == (n, T)
    assert splits.train.C.shape == (n, K)
    assert splits.train.sigma.shape == (n, 1)


def test_window_dataset_len_matches_split(
    tmp_path: Path, small_cfg: DatasetConfig
) -> None:
    build_dataset(small_cfg, output_dir=tmp_path)
    splits = load_dataset(tmp_path / "dataset.npz")
    ds = WindowDataset(splits.test)
    assert len(ds) == len(splits.test)


def test_window_dataset_getitem_shapes_and_dtypes(
    tmp_path: Path, small_cfg: DatasetConfig
) -> None:
    build_dataset(small_cfg, output_dir=tmp_path)
    splits = load_dataset(tmp_path / "dataset.npz")
    ds = WindowDataset(splits.train)
    item = ds[0]
    K = len(small_cfg.frequencies)
    T = small_cfg.window_size
    assert isinstance(item["x_noisy"], torch.Tensor)
    assert item["x_noisy"].shape == (T,)
    assert item["y_clean"].shape == (T,)
    assert item["C"].shape == (K,)
    assert item["sigma"].shape == (1,)
    assert item["x_noisy"].dtype == torch.float32
    assert item["C"].dtype == torch.float32

"""Tests for `services.dataset_builder.build_dataset`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sine_denoising.services.dataset_builder import build_dataset
from sine_denoising.shared.config import DatasetConfig


@pytest.fixture
def small_cfg() -> DatasetConfig:
    return DatasetConfig(
        frequencies=[1.0, 2.0],
        sigmas=[0.1, 0.2],
        fs=1000.0,
        duration=0.04,
        window_size=10,
        n_realisations=20,
        seed=0,
    )


def test_build_dataset_writes_npz_and_manifest(
    tmp_path: Path, small_cfg: DatasetConfig
) -> None:
    build_dataset(small_cfg, output_dir=tmp_path)
    assert (tmp_path / "dataset.npz").exists()
    assert (tmp_path / "manifest.json").exists()


def test_build_dataset_shapes(tmp_path: Path, small_cfg: DatasetConfig) -> None:
    splits = build_dataset(small_cfg, output_dir=tmp_path)
    K = len(small_cfg.frequencies)
    T = small_cfg.window_size
    for s in (splits.train, splits.val, splits.test):
        n = len(s)
        assert s.x_noisy.shape == (n, T)
        assert s.y_clean.shape == (n, T)
        assert s.C.shape == (n, K)
        assert s.sigma.shape == (n, 1)
        assert s.freq_idx.shape == (n,)
        assert s.realisation_id.shape == (n,)
        assert s.window_idx.shape == (n,)


def test_splits_disjoint_by_realisation_id(
    tmp_path: Path, small_cfg: DatasetConfig
) -> None:
    splits = build_dataset(small_cfg, output_dir=tmp_path)
    train_ids = set(splits.train.realisation_id.tolist())
    val_ids = set(splits.val.realisation_id.tolist())
    test_ids = set(splits.test.realisation_id.tolist())
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_every_split_contains_every_freq_sigma_pair(
    tmp_path: Path, small_cfg: DatasetConfig
) -> None:
    splits = build_dataset(small_cfg, output_dir=tmp_path)
    K = len(small_cfg.frequencies)
    for s in (splits.train, splits.val, splits.test):
        for f_idx in range(K):
            for sig in small_cfg.sigmas:
                mask = (s.freq_idx == f_idx) & (
                    s.sigma.flatten() == np.float32(sig)
                )
                assert mask.any(), (
                    f"missing freq_idx={f_idx}, sigma={sig} in a split"
                )


def test_C_is_one_hot_in_every_split(
    tmp_path: Path, small_cfg: DatasetConfig
) -> None:
    splits = build_dataset(small_cfg, output_dir=tmp_path)
    for s in (splits.train, splits.val, splits.test):
        assert np.all(s.C.sum(axis=1) == 1.0)
        assert np.all((s.C == 0.0) | (s.C == 1.0))


def test_manifest_contents(tmp_path: Path, small_cfg: DatasetConfig) -> None:
    splits = build_dataset(small_cfg, output_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["frequencies"] == small_cfg.frequencies
    assert manifest["sigmas"] == small_cfg.sigmas
    assert manifest["fs"] == small_cfg.fs
    assert manifest["window_size"] == small_cfg.window_size
    assert manifest["n_realisations"] == small_cfg.n_realisations
    assert manifest["examples_per_split"]["train"] == len(splits.train)
    assert manifest["examples_per_split"]["val"] == len(splits.val)
    assert manifest["examples_per_split"]["test"] == len(splits.test)
    total_real = sum(manifest["realisations_per_split"].values())
    expected_total = len(small_cfg.sigmas) * small_cfg.n_realisations
    assert total_real == expected_total


def test_split_counts_for_n25_ratios_70_15_15(tmp_path: Path) -> None:
    cfg = DatasetConfig(
        frequencies=[1.0, 2.0],
        sigmas=[0.1, 0.2],
        fs=1000.0,
        duration=0.04,
        window_size=10,
        n_realisations=25,
        seed=0,
    )
    splits = build_dataset(cfg, output_dir=tmp_path)
    for f_idx in range(len(cfg.frequencies)):
        for sig in cfg.sigmas:
            counts = []
            for split in (splits.train, splits.val, splits.test):
                m = (split.freq_idx == f_idx) & (
                    split.sigma.flatten() == np.float32(sig)
                )
                counts.append(len(np.unique(split.realisation_id[m])))
            assert counts == [17, 4, 4]


def test_uses_default_output_dir_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = DatasetConfig(
        frequencies=[1.0],
        sigmas=[0.1],
        fs=1000.0,
        duration=0.04,
        window_size=10,
        n_realisations=10,
        seed=0,
        output_dir=str(tmp_path / "default_out"),
    )
    build_dataset(cfg, output_dir=None)
    assert (tmp_path / "default_out" / "dataset.npz").exists()

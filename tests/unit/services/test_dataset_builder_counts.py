"""Production-config count tests for `build_dataset`.

These tests load the on-disk production config (`config/dataset.yaml`),
build the full dataset into a `tmp_path` once via a module-scoped fixture,
and assert the headline counts the lab report depends on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sine_denoising.services.dataset_builder import build_dataset
from sine_denoising.shared.config import DatasetConfig
from sine_denoising.shared.types import Splits

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def production_cfg() -> DatasetConfig:
    return DatasetConfig.from_yaml(REPO_ROOT / "config" / "dataset.yaml")


@pytest.fixture(scope="module")
def production_splits(
    production_cfg: DatasetConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> Splits:
    """Build the production dataset once; share it across tests in this file."""
    out_dir = tmp_path_factory.mktemp("production_dataset")
    return build_dataset(production_cfg, output_dir=out_dir)


def test_production_dataset_total_counts(production_splits: Splits) -> None:
    assert len(production_splits.train) == 13_600
    assert len(production_splits.val) == 3_200
    assert len(production_splits.test) == 3_200
    total = (
        len(production_splits.train)
        + len(production_splits.val)
        + len(production_splits.test)
    )
    assert total == 20_000


def test_production_per_sigma_counts(
    production_cfg: DatasetConfig, production_splits: Splits,
) -> None:
    expected = {"train": 3_400, "val": 800, "test": 800}
    cases = [
        ("train", production_splits.train),
        ("val", production_splits.val),
        ("test", production_splits.test),
    ]
    for name, split in cases:
        for sigma in production_cfg.sigmas:
            mask = np.isclose(split.sigma.flatten(), sigma, atol=1e-5)
            count = int(mask.sum())
            assert count == expected[name], (
                f"{name} split, sigma={sigma}: "
                f"expected {expected[name]}, got {count}"
            )

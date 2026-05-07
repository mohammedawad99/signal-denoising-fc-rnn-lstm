"""Mixture-extraction invariants for `build_dataset`.

These tests enforce the conditional-component-extraction contract: every
`(realisation_id, window_idx)` group must contain exactly K records with
matching `x_noisy`/`sigma` and four distinct queries, and at `sigma == 0`
the mixture must equal the sum of the clean component windows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sine_denoising.services.dataset_builder import build_dataset
from sine_denoising.shared.config import DatasetConfig
from sine_denoising.shared.types import SplitArrays


def _all_records(*splits: SplitArrays) -> dict[str, np.ndarray]:
    return {
        "C": np.concatenate([s.C for s in splits], axis=0),
        "sigma": np.concatenate([s.sigma for s in splits], axis=0),
        "x_noisy": np.concatenate([s.x_noisy for s in splits], axis=0),
        "y_clean": np.concatenate([s.y_clean for s in splits], axis=0),
        "freq_idx": np.concatenate([s.freq_idx for s in splits], axis=0),
        "realisation_id": np.concatenate(
            [s.realisation_id for s in splits], axis=0
        ),
        "window_idx": np.concatenate(
            [s.window_idx for s in splits], axis=0
        ),
    }


def test_window_idx_values_are_within_realisation(tmp_path: Path) -> None:
    cfg = DatasetConfig(
        frequencies=[1.0, 2.0],
        sigmas=[0.1, 0.2],
        fs=1000.0,
        duration=0.04,
        window_size=10,
        n_realisations=10,
        seed=0,
    )
    splits = build_dataset(cfg, output_dir=tmp_path)
    n_samples = int(round(cfg.fs * cfg.duration))
    expected_windows = n_samples // cfg.window_size
    for s in (splits.train, splits.val, splits.test):
        assert int(s.window_idx.min()) >= 0
        assert int(s.window_idx.max()) < expected_windows


def test_each_group_has_K_records_with_distinct_queries(tmp_path: Path) -> None:
    cfg = DatasetConfig(
        frequencies=[1.0, 2.0, 5.0, 10.0],
        sigmas=[0.1],
        fs=1000.0,
        duration=0.04,
        window_size=10,
        n_realisations=4,
        seed=0,
    )
    splits = build_dataset(cfg, output_dir=tmp_path)
    K = len(cfg.frequencies)
    rec = _all_records(splits.train, splits.val, splits.test)
    seen: set[tuple[int, int]] = set()
    for r, w in zip(rec["realisation_id"], rec["window_idx"], strict=True):
        key = (int(r), int(w))
        if key in seen:
            continue
        seen.add(key)
        mask = (rec["realisation_id"] == r) & (rec["window_idx"] == w)
        assert int(mask.sum()) == K
        # Exactly one record per query frequency.
        assert sorted(rec["freq_idx"][mask].tolist()) == list(range(K))
        # C rows are the four canonical one-hots.
        c_rows = rec["C"][mask]
        for f_idx in range(K):
            row = c_rows[rec["freq_idx"][mask] == f_idx][0]
            assert row.argmax() == f_idx
            assert row.sum() == 1.0


def test_query_group_shares_x_noisy_and_sigma(tmp_path: Path) -> None:
    cfg = DatasetConfig(
        frequencies=[1.0, 2.0, 5.0, 10.0],
        sigmas=[0.1],
        fs=1000.0,
        duration=0.04,
        window_size=10,
        n_realisations=4,
        seed=0,
    )
    splits = build_dataset(cfg, output_dir=tmp_path)
    rec = _all_records(splits.train, splits.val, splits.test)
    for r in np.unique(rec["realisation_id"]):
        for w in np.unique(rec["window_idx"][rec["realisation_id"] == r]):
            mask = (rec["realisation_id"] == r) & (rec["window_idx"] == w)
            x_for_group = rec["x_noisy"][mask]
            sigma_for_group = rec["sigma"][mask]
            for x in x_for_group[1:]:
                np.testing.assert_array_equal(x, x_for_group[0])
            for s in sigma_for_group[1:]:
                np.testing.assert_array_equal(s, sigma_for_group[0])
            # y_clean must vary across queries (otherwise extraction is trivial).
            y_for_group = rec["y_clean"][mask]
            assert not np.allclose(y_for_group[0], y_for_group[1])


def test_sigma_zero_x_noisy_equals_sum_of_clean_components(
    tmp_path: Path,
) -> None:
    cfg = DatasetConfig(
        frequencies=[1.0, 2.0, 5.0, 10.0],
        sigmas=[0.0],
        fs=1000.0,
        duration=0.04,
        window_size=10,
        n_realisations=4,
        seed=0,
    )
    splits = build_dataset(cfg, output_dir=tmp_path)
    rec = _all_records(splits.train, splits.val, splits.test)
    for r in np.unique(rec["realisation_id"]):
        for w in np.unique(rec["window_idx"][rec["realisation_id"] == r]):
            mask = (rec["realisation_id"] == r) & (rec["window_idx"] == w)
            x_window = rec["x_noisy"][mask][0]
            y_sum = rec["y_clean"][mask].sum(axis=0)
            np.testing.assert_allclose(y_sum, x_window, rtol=1e-5, atol=1e-6)

"""Build the dataset (writes `.npz` + `manifest.json`); see PRD §2-§5."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sine_denoising.services.encoding import one_hot
from sine_denoising.services.mixtures import make_mixture_realisation
from sine_denoising.shared.config import DatasetConfig
from sine_denoising.shared.types import SplitArrays, Splits

SPLIT_NAMES = ("train", "val", "test")


def build_dataset(
    cfg: DatasetConfig,
    output_dir: str | Path | None = None,
) -> Splits:
    """Generate, stratify-split, and persist the synthetic dataset.

    Returns the in-memory `Splits` and writes `dataset.npz` + `manifest.json`
    under `output_dir` (or `cfg.output_dir` if `None`).
    """
    out = Path(output_dir) if output_dir is not None else Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    arrays, real_meta = _generate_records(cfg, rng)
    real_split = _stratified_split(real_meta, cfg, rng)
    split_arr = np.array(
        [real_split[int(r)] for r in arrays["realisation_id"]], dtype="<U5"
    )
    archive: dict[str, Any] = {"split": split_arr, **arrays}
    np.savez_compressed(out / "dataset.npz", **archive)
    splits = _to_splits(arrays, split_arr)
    _write_manifest(out / "manifest.json", cfg, real_split, splits)
    return splits


def _generate_records(
    cfg: DatasetConfig, rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
    K = len(cfg.frequencies)
    chunks: dict[str, list[np.ndarray]] = {
        "C": [], "sigma": [], "x_noisy": [], "y_clean": [],
        "freq_idx": [], "realisation_id": [], "window_idx": [],
    }
    real_meta: list[tuple[int, int]] = []
    rid = 0
    for s_idx, sigma in enumerate(cfg.sigmas):
        for _ in range(cfg.n_realisations):
            mixture_w, clean_w = make_mixture_realisation(
                cfg.frequencies, sigma, cfg.fs, cfg.duration,
                cfg.window_size, rng,
            )
            num_w = mixture_w.shape[0]
            for q in range(K):
                chunks["x_noisy"].append(mixture_w)
                chunks["y_clean"].append(clean_w[q])
                chunks["C"].append(
                    np.broadcast_to(one_hot(q, K), (num_w, K)).astype(np.float32)
                )
                chunks["sigma"].append(np.full((num_w, 1), sigma, dtype=np.float32))
                chunks["freq_idx"].append(np.full((num_w,), q, dtype=np.int8))
                chunks["realisation_id"].append(np.full((num_w,), rid, dtype=np.int32))
                chunks["window_idx"].append(np.arange(num_w, dtype=np.int16))
            real_meta.append((rid, s_idx))
            rid += 1
    arrays = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    return arrays, real_meta


def _stratified_split(
    real_meta: list[tuple[int, int]],
    cfg: DatasetConfig,
    rng: np.random.Generator,
) -> dict[int, str]:
    by_stratum: dict[int, list[int]] = {}
    for rid, s_idx in real_meta:
        by_stratum.setdefault(s_idx, []).append(rid)
    assignment: dict[int, str] = {}
    for stratum_rids in by_stratum.values():
        order = np.array(stratum_rids, dtype=np.int64)
        rng.shuffle(order)
        n = order.size
        n_train = int(np.floor(cfg.train_ratio * n))
        n_val = int(np.ceil(cfg.val_ratio * n))
        n_test = n - n_train - n_val
        if n_test < 1:
            if n_val > 1:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1
            n_test = n - n_train - n_val
        for r in order[:n_train]:
            assignment[int(r)] = "train"
        for r in order[n_train:n_train + n_val]:
            assignment[int(r)] = "val"
        for r in order[n_train + n_val:]:
            assignment[int(r)] = "test"
    return assignment


def _to_splits(arrays: dict[str, np.ndarray], split_arr: np.ndarray) -> Splits:
    out: dict[str, SplitArrays] = {}
    for name in SPLIT_NAMES:
        mask = split_arr == name
        out[name] = SplitArrays(
            C=arrays["C"][mask],
            sigma=arrays["sigma"][mask],
            x_noisy=arrays["x_noisy"][mask],
            y_clean=arrays["y_clean"][mask],
            freq_idx=arrays["freq_idx"][mask],
            realisation_id=arrays["realisation_id"][mask],
            window_idx=arrays["window_idx"][mask],
        )
    return Splits(train=out["train"], val=out["val"], test=out["test"])


def _write_manifest(
    path: Path,
    cfg: DatasetConfig,
    real_split: dict[int, str],
    splits: Splits,
) -> None:
    manifest = {
        "frequencies": list(cfg.frequencies),
        "sigmas": list(cfg.sigmas),
        "fs": cfg.fs,
        "duration": cfg.duration,
        "window_size": cfg.window_size,
        "n_realisations": cfg.n_realisations,
        "seed": cfg.seed,
        "realisations_per_split": {
            name: sum(1 for s in real_split.values() if s == name)
            for name in SPLIT_NAMES
        },
        "examples_per_split": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "dataset_version": "v2-mixture",
    }
    path.write_text(json.dumps(manifest, indent=2))

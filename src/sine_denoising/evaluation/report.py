"""Evaluate trained checkpoints and write `summary.json` + a reconstruction PNG.

The function loads the persisted dataset, restores each named model from
`{checkpoint_dir}/{name}_best.pt`, predicts on the **test** split, and
records per-model metrics in a JSON summary plus a single overlay plot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sine_denoising.evaluation.metrics import (
    mse_overall,
    mse_per_freq,
    mse_per_sigma,
    snr_improvement_db,
)
from sine_denoising.evaluation.plots import plot_reconstruction
from sine_denoising.models.base import DenoiserBase
from sine_denoising.models.fc import FCDenoiser
from sine_denoising.models.lstm import LSTMDenoiser
from sine_denoising.models.rnn import RNNDenoiser
from sine_denoising.services.dataset_loader import load_dataset
from sine_denoising.shared.types import SplitArrays


def _build_model(name: str, num_classes: int, window_size: int) -> DenoiserBase:
    """Construct the named model with the per-PRD default `hidden_size`."""
    if name == "fc":
        return FCDenoiser(window_size=window_size, num_classes=num_classes)
    if name == "rnn":
        return RNNDenoiser(num_classes=num_classes)
    if name == "lstm":
        return LSTMDenoiser(num_classes=num_classes)
    raise ValueError(f"unknown model: {name!r}")


def _predict(model: DenoiserBase, test: SplitArrays, device: str) -> np.ndarray:
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(test.x_noisy).to(device)
        C = torch.from_numpy(test.C).to(device)
        sigma = torch.from_numpy(test.sigma).to(device)
        y_pred = model(x, C, sigma)
    out: np.ndarray = y_pred.cpu().numpy()
    return out


def generate_report(
    dataset_path: str | Path,
    checkpoint_dir: str | Path,
    summary_path: str | Path,
    plot_path: str | Path,
    model_names: tuple[str, ...] = ("fc", "rnn", "lstm"),
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate `model_names`, write `summary.json` + reconstruction PNG.

    Raises `FileNotFoundError` (with the offending path) when any
    `{checkpoint_dir}/{name}_best.pt` is missing.
    """
    splits = load_dataset(dataset_path)
    test = splits.test
    num_classes = int(test.C.shape[1])
    window_size = int(test.x_noisy.shape[1])

    summary: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    ckpt_dir = Path(checkpoint_dir)
    for name in model_names:
        ckpt = ckpt_dir / f"{name}_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt}")
        model = _build_model(name, num_classes, window_size)
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True),
        )
        y_pred = _predict(model, test, device)
        preds[name] = y_pred
        summary[name] = {
            "mse_overall": mse_overall(y_pred, test.y_clean),
            "mse_per_freq": mse_per_freq(
                y_pred, test.y_clean, test.freq_idx, num_classes,
            ),
            "mse_per_sigma": mse_per_sigma(y_pred, test.y_clean, test.sigma),
            "snr_improvement_db": snr_improvement_db(
                test.y_clean, test.x_noisy, y_pred,
            ),
            "parameters": model.parameter_count(),
        }

    summary_p = Path(summary_path)
    summary_p.parent.mkdir(parents=True, exist_ok=True)
    summary_p.write_text(json.dumps(summary, indent=2))

    idx = 0
    plot_reconstruction(
        clean=test.y_clean[idx],
        noisy=test.x_noisy[idx],
        preds_by_model={name: preds[name][idx] for name in preds},
        out_path=plot_path,
        title=f"reconstruction (freq_idx={int(test.freq_idx[idx])})",
    )
    return summary

"""Plotting utilities for the lab report.

Uses Matplotlib's "Agg" backend so figures render correctly under headless
environments (CI, WSL without a display server).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_reconstruction(
    clean: np.ndarray,
    noisy: np.ndarray,
    preds_by_model: dict[str, np.ndarray],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Overlay clean / noisy / per-model predictions on the same axes.

    All arrays must be 1-D with the same length `T`. The function creates
    `out_path`'s parent directory if it does not yet exist, writes a PNG,
    closes the figure, and returns the resolved `Path`.
    """
    if clean.ndim != 1:
        raise ValueError(f"clean must be 1-D; got shape {clean.shape}")
    if noisy.shape != clean.shape:
        raise ValueError(
            f"noisy shape {noisy.shape} != clean shape {clean.shape}"
        )
    for name, pred in preds_by_model.items():
        if pred.shape != clean.shape:
            raise ValueError(
                f"prediction {name!r} shape {pred.shape} != "
                f"clean shape {clean.shape}"
            )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.arange(clean.size)
    ax.plot(t, clean, label="clean", color="black", linewidth=2)
    ax.plot(t, noisy, label="noisy", color="gray", linestyle="--", alpha=0.7)
    for name, pred in preds_by_model.items():
        ax.plot(t, pred, label=name)
    ax.set_xlabel("sample index")
    ax.set_ylabel("amplitude")
    ax.legend()
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out

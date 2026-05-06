"""Pydantic configs for dataset generation and training."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetConfig(BaseModel):
    """Configuration for synthetic sine-signal dataset generation.

    Mirrors the parameters fixed in `docs/PRD_dataset.md`.
    """

    model_config = ConfigDict(frozen=True)

    frequencies: list[float] = Field(min_length=1)
    sigmas: list[float] = Field(min_length=1)
    fs: float = Field(gt=0)
    duration: float = Field(gt=0)
    window_size: int = Field(gt=0)
    n_realisations: int = Field(gt=0)
    seed: int = Field(ge=0)
    output_dir: str = "data/generated"
    train_ratio: float = Field(default=0.70, gt=0, lt=1)
    val_ratio: float = Field(default=0.15, gt=0, lt=1)
    test_ratio: float = Field(default=0.15, gt=0, lt=1)

    @model_validator(mode="after")
    def _check_invariants(self) -> DatasetConfig:
        f_max = max(self.frequencies)
        if self.fs <= 2 * f_max:
            raise ValueError(
                f"Sampling rate fs={self.fs} violates Nyquist for "
                f"f_max={f_max} Hz; need fs > 2*f_max."
            )
        n_samples = int(round(self.fs * self.duration))
        if n_samples % self.window_size != 0:
            raise ValueError(
                f"fs*duration ({n_samples}) must be a multiple of "
                f"window_size ({self.window_size})."
            )
        for s in self.sigmas:
            if s < 0:
                raise ValueError(f"sigma must be non-negative; got {s}")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split ratios must sum to 1; got {total} "
                f"(train={self.train_ratio}, val={self.val_ratio}, "
                f"test={self.test_ratio})."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetConfig:
        with open(path) as f:
            data: Any = yaml.safe_load(f)
        return cls(**data)


class TrainingConfig(BaseModel):
    """Configuration for training one of {fc, rnn, lstm}.

    Mirrors the shared training contract in `docs/PRD_models.md`.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    epochs: int = Field(gt=0)
    early_stopping_patience: int = Field(ge=0)
    grad_clip_norm: float = Field(gt=0)
    hidden_size: int = Field(gt=0)
    seed: int = Field(ge=0)
    device: str = "cpu"
    results_dir: str = "results"

    @model_validator(mode="after")
    def _check_model_name(self) -> TrainingConfig:
        allowed = {"fc", "rnn", "lstm"}
        if self.model not in allowed:
            raise ValueError(
                f"model must be one of {sorted(allowed)}; got {self.model!r}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        with open(path) as f:
            data: Any = yaml.safe_load(f)
        return cls(**data)

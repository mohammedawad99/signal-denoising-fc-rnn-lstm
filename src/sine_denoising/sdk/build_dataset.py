"""CLI entry point: build the synthetic sine-signal dataset on disk.

Run with:
    uv run python -m sine_denoising.sdk.build_dataset
optional flags:
    --config path/to/dataset.yaml   (default: config/dataset.yaml)
    --output-dir path/to/output     (overrides cfg.output_dir)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sine_denoising.services.dataset_builder import build_dataset
from sine_denoising.shared.config import DatasetConfig

DEFAULT_CONFIG = "config/dataset.yaml"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_dataset",
        description="Generate the synthetic sine-signal dataset on disk.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to a dataset YAML config (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the config's output_dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = DatasetConfig.from_yaml(args.config)
    if args.output_dir is not None:
        cfg = cfg.model_copy(update={"output_dir": args.output_dir})
    splits = build_dataset(cfg)
    out_dir = Path(cfg.output_dir)
    print(f"Output directory: {out_dir}")
    print(f"Train size: {len(splits.train)}")
    print(f"Val size:   {len(splits.val)}")
    print(f"Test size:  {len(splits.test)}")
    print(f"dataset.npz:   {out_dir / 'dataset.npz'}")
    print(f"manifest.json: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

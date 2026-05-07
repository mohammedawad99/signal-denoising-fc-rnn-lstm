"""CLI entry point: evaluate trained checkpoints on the test split.

Run with:
    uv run python -m sine_denoising.sdk.evaluate
"""
from __future__ import annotations

import argparse
import sys

from sine_denoising.evaluation.report import generate_report

DEFAULT_DATASET_PATH = "data/generated/dataset.npz"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"
DEFAULT_SUMMARY_PATH = "results/summary.json"
DEFAULT_PLOT_PATH = "assets/generated/reconstruction_example.png"
DEFAULT_MODELS = ("fc", "rnn", "lstm")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluate trained denoising checkpoints on the test split.",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Path to dataset.npz (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory holding `{model}_best.pt` (default: %(default)s).",
    )
    parser.add_argument(
        "--summary-path",
        default=DEFAULT_SUMMARY_PATH,
        help="Where to write summary.json (default: %(default)s).",
    )
    parser.add_argument(
        "--plot-path",
        default=DEFAULT_PLOT_PATH,
        help="Where to write the reconstruction PNG (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: %(default)s).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=DEFAULT_MODELS,
        help="Which models to evaluate (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = generate_report(
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        summary_path=args.summary_path,
        plot_path=args.plot_path,
        model_names=tuple(args.models),
        device=args.device,
    )
    print(f"Summary: {args.summary_path}")
    print(f"Plot: {args.plot_path}")
    print(f"Models evaluated: {', '.join(args.models)}")
    for name, entry in summary.items():
        if "mse_overall" in entry:
            print(f"{name}: mse_overall = {entry['mse_overall']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

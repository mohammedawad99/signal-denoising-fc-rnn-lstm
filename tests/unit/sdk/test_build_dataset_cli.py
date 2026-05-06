"""Tests for the `sdk.build_dataset` CLI entry point."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sine_denoising.sdk.build_dataset import main


def _write_small_config(tmp_path: Path, output_dir: Path) -> Path:
    cfg = {
        "frequencies": [1.0, 2.0],
        "sigmas": [0.1, 0.2],
        "fs": 20.0,
        "duration": 1.0,
        "window_size": 10,
        "n_realisations": 10,
        "seed": 0,
        "output_dir": str(output_dir),
    }
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_cli_builds_dataset_and_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "out"
    cfg_path = _write_small_config(tmp_path, out_dir)

    rc = main(["--config", str(cfg_path)])

    assert rc == 0
    assert (out_dir / "dataset.npz").exists()
    assert (out_dir / "manifest.json").exists()
    captured = capsys.readouterr()
    assert "Train size:" in captured.out
    assert "Val size:" in captured.out
    assert "Test size:" in captured.out
    assert "dataset.npz" in captured.out
    assert "manifest.json" in captured.out
    assert str(out_dir) in captured.out


def test_cli_output_dir_override_takes_priority(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    config_out = tmp_path / "from_config"
    cfg_path = _write_small_config(tmp_path, config_out)
    override = tmp_path / "from_cli"

    rc = main([
        "--config", str(cfg_path),
        "--output-dir", str(override),
    ])

    assert rc == 0
    assert (override / "dataset.npz").exists()
    assert (override / "manifest.json").exists()
    # The path from the config file should NOT have been used.
    assert not config_out.exists()


def test_cli_uses_default_config_path_when_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Place a config at the default relative location and chdir there so
    # the CLI's default `config/dataset.yaml` resolves to it.
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    out_dir = tmp_path / "default_out"
    cfg_payload = {
        "frequencies": [1.0],
        "sigmas": [0.1],
        "fs": 20.0,
        "duration": 1.0,
        "window_size": 10,
        "n_realisations": 10,
        "seed": 0,
        "output_dir": str(out_dir),
    }
    (cfg_dir / "dataset.yaml").write_text(yaml.safe_dump(cfg_payload))
    monkeypatch.chdir(tmp_path)

    rc = main([])

    assert rc == 0
    assert (out_dir / "dataset.npz").exists()

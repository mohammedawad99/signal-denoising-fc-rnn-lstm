# PLAN — HW1 Implementation Roadmap

This plan turns the PRDs into a sequence of implementable phases. Each phase ends with a Git commit on `main`. Every Python file written under any phase must stay under 150 lines.

## Phase 0 — Planning and skeleton (this phase)
Outcome: documentation and scaffolding only — **no Python implementation yet**.
- Write `docs/PRD.md`, `docs/PRD_dataset.md`, `docs/PRD_models.md`.
- Write `docs/PLAN.md` (this file) and `docs/TODO.md`.
- Initialize `README.md` as a lab-report outline (placeholders for results).
- Commit message: `docs: planning artefacts for HW1 (PRDs, PLAN, TODO, README outline)`.

Exit criterion: instructor can review the planning artefacts and confirm the interpretation before any code is written.

## Phase 1 — Configuration and shared utilities
Outcome: typed config, RNG seeding, and shared constants live somewhere a future module can import.

Modules:
- `src/sine_denoising/shared/config.py` — pydantic models for `DatasetConfig` and `TrainingConfig` (frequencies, sigmas, fs, duration, T, batch size, lr, epochs, seed).
- `src/sine_denoising/shared/seeding.py` — single `set_seed(seed: int)` function seeding numpy, random, and torch.
- `src/sine_denoising/shared/types.py` — small `TypedDict`/`dataclass` records for split arrays.
- `config/dataset.yaml`, `config/training.yaml` — concrete values matching the PRDs.

Tests (pytest):
- Loading config from YAML returns a validated pydantic object.
- `set_seed` is deterministic (sample two numpy/torch draws after seeding twice with the same seed → equal).

Commit: `feat(shared): config models, seeding, types`.

## Phase 2 — Dataset generation and loading
Outcome: a reproducible `data/generated/dataset.npz` plus a clean `load_dataset` API.

Modules (each under 150 lines):
- `src/sine_denoising/services/signal.py` — `make_signal`, pure-numpy waveform + noise generator.
- `src/sine_denoising/services/encoding.py` — `one_hot`, `window` helpers.
- `src/sine_denoising/services/dataset_builder.py` — orchestrator that sweeps `(freq, sigma, realisation)`, applies splits, writes `.npz` + `manifest.json`.
- `src/sine_denoising/services/dataset_loader.py` — `load_dataset(path) -> SplitArrays` plus a thin `torch.utils.data.Dataset` wrapper.
- `src/sine_denoising/sdk/build_dataset.py` — small CLI entry point (`uv run python -m sine_denoising.sdk.build_dataset`).

Tests:
- `make_signal` shapes and clean-vs-noisy equivalence with `sigma = 0`.
- `one_hot` correctness (sum to 1, correct index).
- `window` correctness (no overlap, exact count).
- Empirical noise std per stratum within tolerance of nominal `sigma`.
- Splits are disjoint at the realisation level and stratified within ±1.

Commit: `feat(data): synthetic sine dataset generator and loader`.

## Phase 3 — Models
Outcome: three small PyTorch modules sharing the same forward signature.

Modules:
- `src/sine_denoising/models/base.py` — abstract base or shared mixin (`forward(x_noisy, C, sigma) -> y_pred`, `parameter_count()` helper).
- `src/sine_denoising/models/fc.py` — Fully Connected.
- `src/sine_denoising/models/rnn.py` — vanilla RNN.
- `src/sine_denoising/models/lstm.py` — LSTM.
- `src/sine_denoising/models/__init__.py` — exports.

Tests (lightweight, no training):
- Each model accepts `(B, 10), (B, 4), (B, 1)` and returns `(B, 10)`.
- `parameter_count()` returns a positive int.
- Forward pass with `B = 1` doesn't error (smoke test).

Commit: `feat(models): FC, RNN, LSTM with shared signature`.

## Phase 4 — Training loop
Outcome: a single trainer that any of the three models can be plugged into.

Modules:
- `src/sine_denoising/training/loops.py` — `train_one_epoch`, `evaluate`. Pure functions taking `(model, dataloader, optimizer)`.
- `src/sine_denoising/training/trainer.py` — `Trainer` class wiring config + model + dataloaders + early stopping + checkpoint saving.
- `src/sine_denoising/sdk/train.py` — CLI entry point: `uv run python -m sine_denoising.sdk.train --model {fc|rnn|lstm}`.

Tests:
- `train_one_epoch` decreases the loss on a tiny synthetic batch (overfitting test).
- Early-stopping triggers when val loss stagnates.
- Checkpoint round-trip: save then load gives identical outputs.

Commit: `feat(training): generic trainer with early stopping`.

## Phase 5 — Evaluation and plotting
Outcome: a single evaluation step that produces all metrics and the plots referenced in the lab report.

Modules:
- `src/sine_denoising/evaluation/metrics.py` — `mse_overall`, `mse_per_freq`, `mse_per_sigma`, `snr_improvement_db`.
- `src/sine_denoising/evaluation/plots.py` — `plot_reconstruction(clean, noisy, preds_by_model, out_path)`.
- `src/sine_denoising/evaluation/report.py` — runs all three trained checkpoints over the test set and writes a `results/summary.json` plus PNG figures under `assets/generated/`.
- `src/sine_denoising/sdk/evaluate.py` — CLI entry point.

Tests:
- Metric correctness on hand-computed mini examples.
- Plot function writes a non-empty PNG file (golden-path check).

Commit: `feat(eval): metrics, plots, summary report`.

## Phase 6 — Lab report
Outcome: README.md filled in with real numbers and figures (replacing placeholders).
- Results table (per-model overall MSE, per-freq, per-sigma, SNR improvement, parameter count).
- Loss curves figure.
- Reconstruction figures.
- Discussion + conclusion paragraphs grounded in the actual numbers.

Commit: `docs: lab report with experimental results`.

## Phase 7 — Quality gate and packaging
Outcome: green CI-like gate locally, ready for submission.
- `uv sync` clean.
- `ruff check .` clean.
- `mypy` clean (per `pyproject.toml` strict config).
- `pytest --cov` green with `fail_under = 85`.
- README link to the GitHub repo present.
- Final commit: `chore: lint, type, coverage gate green`.

## Phase 8 — Conditional component extraction
Outcome: dataset, training, evaluation, and report all reflect the noisy-mixture extraction task — `x_noisy` is a window of the summed mixture and `y_clean` is the clean window of the component selected by `C` (see `docs/PRD.md` §2 and `docs/PRD_dataset.md` §2).

Steps:
- **Docs (8A).** Update `docs/PRD.md`, `docs/PRD_dataset.md`, and `docs/PRD_models.md` to describe the mixture-extraction task; commit.
- **Builder (8B).** Refactor `src/sine_denoising/services/dataset_builder.py` so each realisation produces one noisy mixture and the four clean components, then emits one record per `(window_idx, query_freq)` pair (with `realisation_id` and `window_idx` carried as bookkeeping fields per `docs/PRD_dataset.md` §3.2). Update `tests/unit/services/test_dataset_builder.py` for the new schema and the sigma-only stratification.
- **Regenerate data.** Rebuild `data/generated/dataset.npz` from the new builder.
- **Retrain.** Run `uv run python -m sine_denoising.sdk.train --model {fc,rnn,lstm}` and refresh `results/checkpoints/*.pt` (no model-code change is expected since the external interface is unchanged).
- **Re-evaluate.** Run `uv run python -m sine_denoising.sdk.evaluate` to refresh `results/summary.json` and `assets/generated/reconstruction_example.png`; copy the figure into `assets/report/` for the committed report asset.
- **Update README.** Replace the results tables, discussion, and any wording that still describes isolated-component recovery with the new mixture-extraction numbers and framing.
- **Quality gate.** `ruff check .`, `mypy src`, `pytest -q`, file-size sanity.
- **Push.** Commit each step as it lands and push the final state of `main`.

## Cross-cutting rules
- One responsibility per file. If a file approaches 150 lines, split it.
- Tests live under `tests/unit/` mirroring the source tree; integration-style tests (full small training loop) live under `tests/integration/`.
- Generated artefacts (`data/generated/`, `results/`, `assets/generated/`, `*.pt`) are gitignored. The manifest, the code, the docs, and the small final figures used in the README are committed.
- Every meaningful step ends with a Git commit. No "WIP" omnibus commits.

## Open questions tracked here
- Do we need a `Makefile` or are uv scripts enough? Decision: uv scripts, defined in `pyproject.toml` if/when needed; revisit at Phase 7.
- Do we package as an importable library or as scripts? Decision: importable library with `sdk/` CLI entry points (already implied by the `src/` layout).

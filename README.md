# Sine/Cosine Signal Denoising — Comparing FC, RNN, and LSTM (HW1 Lab Report)

> Course: AI Agent Orchestration and Deep Learning — Homework 1.
> Author: Mohamed Awad.
> Repository: _link will be added once pushed to GitHub_.

This README is also the lab report. Sections are filled in incrementally as the project advances; placeholders marked `TBD` will be replaced with real numbers and figures once the experiments are run.

---

## 1. Abstract
We study a small supervised denoising task: given 10 consecutive noisy samples of a sinusoid, the one-hot encoded identity of its frequency, and the noise level `sigma`, predict the corresponding 10 clean samples. We compare three neural-network architectures — a Fully Connected baseline, a vanilla RNN, and an LSTM — under an identical training regime (same data, same loss, same optimizer, same budget) so the only changing factor is the backbone family. _Results: TBD._

## 2. Introduction
Recurrent networks are typically motivated by tasks with a temporal structure. A short, fixed-length window of a noisy sinusoid is the smallest meaningful instance of such a task, which makes it a clean test bed for comparing architectures and reasoning about why one would beat another. The assignment fixes the loss (MSE) and the dataset shape (10 noisy samples → 10 clean samples), and lets us choose the rest. We treat those choices as engineering decisions and document them in `docs/PRD.md`.

## 3. Problem statement
Predict `y_clean ∈ R^10` from `(x_noisy ∈ R^10, C ∈ {0,1}^4, sigma ∈ R)` by minimizing `MSE(y_pred, y_clean)`. The four frequencies are known and fixed; `C` indicates which one generated the sample. `sigma` is the standard deviation of the additive Gaussian noise expressed as a fraction of the signal amplitude.

## 4. Method

### 4.1 Data
Synthetic sinusoidal signals at four frequencies `{1, 2, 5, 10} Hz`, sampled at `fs = 50 Hz` (well above the Nyquist limit of 20 Hz), 10 s each, with additive Gaussian noise at four `sigma` levels `{0.05, 0.10, 0.20, 0.30}`. Each `(frequency, sigma)` pair contributes 25 independent realisations. Each realisation is sliced into 50 non-overlapping windows of length 10. Splits are stratified by `(frequency, sigma)` at the realisation level (70/15/15). Full details in `docs/PRD_dataset.md`.

### 4.2 Models
All three models share the signature `forward(x_noisy, C, sigma) -> y_pred ∈ R^10`. The differences:

| Model | Inputs combined | Backbone                              | Approx. params |
|-------|------------------|----------------------------------------|----------------|
| FC    | flat `R^15`      | `15 → 64 → 64 → 10` (ReLU)             | TBD            |
| RNN   | per-step `R^6`   | `nn.RNN(6, 32)` + `Linear(32, 1)`      | TBD            |
| LSTM  | per-step `R^6`   | `nn.LSTM(6, 32)` + `Linear(32, 1)`     | TBD            |

Full details in `docs/PRD_models.md`.

### 4.3 Training
- Loss: MSE.
- Optimizer: Adam, `lr = 1e-3`.
- Batch size: 64, epochs: 30, early stopping patience: 5.
- Gradient clipping at `1.0` for RNN/LSTM.
- Single seed for everything (default `42`); seed and hyperparameters are persisted with each run.

### 4.4 Evaluation
- `MSE_overall` on the test set.
- `MSE_per_freq` and `MSE_per_sigma`.
- `SNR_improvement_dB = 10·log10(MSE_input / MSE_output)`.
- Reconstruction plots (clean / noisy / each model) at three sigma levels per frequency.

## 5. Experimental setup
- Hardware: `TBD` (CPU/GPU, RAM).
- Software: Python 3.11, PyTorch (see `pyproject.toml` for exact versions), uv-managed environment.
- Reproducibility: `set_seed(42)` at the top of every entry point; dataset persisted to `data/generated/dataset.npz` with a manifest.

## 6. Results
_All numbers below are placeholders; they will be filled in after Phase 5._

### 6.1 Headline metrics

| Model | Overall MSE ↓ | SNR improvement (dB) ↑ | Params |
|-------|---------------|-------------------------|--------|
| FC    | TBD           | TBD                     | TBD    |
| RNN   | TBD           | TBD                     | TBD    |
| LSTM  | TBD           | TBD                     | TBD    |

### 6.2 Breakdown by frequency (test MSE)

| Model | 1 Hz | 2 Hz | 5 Hz | 10 Hz |
|-------|------|------|------|-------|
| FC    | TBD  | TBD  | TBD  | TBD   |
| RNN   | TBD  | TBD  | TBD  | TBD   |
| LSTM  | TBD  | TBD  | TBD  | TBD   |

### 6.3 Breakdown by sigma (test MSE)

| Model | σ=0.05 | σ=0.10 | σ=0.20 | σ=0.30 |
|-------|--------|--------|--------|--------|
| FC    | TBD    | TBD    | TBD    | TBD    |
| RNN   | TBD    | TBD    | TBD    | TBD    |
| LSTM  | TBD    | TBD    | TBD    | TBD    |

### 6.4 Loss curves
_Figure: training and validation loss vs epoch for each model._ — `assets/generated/loss_curves.png` (TBD).

### 6.5 Reconstructions
_Figure: clean vs noisy vs FC/RNN/LSTM, one row per frequency, three sigmas._ — `assets/generated/reconstructions.png` (TBD).

## 7. Discussion
_To be written based on actual numbers._ Topics to cover:
- Does the LSTM beat the RNN? By how much? Is the gap meaningful given the standard error?
- Does the FC baseline keep up? On a 10-step window with frequency given as a one-hot, the FC has very few unknowns to recover — we should expect it to be competitive.
- How does each model degrade as sigma grows?
- Per-frequency behaviour: do the higher frequencies (less periods per window vs more periods per window) matter?

## 8. Limitations
- Window length fixed at 10 — too short to showcase long-range memory advantages of LSTM.
- Synthetic data only; no real-world noise statistics.
- No transformer baseline.
- Hyperparameters were tuned only enough to produce a fair comparison, not to find each model's best.

## 9. Future work
- Try sliding-window inputs at inference time (overlap-add reconstruction) to see if LSTM benefits from longer effective context.
- Add a `sigma = 0.0` stratum to verify the model passes clean inputs through unchanged.
- Try coloured noise to see whether the gap between models widens.
- Compare with a small 1-D temporal convolutional baseline.

## 10. Conclusion
_To be written once results are in._

## 11. Project structure
```
.
├── config/                       # YAML configs (dataset, training)
├── data/
│   └── generated/                # built dataset (.npz, manifest.json) — gitignored
├── docs/
│   ├── ASSIGNMENT_NOTES.md
│   ├── PRD.md
│   ├── PRD_dataset.md
│   ├── PRD_models.md
│   ├── PLAN.md
│   └── TODO.md
├── results/                      # checkpoints, summary.json — gitignored
├── assets/generated/             # figures used in this report — gitignored
├── src/sine_denoising/
│   ├── shared/                   # config, seeding, types
│   ├── services/                 # signal generation, encoding, dataset
│   ├── models/                   # fc.py, rnn.py, lstm.py, base.py
│   ├── training/                 # trainer, loops
│   ├── evaluation/               # metrics, plots, report
│   └── sdk/                      # CLI entry points
├── tests/
│   ├── unit/                     # mirrors src/
│   └── integration/              # small end-to-end smoke training
├── pyproject.toml
├── README.md                     # this file
└── uv.lock
```

## 12. Reproducing the experiments
_Commands will be finalized once the SDK entry points exist; the intended workflow is:_

```bash
uv sync
uv run python -m sine_denoising.sdk.build_dataset
uv run python -m sine_denoising.sdk.train --model fc
uv run python -m sine_denoising.sdk.train --model rnn
uv run python -m sine_denoising.sdk.train --model lstm
uv run python -m sine_denoising.sdk.evaluate
uv run pytest
```

## 13. Acknowledgements
- Course staff for the assignment specification.
- Claude Code (CLI) used as the AI pair-programming assistant, in line with the course workflow.

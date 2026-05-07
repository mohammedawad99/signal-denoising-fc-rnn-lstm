# Conditional Component Extraction from a Noisy Sine Mixture — Comparing FC, RNN, and LSTM

> Course: AI Agent Orchestration and Deep Learning — Homework 1.
> Author: Mohamed Awad.
> Repository: <https://github.com/mohammedawad99/signal-denoising-fc-rnn-lstm>.

## Abstract

We compare three neural-network architectures — Fully Connected (FC), vanilla Elman RNN, and LSTM — on a controlled conditional-extraction task. The model receives a 10-sample window of a noisy mixture that sums one independently-noised realisation of each of four known frequency components, plus a one-hot query `C` selecting which component to reconstruct, and must predict the clean window of that selected component. All three models are trained on the same data with the same loss (MSE), optimiser (Adam @ `lr = 1e-3`), batch size, and 30-epoch schedule with patience-5 early stopping. On a held-out test set of 3,200 query records the FC achieves the lowest test MSE (**0.137753**) and the largest SNR improvement (**+10.76 dB**); the LSTM (0.259095 / +8.02 dB) and the vanilla RNN (0.262388 / +7.96 dB) are roughly tied at the harder task. Per-frequency behaviour shows a strong monotonic pattern: the higher frequencies (5 Hz, 10 Hz) are markedly easier to extract than the lower ones (1 Hz, 2 Hz), because at `fs = 50 Hz` a 10-sample window covers only 0.2 s — only 0.2 / 0.4 cycles of 1 / 2 Hz components, but 1 / 2 full cycles of 5 / 10 Hz components.

## 1. Assignment interpretation

The dataset entry pairs a noisy mixture window with the clean component identified by a one-hot query. The interpretation we adopt is:

> Each model receives `(x_noisy, C, sigma)` and predicts `y_clean` of the same length as `x_noisy`. `x_noisy` is a 10-sample window of the noisy mixture (sum of four independently-noised sine components), `C` is a one-hot vector that selects which of the four known frequencies to reconstruct, and `y_clean` is the clean window of that selected component. Training minimises `MSE(y_pred, y_clean)`.

For RNN/LSTM, `C` and `sigma` are broadcast across all 10 timesteps, so each step sees `[x_t, C, sigma] ∈ R^6`. For FC, the entire window is flattened together with `C` and `sigma` into a single `R^15` vector. The full chain of assumptions and their justifications lives in `docs/PRD.md`.

## 2. Dataset generation

For realisation `r` and component `k ∈ {0, 1, 2, 3}` (frequencies `{1, 2, 5, 10}` Hz):

```
clean_{r,k}(t) = sin(2π f_k t + φ_{r,k})
noisy_{r,k}(t) = clean_{r,k}(t) + ε_{r,k}(t),   ε_{r,k} ~ N(0, σ)
mixture_r(t)   = Σ_k noisy_{r,k}(t)
```

with `φ_{r,k} ~ Uniform[0, 2π)` re-drawn independently per realisation per frequency, and `σ` taken from the four levels listed below (a fraction of unit amplitude `A = 1`). The four per-component noise traces are independent, so the noise component of `mixture_r` has standard deviation `2 · σ · A` (variance four times each component's).

| Parameter | Value |
|---|---|
| Frequencies | `{1, 2, 5, 10}` Hz |
| Sigmas | `{0.05, 0.10, 0.20, 0.30}` (fraction of amplitude) |
| Sampling rate | `fs = 50 Hz` |
| Duration | 10 s → 500 samples per mixture |
| Realisations per σ | 25 |
| Window | 10 samples, **non-overlapping** |
| Queries per window | 4 (one per frequency) |
| Total examples | `4 σ × 25 mixtures × 50 windows × 4 queries = 20,000` |

Splits are stratified by `σ` at the **realisation** level (not the window level, not the query level): all `4 × 50 = 200` records belonging to one mixture stay together. Per σ stratum we use 17 / 4 / 4 mixtures for train / val / test, giving **13,600 / 3,200 / 3,200** examples in total.

The dataset is regenerated deterministically from `seed = 42` and persisted to `data/generated/dataset.npz` plus a `manifest.json` (with `dataset_version: "v2-mixture"`). Full spec in `docs/PRD_dataset.md`.

### Why these four frequencies?

With `fs = 50 Hz` and `T = 10` samples, a single window spans **0.2 s**. The four frequencies cover very different temporal scales **inside that window**:

| f | cycles per 0.2 s window |
|---|---|
| 1 Hz | 0.2 |
| 2 Hz | 0.4 |
| 5 Hz | 1.0 |
| 10 Hz | 2.0 |

The mixture therefore contains both slowly- and rapidly-varying components, every frequency stays safely below the Nyquist limit of 25 Hz, and the per-frequency analysis (§6.4) shows how the model's behaviour changes as the in-window oscillation count grows from 0.2 to 2.

### Mixture-query structure

Every unique `(realisation_id, window_idx)` pair generates **four** dataset records — one per query frequency. The four records share the same `x_noisy` window (the same window of the same noisy mixture) and the same `sigma`; only `C` and `y_clean` change between them:

- record 1 has `C = [1, 0, 0, 0]` → `y_clean` is the clean **1 Hz** component window;
- record 2 has `C = [0, 1, 0, 0]` → `y_clean` is the clean **2 Hz** component window;
- record 3 has `C = [0, 0, 1, 0]` → `y_clean` is the clean **5 Hz** component window;
- record 4 has `C = [0, 0, 0, 1]` → `y_clean` is the clean **10 Hz** component window.

This is the structural reason the task is *conditional component extraction* rather than recovery of an already-isolated component: with a fixed mixture input the model has to use `C` as a query to pick the right one of four overlapping waveforms.

![Mixture query example](assets/report/mixture_query_example.png)

### Dataset correctness checks

The behaviour above is enforced by unit tests in `tests/unit/services/`:

- every `(realisation_id, window_idx)` query group has exactly four records;
- the four `C` vectors are the canonical one-hots `[1,0,0,0]`, `[0,1,0,0]`, `[0,0,1,0]`, `[0,0,0,1]`;
- `x_noisy` is identical across the four records in a group;
- `y_clean` differs across queries (otherwise the extraction would be trivial);
- splits are stratified at the **realisation** level, so all four queries × all 50 windows of one mixture stay together — no leakage across train, val, and test;
- a dedicated `sigma = 0` test verifies that, with no noise, `x_noisy` matches the sum of the four `y_clean` component windows within numerical tolerance for every query group, proving `x_noisy` is the combined mixture rather than any single component.

## 3. Model architectures

All three models share the external signature `forward(x_noisy, C, sigma) -> y_pred ∈ R^10`. Only the way the inputs are combined and the backbone differ.

| Model | Inputs combined | Backbone | Parameters |
|---|---|---|---|
| **FCDenoiser**   | flat `R^15` (concat of `x_noisy`, `C`, `sigma`) | `Linear(15→64) → ReLU → Linear(64→64) → ReLU → Linear(64→10)` | **5,834** |
| **RNNDenoiser**  | per-step `R^6` (broadcast `C` and `sigma` over `T`) | `nn.RNN(6, 32, tanh, batch_first=True)` + `Linear(32→1)` per step | **1,313** |
| **LSTMDenoiser** | per-step `R^6` (same as RNN) | `nn.LSTM(6, 32, batch_first=True)` + `Linear(32→1)` per step | **5,153** |

Output activation: linear (no `tanh` / `sigmoid`). The targets are real-valued samples in roughly `[-1, 1]`, so a linear head is the natural choice.

## 4. Training setup

| Hyperparameter | Value |
|---|---|
| Loss | `nn.MSELoss()` (mean over batch and time) |
| Optimiser | `torch.optim.Adam`, `lr = 1e-3` |
| Batch size | 64 |
| Max epochs | 30 |
| Early-stopping patience | 5 epochs (on validation MSE) |
| Gradient clipping | Not used in the final implementation. |
| Seed | 42 (NumPy, Python `random`, and PyTorch) |
| Device | CPU |

Each model is trained with its own `Adam` instance; the `Trainer` saves only the best-validation `state_dict` to `results/checkpoints/{model}_best.pt`, and exposes `load_best_checkpoint()` for evaluation.

Observed training behaviour (single seeded run, full schedule):

- **FC** early-stopped at epoch 15 (best at epoch 10).
- **RNN** early-stopped at epoch 23 (best at epoch 18).
- **LSTM** early-stopped at epoch 19 (best at epoch 14).

All three models clearly converged within the 30-epoch budget.

## 5. Evaluation metrics

All metrics below are computed on the held-out **test** split (3,200 records, 800 per σ).

- **`mse_overall`** = `mean((y_pred - y_true)²)` across every entry.
- **`mse_per_freq`** = `mse_overall` restricted to records with each query frequency.
- **`mse_per_sigma`** = `mse_overall` restricted to records with each `σ`.
- **`snr_improvement_db`** = `10 · log10(MSE(clean, noisy_mixture) / MSE(clean, predicted))`. Higher is better; `0 dB` means the prediction is no closer to the clean target than the noisy mixture itself.

## 6. Results

### 6.1 Headline metrics

| Model | Test MSE ↓ | SNR improvement (dB) ↑ | Parameters |
|---|---|---|---|
| **FC**   | **0.137753** | **+10.76** | 5,834 |
| **RNN**  | 0.262388     | +7.96      | 1,313 |
| **LSTM** | 0.259095     | +8.02      | 5,153 |

![Overall test MSE](assets/report/overall_test_mse.png)

![SNR improvement](assets/report/snr_improvement.png)

### 6.2 Per-σ test MSE

| Model | σ = 0.05 | σ = 0.10 | σ = 0.20 | σ = 0.30 |
|---|---|---|---|---|
| FC   | 0.0989 | 0.1202 | 0.1505 | 0.1814 |
| RNN  | 0.2478 | 0.2612 | 0.2686 | 0.2719 |
| LSTM | 0.2392 | 0.2586 | 0.2671 | 0.2715 |

![MSE vs sigma](assets/report/mse_by_sigma.png)

### 6.3 Per-frequency test MSE

| Model | f = 1 Hz | f = 2 Hz | f = 5 Hz | f = 10 Hz |
|---|---|---|---|---|
| FC   | 0.2104 | 0.2465 | 0.0627 | 0.0313 |
| RNN  | 0.3026 | 0.3417 | 0.2538 | 0.1514 |
| LSTM | 0.3035 | 0.3387 | 0.2496 | 0.1445 |

![MSE vs frequency](assets/report/mse_by_frequency.png)

### 6.4 Reconstruction example

A single test window with all three model predictions overlaid on the clean component selected by `C` and the noisy mixture window:

![Reconstruction example](assets/report/reconstruction_example.png)

The figure was produced by `evaluation.report.generate_report` from the same trained checkpoints used for the tables above.

## 7. Discussion

**FC achieves the best test MSE and SNR improvement.** At `0.137753 / +10.76 dB` it sits at roughly half the MSE of either recurrent model and ~3 dB better in SNR improvement. The task is genuinely demanding — the model has to isolate one of four overlapping sine components from a single mixed input that also contains the sum of their independent noise traces — yet the FC's flat-input view of the 10-sample window plus the one-hot query and `σ` remains a strong fit at this short horizon. About 5,800 parameters are sufficient.

**LSTM marginally outperforms RNN** (test MSE 0.259095 vs 0.262388; SNR improvement +8.02 vs +7.96 dB). The direction is consistent with the textbook intuition that gating helps gradient flow, but the gap is small at this horizon — there are not many timesteps over which gating can compound. With a single training run we do not claim the gap is statistically significant.

**Per-frequency behaviour is the most informative finding.** For every model, MSE is monotonically lower at higher frequencies: FC drops from 0.21 (1 Hz) to 0.03 (10 Hz), and the recurrent models follow the same trend. This is consistent with a basic signal-processing intuition: at `fs = 50 Hz` the 10-sample window covers only 0.2 s, which is 0.2 cycles of a 1 Hz component — essentially a near-monotonic arc whose extraction from the mixture is highly ambiguous — but 2 full cycles of a 10 Hz component, where periodicity inside the window provides much more information about phase and amplitude. The query `C` tells the model which frequency to reconstruct, but it does not help with the underlying observability problem. The FC's spread between the easiest and the hardest frequency (0.031 vs 0.247) is the largest in absolute terms, which is consistent with its overall lower error: it has more dynamic range to lose at the hard frequencies.

**Per-σ behaviour differs across architectures.** FC's MSE grows from 0.099 to 0.181 as σ rises from 0.05 to 0.30 — about an 83% increase, the slope one would expect from a model that is genuinely denoising and so leaves noise as a sizable share of its remaining error. RNN and LSTM, by contrast, grow only ~10–13% over the same range (RNN 0.248 → 0.272; LSTM 0.239 → 0.272). This flatter curve suggests the recurrent models' error is dominated by structural extraction error rather than by the additive noise term, so adding more noise barely makes the prediction worse on top of what the model is already missing. The FC has lower error overall and is more affected by σ; the recurrent models have higher error overall and are dominated by the extraction component of that error.

**Single seeded run.** All numbers come from a single (seeded, reproducible) run of each model. We do not report standard errors; multi-seed re-runs are listed in §9 as future work.

## 8. Limitations

- **Window length fixed at `T = 10`.** This is a stated part of the assignment but it caps the comparison: any architectural advantage that requires long-range memory cannot manifest at this horizon, and the per-frequency MSE pattern is partly an artefact of how many cycles fit inside 0.2 s.
- **Synthetic data only.** Independent additive Gaussian noise summed onto perfect sinusoids; no real-world signals, no out-of-distribution frequencies, no amplitude variation, no phase coupling between components.
- **No hyperparameter search.** Each model trains once at PRD-specified defaults so the comparison stays fair across architectures, not optimal for any one of them.
- **Single training seed.** Numbers reported come from one run per model; standard error is not estimated.
- **No transformer baseline.** Out of scope for HW1.

## 9. Future work

- Increase the context window (e.g. `T = 50` or `T = 100`) to test whether recurrent gating closes the gap with the FC at the hardest (low-frequency) components.
- Repeat training over multiple seeds and report mean and standard error per model.
- Sliding-window inference with overlap-add reconstruction to give the recurrent models a longer effective context.
- Add a `σ = 0.0` stratum in evaluation to verify each model passes a clean mixture through with negligible distortion.
- Replace independent white Gaussian noise with coloured (e.g. 1/f) noise to test whether the relative ordering changes.
- Add a small 1-D temporal-convolutional baseline as a fourth model.

## 10. How to reproduce

```bash
uv sync

# Build the v2-mixture dataset (writes data/generated/dataset.npz + manifest.json).
uv run python -m sine_denoising.sdk.build_dataset

# Train each architecture in turn (writes results/checkpoints/{model}_best.pt).
uv run python -m sine_denoising.sdk.train --model fc
uv run python -m sine_denoising.sdk.train --model rnn
uv run python -m sine_denoising.sdk.train --model lstm

# Evaluate on the test split (writes results/summary.json + a reconstruction PNG).
uv run python -m sine_denoising.sdk.evaluate

# Quality gate.
uv run pytest -q
uv run ruff check .
uv run mypy src
```

The dataset, checkpoints, and `results/summary.json` are gitignored. The figures used in this README are committed under `assets/report/`.

## 11. Repository structure

```
.
├── assets/
│   ├── report/                # final figures committed for the lab report
│   └── generated/             # transient evaluator output — gitignored
├── config/                    # YAML configs (dataset, training)
├── data/
│   └── generated/             # built dataset (.npz + manifest.json) — gitignored
├── docs/
│   ├── ASSIGNMENT_NOTES.md
│   ├── PRD.md                 # top-level PRD
│   ├── PRD_dataset.md         # dataset spec
│   ├── PRD_models.md          # model + training spec
│   ├── PLAN.md                # implementation plan
│   └── TODO.md                # phase-by-phase checklist
├── results/                   # checkpoints + summary.json — gitignored
├── src/sine_denoising/
│   ├── shared/                # config (pydantic), seeding, types
│   ├── services/              # signal generation, encoding, mixtures, dataset builder/loader
│   ├── models/                # base + fc + rnn + lstm
│   ├── training/              # loops + Trainer (early stopping + checkpoints)
│   ├── evaluation/            # metrics, plotting, report generator
│   └── sdk/                   # CLI entry points (build_dataset, train, evaluate)
├── tests/
│   ├── unit/                  # mirrors src/
│   └── integration/           # reserved for end-to-end smoke tests
├── pyproject.toml
├── README.md                  # this file
└── uv.lock
```

# PRD — Models

## 1. Purpose
Specify the three model architectures (FC, RNN, LSTM), their shared training contract, and the comparison protocol. The goal is a *fair* comparison: only the recurrent backbone changes; everything else (data, loss, optimizer family, training budget, evaluation) is held constant.

## 2. Shared training contract

### 2.1 Common input/output
All three models implement the same external signature:

```
forward(x_noisy: (B, 10), C: (B, 4), sigma: (B, 1)) -> y_pred: (B, 10)
```

`B` is the batch size. `x_noisy` is a 10-sample window of the noisy mixture (see `docs/PRD_dataset.md` §2), `C` is the one-hot query selecting which of the four frequency components to reconstruct, and `y_pred` is the predicted clean window of that selected component. The internal way the inputs are combined differs by architecture (see §3–§5).

### 2.2 Loss and metrics
- Training loss: `MSE(y_pred, y_clean)` (mean over batch and time).
- Reported metrics on val/test:
  - `MSE_overall` (mean over all examples)
  - `MSE_per_freq` (broken down by `freq_idx`)
  - `MSE_per_sigma` (broken down by sigma bucket)
  - `SNR_improvement_dB` = `10 * log10(MSE_input / MSE_output)`, where `MSE_input = MSE(x_noisy, y_clean)`. This shows how much denoising the model actually does relative to the noisy input.

### 2.3 Optimizer and schedule
- Optimizer: `torch.optim.Adam`.
- Learning rate: `1e-3` (default Adam) — same across models. Only adjust if a model fails to learn at all; document any deviation in the lab report.
- Batch size: `64`.
- Epochs: `30` (initial budget; revisit only if all models clearly haven't converged).
- Early stopping on validation loss with patience `5` epochs.
- Gradient clipping: `clip_grad_norm_` at `1.0` for the recurrent models (RNN/LSTM), to defend against rare exploding-gradient spikes; FC does not need it but applying it uniformly is harmless.

### 2.4 Reproducibility
- Seed numpy, Python's `random`, and torch (CPU + CUDA if available) at the start of every run.
- Persist the chosen seed and all hyperparameters into the run's results folder.

### 2.5 Hardware
- Default device: CPU. The dataset and models are tiny enough that CPU training is acceptable. CUDA is auto-used when available.

## 3. Fully Connected (FC) model

### 3.1 Inputs
Concatenate everything into a flat vector of dimension `T + K + 1 = 10 + 4 + 1 = 15`:
```
z = concat(x_noisy, C, sigma) ∈ R^15
```

### 3.2 Architecture (initial proposal)
- Linear `15 → 64` + ReLU
- Linear `64 → 64` + ReLU
- Linear `64 → 10`

Approximate parameter count: `15·64 + 64 + 64·64 + 64 + 64·10 + 10 ≈ 5,834`.

### 3.3 Why this size
It must be small enough to be a fair baseline (so its capacity advantage from "seeing all inputs at once" is not overwhelming) and large enough to fit a non-trivial mapping. Two hidden layers of 64 strike that balance; the report will run a brief ablation if results are surprising.

## 4. Vanilla RNN model

### 4.1 Inputs
At each timestep `t ∈ {0, …, 9}` the RNN sees a vector of dimension `1 + K + 1 = 6`:
```
u_t = [x_noisy[t], C, sigma]   ∈ R^6
```
`C` and `sigma` are broadcast (repeated) across all 10 timesteps.

### 4.2 Architecture
- `nn.RNN(input_size=6, hidden_size=32, num_layers=1, nonlinearity="tanh", batch_first=True)`
- Linear head: `Linear(hidden_size, 1)` applied at every timestep, producing `(B, 10, 1)` then squeezed to `(B, 10)`.

Approximate parameter count: `(6·32 + 32·32 + 32) + (32·1 + 1) ≈ 1,281`.

### 4.3 Why this size
Hidden size 32 is small but sufficient for a 10-step sine reconstruction; matching the recurrent and FC models exactly on parameter count is not the goal — the goal is to compare *architecture families* at sensible defaults.

## 5. LSTM model

### 5.1 Inputs
Same as the RNN: `u_t = [x_noisy[t], C, sigma] ∈ R^6` per timestep.

### 5.2 Architecture
- `nn.LSTM(input_size=6, hidden_size=32, num_layers=1, batch_first=True)`
- Linear head: `Linear(hidden_size, 1)` per timestep → `(B, 10, 1)` → `(B, 10)`.

Approximate parameter count: ~`5,025` (LSTM has 4× the gates of a vanilla RNN).

### 5.3 Why this size
Same hidden size as the RNN so the comparison highlights the gating mechanism rather than width. Parameter count is closer to the FC, which is fine — we are not claiming a parameter-matched study.

## 6. Initialization
- Linear layers: PyTorch defaults (Kaiming uniform).
- Recurrent layers: PyTorch defaults; we do not override.
- Biases: PyTorch defaults.

If the RNN diverges, fall back to orthogonal initialization for the hidden-to-hidden matrix; document the change.

## 7. Output activation
None. The targets are real-valued sine samples in roughly `[-1, 1]`, so a linear output is appropriate. Adding a `tanh` would clamp outputs but also bias the gradient near saturation; we prefer the linear head and rely on MSE.

## 8. File layout (target)
Each model lives in its own file to satisfy the 150-line rule and keep diffs reviewable:

- `src/sine_denoising/models/fc.py`
- `src/sine_denoising/models/rnn.py`
- `src/sine_denoising/models/lstm.py`
- `src/sine_denoising/models/base.py` — common interface (signature, parameter count helper).
- `src/sine_denoising/models/__init__.py` — exports.

Trainer and evaluation live under `src/sine_denoising/training/` and `src/sine_denoising/evaluation/` respectively (also under 150 lines each).

## 9. Comparison protocol
1. Train each of the three models on the same `(seed, train, val)` data with the shared schedule (§2.3).
2. Pick the checkpoint with the lowest val loss (early stopping does this naturally).
3. Evaluate all three on the same test set; report `MSE_overall`, `MSE_per_freq`, `MSE_per_sigma`, `SNR_improvement_dB`.
4. Pick three test windows per frequency (one each at `sigma ∈ {0.05, 0.20, 0.30}`) and plot `clean / noisy / FC / RNN / LSTM` overlaid.
5. Tabulate all results in the README's "Results" section.

## 10. Hyperparameters that may move
The following are the only values we may revisit if results look pathological:
- Learning rate (within `[1e-4, 5e-3]`).
- Hidden size (within `[16, 64]` for RNN/LSTM).
- Epoch budget (within `[15, 60]`).

Any change is logged in the lab report alongside the reason. We do **not** touch frequencies, sigmas, `fs`, `T`, `K`, the optimizer family, or the loss.

## 11. Open questions
- Do we need a final non-linearity for stability at high sigma? Initial bet: no. Reconsider if the LSTM saturates.
- Do we report parameter counts in the comparison table? Yes — it makes the "free-lunch / no-free-lunch" reading honest.

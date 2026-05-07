# PRD — HW1: Sine/Cosine Signal Denoising with FC, RNN, and LSTM

## 1. Background and motivation
This homework is part of a university course on AI agent orchestration and deep learning. The pedagogical goal is to compare three classes of neural networks — a Fully Connected (FC) network, a vanilla Recurrent Neural Network (RNN), and a Long Short-Term Memory (LSTM) network — on the same supervised regression task: reconstruct a clean sine-based signal from a noisy version of it, given the underlying frequency (one-hot encoded) and the noise level `sigma`.

The task is small enough that we can fully control the data-generating process, which lets us isolate the effect of model architecture. It also lets us reason about classical signal-processing intuitions (Nyquist, signal-to-noise ratio) while practising the engineering discipline expected in the course (uv-managed environment, modular `src/` layout, tests, lab-report style README).

## 2. Problem statement
Given:
- a context window of `T = 10` consecutive samples `x_noisy ∈ R^T` taken from a noisy mixture that sums one independently noised realisation of each of the four known frequency components,
- a one-hot vector `C` of length `K = 4` acting as a query that selects which frequency component to reconstruct,
- a scalar `sigma` representing the noise standard deviation applied to each component (as a fraction of amplitude `A`),

predict the clean `T = 10` samples `y ∈ R^T` of the component selected by `C`. The loss function is mean squared error (MSE) between prediction and ground truth.

Three models will be trained on identical data and compared:
1. Fully Connected feed-forward network.
2. Vanilla RNN (Elman) with a final linear projection head.
3. LSTM with a final linear projection head.

## 3. Goals
- **G1.** A reproducible synthetic dataset of (one-hot, sigma, noisy window, clean window) tuples that respects the Nyquist criterion.
- **G2.** Three trained PyTorch models (FC, RNN, LSTM) sharing the same inputs/targets and the same loss.
- **G3.** A clear, fair comparison: identical train/val/test splits, identical optimizer family, identical training budget.
- **G4.** Evaluation: per-model and per-frequency MSE on the test set, plus qualitative plots (clean vs noisy vs reconstructed) for at least one example per frequency.
- **G5.** A README written as a lab report (intro, method, experiments, results, discussion, conclusion).
- **G6.** Engineering quality: every Python file under 150 lines, modular `src/` layout, ≥ 85% unit-test coverage with pytest.

## 4. Non-goals
- No real-world signal data (only synthetic).
- No exhaustive hyperparameter search; we tune only enough to demonstrate fair comparison.
- No deployment, no API, no UI. The deliverables are the repository and a PDF report.
- No transformer or attention baselines — out of scope for HW1.

## 5. Stakeholders and audience
- **Primary:** the course instructor, who will read the PDF report and possibly inspect the repository.
- **Secondary:** the student (myself), as a study artefact for future reference.

## 6. Working interpretation of the assignment
The dataset entry pairs a noisy mixture window with the clean component identified by a one-hot query. The interpretation we adopt is:

> Each model receives `(x_noisy, C, sigma)` and predicts `y_clean` of the same length as `x_noisy`. `x_noisy` is a 10-sample window of the noisy mixture, `C` selects which of the four known frequencies to reconstruct, and `y_clean` is the clean window of that selected component. Training minimises `MSE(y_pred, y_clean)`.

For RNN/LSTM the noisy mixture window is fed sequentially (one sample per timestep); `C` and `sigma` are broadcast and concatenated to every timestep, so the recurrent layer sees `[x_t, C, sigma]` of dimension `1 + K + 1 = 6` at each step. For FC, the entire window is flattened together with `C` and `sigma` into a vector of dimension `T + K + 1 = 15`.

## 7. Key assumptions (and why)
The assignment is partially open. For each open choice we record the choice and the reason.

- **A1. Four frequencies = `{1, 2, 5, 10}` Hz.** With `fs = 50 Hz` and a 10-sample window, the window length is 0.2 s and the four frequencies span very different temporal scales inside that window: 1 Hz fits 0.2 cycles, 2 Hz fits 0.4 cycles, 5 Hz fits exactly 1 cycle, and 10 Hz fits 2 cycles. The mixture therefore contains both slowly varying and rapidly varying components, while every frequency stays safely below the Nyquist limit.
- **A2. Amplitude `A = 1.0`.** Conventional unit amplitude; lets `sigma` be read directly as an SNR-like quantity.
- **A3. Sampling rate `fs = 50 Hz`.** Comfortably above Nyquist (`2 × 10 = 20 Hz`); 5× oversampling gives the network enough resolution per period at the highest frequency without inflating the dataset.
- **A4. Signal length = 10 s.** Stated by the assignment; yields `10 × 50 = 500` samples per generated signal.
- **A5. Sigma grid = `{0.05, 0.10, 0.20, 0.30}` of `A`.** Four discrete noise levels covering low to moderately high noise; matches the "percentage of amplitude" wording.
- **A6. Each realisation is a noisy mixture of all four components.** Per realisation we draw an independent random phase `φ_k ~ Uniform[0, 2π)` for each frequency component (so the network cannot memorise a fixed waveform), add independent Gaussian noise `ε_k ~ N(0, σA)` to each clean component, and sum the four noisy components into one combined noisy signal. The model's task is to extract the clean component identified by `C` from this mixture, which makes the setup a compact conditional source-extraction problem rather than recovery of an already-isolated component.
- **A7. Context windowing = non-overlapping slices of length 10.** Cleaner train/test separation and avoids near-duplicates between adjacent windows; assumption stated explicitly because the assignment does not specify stride.
- **A8. Multiple mixture realisations per `sigma`.** Each `sigma` value generates `N_realisations` independent mixtures (fresh phases per component, fresh noise draws). From every mixture window we then emit `K = 4` training records — one per query frequency. Exact `N_realisations` chosen in `PRD_dataset.md`.
- **A9. Train/val/test split = 70/15/15 by mixture realisation, stratified by `sigma`.** Splitting by realisation (not by window or by query) prevents leakage where two windows from the same mixture end up in different splits. Strata are now `sigma` only because each realisation already covers all four frequencies.
- **A10. Loss = MSE; optimizer = Adam; framework = PyTorch.** Assignment fixes MSE; PyTorch is the de-facto choice and is already in `pyproject.toml`.

## 8. Inputs and outputs (contract)
- **Model input:** `(x_noisy, C, sigma)` with `x_noisy ∈ R^10` (a 10-sample window of the noisy mixture), `C ∈ {0,1}^4` (exactly one 1 — the query frequency), `sigma ∈ R` (the per-component noise level used to build the mixture).
- **Model output:** `y_pred ∈ R^10`.
- **Training target:** `y_clean ∈ R^10` (the clean window of the component selected by `C`).
- **Loss:** `MSE(y_pred, y_clean)`.

## 9. Success criteria
- All three models train to convergence (validation loss plateaus) within the chosen budget.
- The comparison does not assume a fixed ranking between FC, RNN, and LSTM. The context window is short (`T = 10`) but the model must now use the one-hot query `C` to extract one of four overlapping components from the mixture, so the relative strengths of the architectures may differ from a simpler isolated-component baseline. The report will present the measured results honestly and discuss why each architecture performs as it does.
- Test-set plots clearly show the reconstructed signal tracking the clean signal at low/medium sigma and degrading gracefully at high sigma.
- Repository builds with `uv sync` and `pytest` runs green with ≥ 85% coverage.

## 10. Risks and mitigations
- **R1. RNN/LSTM may underperform FC on length-10 windows** because the temporal context is short. Mitigation: report honestly, discuss why (limited memory horizon), and show qualitative behaviour as well as raw MSE.
- **R2. File-size limit of 150 lines.** Mitigation: split modules early (one model per file, separate trainer, separate dataset module).
- **R3. Coverage target ≥ 85%.** Mitigation: write small pure functions for signal generation, encoding, windowing, and metrics — these are easy to unit-test even without training.
- **R4. Reproducibility.** Mitigation: seed numpy and torch; persist generated dataset to `data/generated/` so subsequent training runs use the same examples.

## 11. Deliverables
- This repository (code, tests, docs).
- A PDF report containing the explanation and a link to the repository (built from the README content).
- The trained model checkpoints under `results/` (gitignored if too large; otherwise the smallest informative artefact only).

## 12. Out-of-scope clarifications
- Any change to the four chosen frequencies, the four sigma values, `fs`, or `T` after training begins must be flagged in the README's "Limitations" section, not silently swapped.

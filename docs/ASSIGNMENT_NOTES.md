# HW1 Assignment Notes

## Goal
Build a Python deep learning project that compares Fully Connected, RNN, and LSTM models on a signal reconstruction task.

## Signal task
Create synthetic sine/cosine-based signals.
Each signal should be generated in two versions:
1. Pure signal
2. Noisy signal

The model receives noisy samples and should reconstruct or predict the clean samples.

## Dataset requirements
Each dataset entry should contain:
- C: one-hot encoded vector representing the selected frequency
- sigma: noise level, interpreted as a percentage of amplitude A
- noisy samples S_c for the selected frequency
- clean samples as the target

Known assignment constraints:
- Use 4 known frequencies chosen by us.
- Generate a 10-second signal window.
- Context window is 10 samples.
- Each entry includes 10 noisy samples and 10 clean samples.
- Sampling rate must respect the Nyquist idea: at least twice the maximum frequency.
- The loss function is MSE between prediction and truth.

## Models to compare
1. Fully Connected network
2. RNN
3. LSTM

## Software/project constraints
- Work from terminal / CLI.
- Use Claude Code CLI as the AI assistant.
- Use uv for Python environment and dependencies.
- Keep every Python file under 150 lines.
- Add unit tests.
- README should be a detailed lab report.
- PDF should contain the explanation and a link to the GitHub repository.
- When something is not explicitly defined, we may choose it, but we must explain and justify the choice.

## Important ambiguity to handle
The assignment mentions both:
- 10-second signal window
- 10-sample context window

We will interpret this as:
- Generate a full 10-second signal.
- Slice it into training examples, where each example has 10 samples.

## Initial design assumption
Input:
- 10 noisy samples
- frequency one-hot vector
- sigma value

Target:
- 10 clean samples

Evaluation:
- Compare MSE of Fully Connected, RNN, and LSTM.
- Produce plots of clean vs noisy vs reconstructed signals.
- Summarize results in README and final PDF.

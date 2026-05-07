# TODO — HW1 Checklist

A flat checklist mirroring `PLAN.md`. Tick boxes as we go. Group titles match the plan phases.

## Phase 0 — Planning and skeleton
- [x] Read `docs/ASSIGNMENT_NOTES.md` and pull constraints into PRDs.
- [x] Write `docs/PRD.md`.
- [x] Write `docs/PRD_dataset.md`.
- [x] Write `docs/PRD_models.md`.
- [x] Write `docs/PLAN.md`.
- [x] Write `docs/TODO.md`.
- [x] Initialize `README.md` as a lab-report outline.
- [x] Commit Phase 0 artefacts.
- [ ] Confirm interpretation with the instructor before any code is written.

## Phase 1 — Configuration and shared utilities
- [x] `src/sine_denoising/shared/config.py` (pydantic configs).
- [x] `src/sine_denoising/shared/seeding.py` (`set_seed`).
- [x] `src/sine_denoising/shared/types.py` (split-array record types).
- [x] `config/dataset.yaml`, `config/training.yaml`.
- [x] `tests/unit/shared/test_config.py`.
- [x] `tests/unit/shared/test_seeding.py`.
- [x] `tests/unit/shared/test_types.py`.
- [x] Commit Phase 1.

## Phase 2 — Dataset generation and loading
- [x] `src/sine_denoising/services/signal.py` (`make_signal`).
- [x] `src/sine_denoising/services/encoding.py` (`one_hot`, `window`).
- [x] `src/sine_denoising/services/dataset_builder.py`.
- [x] `src/sine_denoising/services/dataset_loader.py`.
- [x] `src/sine_denoising/sdk/build_dataset.py` CLI.
- [x] Generate `data/generated/dataset.npz` once and verify.
- [x] `tests/unit/services/test_signal.py`.
- [x] `tests/unit/services/test_encoding.py`.
- [x] `tests/unit/services/test_dataset_builder.py` (split disjointness, stratification).
- [x] `tests/unit/services/test_dataset_loader.py`.
- [x] Commit Phase 2.

## Phase 3 — Models
- [x] `src/sine_denoising/models/base.py`.
- [x] `src/sine_denoising/models/fc.py`.
- [x] `src/sine_denoising/models/rnn.py`.
- [x] `src/sine_denoising/models/lstm.py`.
- [x] Update `src/sine_denoising/models/__init__.py` exports.
- [x] `tests/unit/models/test_fc.py` (shape + smoke).
- [x] `tests/unit/models/test_rnn.py`.
- [x] `tests/unit/models/test_lstm.py`.
- [x] `tests/unit/models/test_model_common.py`.
- [x] Commit Phase 3.

## Phase 4 — Training loop
- [x] `src/sine_denoising/training/loops.py`.
- [x] `src/sine_denoising/training/trainer.py`.
- [x] `src/sine_denoising/sdk/train.py` CLI.
- [x] `tests/unit/training/test_loops.py` (overfitting test on a tiny batch).
- [x] `tests/unit/training/test_trainer.py` (early stopping, checkpoint round-trip).
- [x] Run training for FC, RNN, LSTM (full schedule).
- [x] Save checkpoints under `results/`.
- [x] Commit Phase 4.

## Phase 5 — Evaluation and plotting
- [x] `src/sine_denoising/evaluation/metrics.py`.
- [x] `src/sine_denoising/evaluation/plots.py`.
- [x] `src/sine_denoising/evaluation/report.py`.
- [x] `src/sine_denoising/sdk/evaluate.py` CLI.
- [x] `tests/unit/evaluation/test_metrics.py`.
- [x] `tests/unit/evaluation/test_plots.py` (output file exists, non-empty).
- [x] `tests/unit/evaluation/test_report.py`.
- [x] `tests/unit/sdk/test_evaluate_cli.py`.
- [x] Run evaluation, write `results/summary.json` + figures.
- [x] Commit Phase 5.

## Phase 6 — Lab report
- [x] Fill in README "Experimental setup" with concrete config values.
- [x] Fill in "Results" table with real numbers from `summary.json`.
- [ ] Embed loss-curve figure(s).
- [x] Embed reconstruction figures.
- [x] Write "Discussion" grounded in the actual numbers (no boilerplate).
- [x] Write "Conclusion" + "Limitations" + "Future work".
- [x] Add link to the GitHub repository.
- [x] Commit Phase 6.

## Phase 7 — Quality gate and packaging
- [x] `uv sync` runs clean.
- [x] `uv run ruff check .` passes.
- [x] `uv run mypy src` passes.
- [x] `uv run pytest --cov` ≥ 85%.
- [x] Sanity-check that every Python file is < 150 lines.
- [x] Final commit.

## Phase 8 — Correct dataset to noisy mixture extraction
- [x] Update docs to the final mixture-extraction formulation (`docs/PRD.md`, `docs/PRD_dataset.md`, `docs/PRD_models.md`).
- [x] Update `src/sine_denoising/services/dataset_builder.py` to emit `(window_idx, query_freq)` records over a noisy mixture (with `realisation_id` and `window_idx` carried as bookkeeping fields).
- [x] Update `tests/unit/services/test_dataset_builder.py` for the new schema and sigma-only stratification.
- [x] Add mixture-invariant tests for shared `x_noisy`, query-specific `y_clean`, sigma-zero mixture equality, production counts, and per-sigma stratification.
- [x] Rebuild `data/generated/dataset.npz`.
- [x] Retrain FC, RNN, LSTM on the rebuilt dataset.
- [x] Re-run `sdk.evaluate` to refresh `results/summary.json`.
- [x] Regenerate the reconstruction figure and refresh `assets/report/reconstruction_example.png`.
- [x] Update `README.md` with the final mixture-extraction results.
- [x] Run the full quality gate (`ruff check .`, `mypy src`, `pytest -q`, file-size check).
- [x] Commit and push Phase 8.

## Phase 9 — README enrichment and final polish
- [x] Add `assets/report/mixture_query_example.png` to visualize the same noisy mixture window with four query-specific clean targets.
- [x] Add README explanation for the `(realisation_id, window_idx)` mixture-query structure.
- [x] Add README dataset-correctness checks explaining shared `x_noisy`, query-specific `y_clean`, split-by-realisation, and the `sigma = 0` invariant test.
- [x] Run the full quality gate after README enrichment.

## Phase 10 — Align sampling with 10,000-sample signal setup
- [x] Update docs/config to use fs=1000 and 10,000 samples per realisation.
- [x] Update dataset count and stratification tests for 400,000 records.
- [x] Rebuild dataset with fs=1000.
- [x] Retrain FC, RNN, LSTM.
- [ ] Re-run evaluation.
- [ ] Regenerate report figures.
- [ ] Update README with final fs=1000 results and analysis.
- [ ] Run full quality gate.
- [ ] Commit and push Phase 10.

## Cross-cutting reminders
- After every phase: `git status`, `git diff`, then a focused commit. No omnibus WIP commits.
- If any assumption from the PRDs changes, edit the PRD *first*, then the code.
- If a file approaches 150 lines, split it before merging.
- Generated artefacts go to `data/generated/`, `results/`, `assets/generated/` (all gitignored).

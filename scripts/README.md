# Scripts

This directory contains the training, evaluation, verification, and final report
builders used for the GRPO coursework submission.

## Core Training

- `config.py`: shared hyperparameters and local paths.
- `data.py`: GSM8K prompt and split handling.
- `rewards.py`: baseline and format-aware reward functions.
- `model.py`: model, tokenizer, mesh, and LoRA setup.
- `train.py`: GRPO training entry point.
- `evaluate.py` and `evaluate_checkpoints.py`: checkpoint evaluation.
- `grpo_observability.py`: scalar and rollout trace logging helpers.

## Final Submission Utilities

- `verify_experiment_package.py`: validates one fetched TPU run.
- `build_rollout320_official_comparison_package.py`: builds
  `artifacts/reports/final-comparison`.
- `build_rollout320_report_figures.py`: builds
  `artifacts/reports/final-figures`.
- `rollout320_report_utils.py`: shared rollout-aligned report helpers.

Historical exploration builders have been moved to `archive_local/` so the
submission-facing source tree only exposes final R0-R6 workflows.

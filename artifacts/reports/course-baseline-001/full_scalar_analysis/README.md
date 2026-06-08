# Full scalar analysis for `course-baseline-001`

This folder fixes the important limitation in the first summary: checkpoint eval starts at the saved/fetched checkpoints, but the scalar training/eval metrics peak much earlier.

## No-drop tables

- `tables/full_scalar_long.csv`: every selected TensorBoard scalar row, no downsampling.
- `tables/full_scalar_pivot.csv`: one row per step with metric columns.
- `tables/scalar_peak_summary.csv`: max/min/latest for every selected scalar, including peak step.
- `tables/key_scalar_peak_summary.csv`: compact subset for report discussion.

## Key corrections

- `eval_reward_score` peaks at step 448 with value 6.84525; latest is -0.960227.
- `eval_numeric_exact_rate` peaks at step 256 with value 0.540107; latest is 0.0815508.
- `eval_format_accuracy` peaks at step 704 with value 0.918449; latest is 0.129679.
- `train_reward_score` peaks at step 157 with value 10; latest is 6.25.

## Interpretation guardrail

Early scalar peaks do not automatically mean the model checkpoint at that exact step can be evaluated. They show when the training/eval signals were highest. A checkpoint-level claim requires a saved and restorable checkpoint at or near that step.

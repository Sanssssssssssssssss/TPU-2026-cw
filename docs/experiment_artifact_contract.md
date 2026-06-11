# Experiment Artifact Contract

This repository must keep every reported training or evaluation run auditable.
For each run referenced in the report, the local or linked artifact package
must include the files below.

## Required Run Package

Each run under `artifacts/cloud/<run_id>/` or an equivalent external storage
prefix should contain:

- `pipeline.log`: full remote orchestration log, including start/end markers.
- `artifacts/*manifest*.json`: run id, source commit, dirty patch status,
  reward mode, LoRA rank/alpha, GRPO hyperparameters, TPU/project metadata,
  checkpoint/eval schedule, and output locations.
- `runs/<branch>/run_env.txt`: exact environment variables used for training.
- `runs/<branch>/train.log`: stdout/stderr from `scripts/train.py`.
- `runs/<branch>/tensorboard/`: raw TensorBoard event files.
- `runs/<branch>/artifacts/rollout_traces/*.jsonl`: sampled rollout traces.
- `runs/<branch>/artifacts/checkpoint_eval/`: aggregate eval JSON/CSV.
- `runs/<branch>/artifacts/checkpoint_eval_examples/`: per-example JSONL eval
  outputs when available.
- `checkpoint_archives/` and `checkpoint_archives.txt`: local checkpoint
  archives fetched per branch/step, or a clear note explaining why checkpoints
  are intentionally not archived.
- `artifacts/reports/<run_id>-clean/`: clean plots and tables generated from
  raw logs, not manually edited screenshots.

## Required Report Links

Every URL in the submitted report must be clickable. This includes:

- GitLab repository used for submission.
- GitHub mirror, if cited.
- W&B dashboard or run, if used.
- TensorBoard or exported TensorBoard artifact, if used.
- Dataset cards, papers, upstream repositories, and external baselines.

## Minimum Metrics

Training comparison plots should include the same metric families for all
reward experiments whenever available:

- `reward_score_mean` and reward component means, split into numeric/task and
  format/hygiene components.
- `KL`, `loss`, `pg_clipfrac`.
- `numeric_exact_rate`, `format_accuracy`, `partial_accuracy`.
- `empty_response_rate`, `extracted_none_rate`, `no_close_answer_rate`.
- `reward_std`, `frac_reward_zero_std`, `advantage_std`.
- `completion_length` and overlong rates.

## Current R64 Note

`scripts/config.py` defaults to `RANK=64` and `ALPHA=64.0`. Cloud workflows
should still export and record `RANK=64` and `ALPHA=64` explicitly in
`run_env.txt` and the run manifest so the report can prove which LoRA adapter
configuration was used.

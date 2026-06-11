# R12 Reward-Only Completion Runbook

This document records the reproducible path for completing the R12 reward-only
line after the original stopped run.

## Canonical run ids

- Stopped source run: `reward-only-r12-full-001`
- Completion run: `reward-only-r12-full-complete-001`
- Source experiment: `R12_reward_only_baseline_kkl`
- Source checkpoint: `reward-only-r12-full-001/runs/R12_reward_only_baseline_kkl/ckpts/actor/500`

## Why a new run id is used

The original reward-only run was intentionally stopped after checkpoint `500`.
It is retained as a traceable stopped negative-ablation run, not as a completed
3364-step result.

An attempted generic `resume-k8-pilot` restart was immediately stopped because
that path re-ran `run_k8_pilot.sh` without the original submission-time
environment overrides. It briefly used the default K8 pilot parameters rather
than the intended K=2 reward-only configuration. The completion line therefore
uses a fresh run id and explicitly seeds from checkpoint `500`.

## Submit command

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\cloud\submit_tpu_job.ps1 submit-reward-only-r12-complete-from500 -RunId reward-only-r12-full-complete-001
```

## Intended training configuration

- Reward mode: `gsm8k_verifiable_simple`
- Branch/run name: `R12_reward_only_baseline_kkl`
- Source checkpoint root: `~/tpu-runs/reward-only-r12-full-001/runs/R12_reward_only_baseline_kkl/ckpts/actor`
- Source step: `500`
- Max steps: `3364`
- LR schedule steps: `3364`
- Warmup steps: `336.4`
- Checkpoint/eval steps: `500 1000 1500 2000 2500 3000 3364`
- Number of generations: `2`
- Beta: `0.08`
- Learning rate: `3e-6`
- LoRA rank: `64`
- LoRA alpha: `64`
- Epsilon: `0.2`

The remote runner records these values in:

- `pipeline.log`
- `artifacts/reward_k8_pilot_manifest.json`
- `runs/R12_reward_only_baseline_kkl/run_env.txt`
- `runs/R12_reward_only_baseline_kkl/branch_metadata.json`
- `meta/git_commit.txt`
- `meta/git_status.txt`
- `meta/dirty.patch`

## Completion workflow

When the run finishes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\cloud\submit_tpu_job.ps1 fetch-k8-pilot -RunId reward-only-r12-full-complete-001
py -3 scripts\build_reward_sweep_clean_plots.py --input-dir artifacts\cloud\reward-only-r12-full-complete-001 --output-dir artifacts\reports\reward-only-r12-full-complete-001-clean --rolling-window 64
py -3 scripts\build_three_line_grpo_package.py
py -3 scripts\verify_experiment_package.py artifacts\cloud\reward-only-r12-full-complete-001
```

If TensorBoard scalar tables are unavailable locally, use
`scripts\build_stopped_reward_clean_plots.py` as the trace-based fallback until
the full scalar export is present.

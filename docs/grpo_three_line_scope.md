# GRPO Three-Line Scope

We are keeping three canonical lines for the GRPO analysis:

1. `course-baseline-001`
   - Purpose: course baseline.
   - Reward mode: default `baseline` reward in `scripts/rewards.py`.
   - Status: complete local raw/report package.
   - Local raw: `artifacts/cloud/course-baseline-001`
   - Local report: `artifacts/reports/course-baseline-001`

2. `reward-only-r12-full-complete-001`
   - Purpose: R12 reward-only ablation against the course baseline training setup.
   - Reward mode: `gsm8k_verifiable_simple`.
   - Status: complete local raw/report package; seeded from stopped source run checkpoint `500`.
   - Local raw: `artifacts/cloud/reward-only-r12-full-complete-001`
   - Local report: `artifacts/reports/reward-only-r12-full-complete-001-clean`
   - Source run raw was cleaned after evidence packaging; seed provenance is
     preserved in `artifacts/reports/grpo-three-line-evidence-001/raw_refs/r12_reward_only_source`
     and in the completion run's copied checkpoint metadata.
   - Completion runbook: `docs/r12_reward_only_completion.md`

3. `reward-k8-beta004-r12-full-001`
   - Purpose: best R12 full run.
   - Reward mode: `gsm8k_verifiable_simple`.
   - Status: complete local raw/report package.
   - Local raw: `artifacts/cloud/reward-k8-beta004-r12-full-001`
   - Local report: `artifacts/reports/reward-k8-beta004-r12-full-001-clean`

## Reward-Only Versus Baseline

For the reward-only line, the core training setup is aligned to the baseline and
the reward mechanism is the intended experimental change:

| Field | Baseline | R12 reward-only |
| --- | --- | --- |
| `REWARD_MODE` | `baseline` default | `gsm8k_verifiable_simple` |
| `MAX_STEPS` | `3364` | `3364` |
| `NUM_GENERATIONS` | `2` | `2` |
| `TOTAL_GENERATION_STEPS` | `768` | `768` |
| `LEARNING_RATE` | `3e-6` | `3e-6` |
| `BETA` | `0.08` | `0.08` |
| `EPSILON` | `0.2` | `0.2` |
| `RANK` | `64` | `64` |
| `ALPHA` | `64` | `64` |
| `SAVE_INTERVAL_STEPS` | `500` | `500` |
| `EVAL_EVERY_N_STEPS` | `64` | `64` |
| `WARMUP_STEPS` | `336.4` | `336.4` |

Known non-core differences:

- `MAX_TO_KEEP` is larger for reward-only (`16` versus baseline `4`) so more
  checkpoint evidence is retained.
- Reward-only records denser rollout traces for diagnostics. This changes
  observability, not the reward function or GRPO optimizer configuration.
- The completion run is seeded from the stopped reward-only checkpoint `500`;
  the stopped source run used the same reward-only training parameters above.

## R12 Full Is A Separate Full-Run Line

The R12 full run is intentionally not baseline-matched. It is the best full run
line and uses:

- `NUM_GENERATIONS=8`
- `BETA=0.04`
- `MAX_STEPS=841`
- `LR_SCHEDULE_STEPS=841`
- `SAVE_INTERVAL_STEPS=128`

Keep it as the performance line, while using the reward-only line for the clean
reward-mechanism ablation against baseline.

## Local Cleanup Scope

Top-level local report directories have been reduced to:

- `artifacts/reports/course-baseline-001`
- `artifacts/reports/reward-only-r12-full-complete-001-clean`
- `artifacts/reports/reward-k8-beta004-r12-full-001-clean`
- `artifacts/reports/r12-full-autotune-tail512-001-clean`
- supporting evidence packages: `grpo-three-line-evidence-001`,
  `r12-final-evidence-001`, `r12-full-autotune-evidence-tail512-001`

Non-canonical historical report outputs were moved to:

- `artifacts/_archive_noncanonical_reports_20260611-231418`

Top-level local raw cloud artifact directories have been reduced to:

- `artifacts/cloud/course-baseline-001`
- `artifacts/cloud/reward-only-r12-full-complete-001`
- `artifacts/cloud/reward-k8-beta004-r12-full-001`
- `artifacts/cloud/r12-full-autotune-tail512-001`

Supporting large-eval tables and old reward-only source provenance are retained
inside `artifacts/reports/grpo-three-line-evidence-001`; their top-level raw
directories were cleaned.

Non-canonical historical raw outputs were moved to:

- `artifacts/_archive_noncanonical_cloud_20260611-231556`

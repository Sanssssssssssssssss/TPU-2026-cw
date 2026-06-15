# Final Rollout320 Run Registry

All final runs use the same GSM8K data/evaluation split and a shared rollout budget of 6,728 generated rollouts. K=2 runs use 3,364 GRPO steps and K=8 runs use 841 GRPO steps. Checkpoint/eval points are aligned every 320 rollouts, with the final point at 6,728 rollouts.

## Run Definitions

| Line | Run id | Reward mode | K | Steps | LR | Beta | LoRA rank/alpha | Advantage estimator | Source checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| R0 | `baseline-rollout320-full-001` | `baseline` | 2 | 3364 | 3e-6 | 0.08 | 64/64 | `grpo` | none |
| R1 | `r1-format-rollout320-full-001` | `gsm8k_verifiable_format` | 2 | 3364 | 3e-6 | 0.08 | 64/64 | `grpo` | none |
| R2 | `r2-k8-beta004-rollout320-full-001` | `baseline` | 8 | 841 | 3e-6 | 0.04 | 64/64 | `grpo` | none |
| R3 | `r3-loo-advantage-rollout320-full-001` | `baseline` | 2 | 3364 | 3e-6 | 0.08 | 64/64 | `rloo` | none |
| R4 | `r4-r12-format-rollout320-lr3e6-001` | `gsm8k_verifiable_format` | 8 | 841 | 3e-6 | 0.04 | 64/64 | `grpo` | none |
| R5 | `r5-lora-r16-rollout320-full-001` | `baseline` | 2 | 3364 | 3e-6 | 0.08 | 16/16 | `grpo` | none |
| R6 | `r6-lora-r32-rollout320-full-002` | `baseline` | 2 | 3364 | 3e-6 | 0.08 | 32/32 | `grpo` | none |

## Final Metrics

| Line | Best exact | Best exact rollout | Best partial | Best partial rollout | Final exact | Final partial | Final format |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 48.4375 | 1920 | 51.5625 | 1920 | 3.1250 | 6.2500 | 12.5000 |
| R1 | 54.6875 | 640 | 59.3750 | 640 | 3.1250 | 3.1250 | 56.2500 |
| R2 | 59.3750 | 2560 | 67.1875 | 5440 | 56.2500 | 60.9375 | 87.5000 |
| R3 | 57.8125 | 320 | 60.9375 | 320 | 40.6250 | 45.3125 | 62.5000 |
| R4 | 60.9375 | 5440 | 64.0625 | 4160 | 56.2500 | 59.3750 | 92.1875 |
| R5 | 56.2500 | 5440 | 60.9375 | 5440 | 48.4375 | 51.5625 | 50.0000 |
| R6 | 56.2500 | 6720 | 60.9375 | 5760 | 56.2500 | 59.3750 | 60.9375 |

## Controlled-Comparison Notes

- R1 changes only the reward mechanism relative to R0.
- R2 changes K and beta together to keep the rollout budget aligned with the K=2 baseline.
- R3 changes only the advantage estimator to leave-one-out.
- R4 is the selected final full-from-zero format-aware K=8 run.
- R5 and R6 change only LoRA rank and alpha relative to R0.
- No final run is initialized from a source checkpoint or tail continuation.

## Artifact Locations

- Official comparison manifest: `artifacts/reports/final-comparison/manifest_rollout320_official_comparison.json`
- Checkpoint evaluation table: `artifacts/reports/final-comparison/tables/checkpoint_eval_rollout_aligned.csv`
- Scalar table: `artifacts/reports/final-comparison/tables/scalar_long_rollout_aligned.csv`
- Report figures and compact logs: `artifacts/reports/final-figures/`

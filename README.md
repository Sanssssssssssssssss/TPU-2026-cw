# TPU-2026 GRPO Coursework Repository

This repository contains the final rollout-aligned GRPO evidence package for the coursework submission. The submission-facing comparison uses exactly seven complete runs, labelled `R0` through `R6`, all aligned to 6,728 generated rollouts and 22 checkpoint/evaluation points.

## Final Runs

| Line | Run id | Controlled change relative to R0 |
|---|---|---|
| R0 | `baseline-rollout320-full-001` | Course baseline: K=2, beta=0.08, LoRA rank/alpha 64/64 |
| R1 | `r1-format-rollout320-full-001` | Reward changed to the format-aware verifiable reward |
| R2 | `r2-k8-beta004-rollout320-full-001` | K=8 and beta=0.04 |
| R3 | `r3-loo-advantage-rollout320-full-001` | Advantage estimator changed from group-mean GRPO to leave-one-out |
| R4 | `r4-r12-format-rollout320-lr3e6-001` | Format-aware K=8 run with beta=0.04 and lr=3e-6 |
| R5 | `r5-lora-r16-rollout320-full-001` | LoRA rank/alpha changed to 16/16 |
| R6 | `r6-lora-r32-rollout320-full-002` | LoRA rank/alpha changed to 32/32 |

The older simple-reward `R4` and historical R12/tail/autotune exploration runs are excluded from the final comparison package.

## Evidence Packages

- `artifacts/reports/final-comparison/`: final aligned comparison tables, figures, raw refs, and manifest.
- `artifacts/reports/final-figures/`: report-ready figures plus web-readable data for each final run.
- `artifacts/reports/final-figures/data/<R*>/`: compact TensorBoard-derived scalar CSVs, checkpoint eval JSON/CSV, trace summaries, run manifests, and run environments.
- `artifacts/reports/final-figures/data_manifest.json`: file-level index for included compact logs and omitted large local raw files.

Raw checkpoint archives and full TensorBoard event files remain under local `artifacts/cloud/` and are intentionally not committed to ordinary Git. The report package includes TensorBoard-equivalent scalar CSVs and SHA256-indexed references for omitted large raw sources.

## Rebuild And Verify

Run these commands from the repository root:

```powershell
py -3 scripts\verify_experiment_package.py artifacts\cloud\baseline-rollout320-full-001
py -3 scripts\verify_experiment_package.py artifacts\cloud\r1-format-rollout320-full-001
py -3 scripts\verify_experiment_package.py artifacts\cloud\r2-k8-beta004-rollout320-full-001
py -3 scripts\verify_experiment_package.py artifacts\cloud\r3-loo-advantage-rollout320-full-001
py -3 scripts\verify_experiment_package.py artifacts\cloud\r4-r12-format-rollout320-lr3e6-001
py -3 scripts\verify_experiment_package.py artifacts\cloud\r5-lora-r16-rollout320-full-001
py -3 scripts\verify_experiment_package.py artifacts\cloud\r6-lora-r32-rollout320-full-002
py -3 scripts\build_rollout320_official_comparison_package.py
py -3 scripts\build_rollout320_report_figures.py
```

Expected invariants:

- Every final run has 22 checkpoint/eval rows.
- K=2 runs finish at step 3,364; K=8 runs finish at step 841.
- `rollouts_seen = step * num_generations` aligns all runs to 6,728 rollouts.
- Final comparison manifests contain only `R0`, `R1`, `R2`, `R3`, `R4`, `R5`, and `R6`.
- Report-facing reward plots use a shared baseline 0-10 scale; native reward values are preserved in the scalar data.

## Documentation

- `docs/final_run_registry.md`: exact final run definitions and key results.
- `docs/baseline_patches.md`: baseline patch log for the report.
- `docs/reward_mechanisms.md`: reward definitions and the reward scaling policy used only for visual comparison.

## Submission Notes

The final PDF report should cite the exact Git commit used for submission and link the GitLab mirror of this repository. The local `archive_local/` folder is ignored and is not part of the submitted repository.

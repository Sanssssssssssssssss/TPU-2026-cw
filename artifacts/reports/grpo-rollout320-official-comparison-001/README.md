# GRPO Rollout320 Official Comparison

This package compares fixed official R0/R1/R2/R3 lines and the R4 slot. R1 now uses the current format-aware reward, while the earlier simple-reward R1 is superseded. R3 changes only the advantage estimator from Tunix GRPO to RLOO. R4 currently has the lr1e-6 reference plus format-aware full-from-zero alternatives until final selection. The original three-line evidence package is not overwritten. All comparison figures use `rollouts_seen`, not raw step.

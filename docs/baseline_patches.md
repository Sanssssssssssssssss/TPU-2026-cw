# Baseline Patch Log

This file records implementation and packaging patches that affect reproducibility reporting. It is intended to support the coursework report's baseline patch section.

## Training Baseline

- The final R0 baseline is `baseline-rollout320-full-001`.
- It uses the course baseline reward, K=2, beta=0.08, LoRA rank/alpha 64/64, and the standard group-mean GRPO advantage estimator.
- It is trained from the base model with no source checkpoint.
- It is rollout-aligned to the later comparison runs by saving/evaluating every 160 K=2 steps, equivalent to every 320 generated rollouts.

## Packaging And Reporting Patches

- Report builders use `rollouts_seen = step * num_generations` as the comparison axis.
- TensorBoard raw event parsing falls back to already materialized scalar CSVs when the local `tensorboard` Python package is unavailable.
- Report-facing reward curves are normalized to a baseline 0-10 scale only in generated tables/figures; native reward values remain in `*_native` columns.
- Large raw checkpoint and TensorBoard event files are not committed to ordinary Git. Their compact scalar/eval equivalents and indexed local references are included in the report package.

## Non-Baseline Exclusions

Historical R12 tail continuation, old simple-reward R4, reward-only completion, and autotune exploration runs are excluded from the final comparison. They were moved to `archive_local/` and are not submission artifacts.

# Reward Mechanisms And Plot Scaling

The final comparison includes multiple reward mechanisms. Native reward magnitudes are not directly comparable, so the report package keeps native values and also provides a derived baseline 0-10 scale for visual comparison.

## Baseline Reward

Used by R0, R2, R3, R5, and R6.

Native maximum: 10.0

Components:

- Strict template format: 3.0
- Approximate tag format: 2.5
- Answer correctness: 3.0
- Numeric fallback correctness: 1.5

## Format-Aware Verifiable Reward

Used by R1 and the final R4.

Native maximum: 1.8

Components:

- GSM8K numeric correctness: 1.0
- Answer-tag helper: 0.2
- Reasoning/answer envelope structure: 0.6

This reward is designed to keep the verifiable numeric signal while explicitly rewarding the strict `<reasoning>...</reasoning><answer>...</answer>` response structure.

## Report Scaling Policy

The report-facing reward scale is:

```text
reward_score_report = reward_score_native / reward_native_max * 10.0
```

This scaling is applied only to copied report tables and figures. It does not modify TensorBoard event files, raw artifacts, checkpoint evals, or the original training traces.

Native values remain available in:

- `value_native` in `artifacts/reports/grpo-rollout320-official-comparison-001/tables/scalar_long_rollout_aligned.csv`
- `train_reward_score_native` and `eval_reward_score_native` in each run's `tensorboard_derived/scalar_pivot.csv`
- `tensorboard_derived/scalar_pivot_raw.csv` for the unmodified scalar pivot

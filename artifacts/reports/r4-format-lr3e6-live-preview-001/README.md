# R4 format-aware live preview

Run: `r4-r12-format-rollout320-lr3e6-001` / `R4_r12_format_lr3e-6_rollout320`

These are lightweight live trace plots generated without downloading checkpoint archives.
The x-axis is an approximate live-trace alignment to the latest fetched checkpoint: step 841 x K=8 = 6728 official rollouts.
The raw observability `call_index` is not an official rollout count. The numeric-exact rate is a rollout-trace proxy; it is not the held-out checkpoint eval accuracy.

## Latest smoothed validation snapshot

- `reward_total`: early 0.7296 -> latest 1.3721
- `numeric`: early 0.4704 -> latest 0.6093
- `reasoning_format`: early 0.0992 -> latest 0.5704
- `format_ok`: early 0.0612 -> latest 0.9235
- `numeric_exact`: early 0.3852 -> latest 0.5536
- `extracted_none`: early 0.1760 -> latest 0.0714

## Figures

- `01_r4_live_reward_format_accuracy.png`
- `02_r4_live_reward_components.png`
- `03_r4_live_format_accuracy_proxy_rates.png`
- `04_r4_checkpoint_eval_accuracy.png`

# R12 Tail512 Winner Visual Summary

Winner retained in this package:

- Run: `r12-full-autotune-tail512-001`
- Branch: `R12_tail_lr1e-6_beta004_from512`
- Source: canonical `reward-k8-beta004-r12-full-001` checkpoint `512`
- Config: `REWARD_MODE=gsm8k_verifiable_simple`, `K=8`, `BETA=0.04`, `LEARNING_RATE=1e-6`, `RANK=64`, `ALPHA=64`
- Best checkpoint: step `841`, exact `65.625`, partial `67.1875`
- Canonical R12 full best: exact `62.5`, partial `65.625`

## Combined Comparison

These figures compare the retained winner branch against the non-winning
`beta=0.06, lr=1e-6` branch from the same tail512 run.

![Checkpoint eval](figures/combined/01_checkpoint_eval.png)

![Reward score](figures/combined/02_reward_score.png)

![KL loss clipfrac](figures/combined/03_kl_loss_clipfrac.png)

![GRPO health](figures/combined/04_grpo_health.png)

![Response health](figures/combined/05_response_health.png)

![Reward audit](figures/combined/06_reward_audit.png)

![Reward components](figures/combined/07_reward_components.png)

![Reward composition format share](figures/combined/08_reward_composition_format_share.png)

## Winner Raw Timelines

These winner-only figures preserve the raw curve view, not just the combined
rolling-mean comparison.

![Winner checkpoint eval](figures/winner_raw/01_checkpoint_eval.png)

![Winner reward score raw](figures/winner_raw/02_reward_score_raw.png)

![Winner KL loss clipfrac raw](figures/winner_raw/03_kl_loss_clipfrac_raw.png)

![Winner GRPO health raw](figures/winner_raw/04_grpo_health_raw.png)

![Winner response health raw](figures/winner_raw/05_response_health_raw.png)

![Winner reward audit raw](figures/winner_raw/06_reward_audit_raw.png)

![Winner reward components raw](figures/winner_raw/07_reward_components_raw.png)

![Winner trace audit raw](figures/winner_raw/08_trace_audit_raw.png)

![Winner reward composition format share raw](figures/winner_raw/09_reward_composition_format_share_raw.png)

## Reading Notes

- Checkpoint eval is the decisive performance read: the winner improves from
  the source step 512 to step 841 and beats the canonical R12 full best.
- Reward composition latest for the winner is mostly numeric reward
  (`0.775` mean numeric component, `0.2` mean format component).
- The retained raw run remains at
  `artifacts/cloud/r12-full-autotune-tail512-001`; this visual package only
  copies report-ready figures and small provenance tables.

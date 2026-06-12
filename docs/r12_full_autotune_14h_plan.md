# R12 Full 14h Autotune Plan

This is a post-reward-only follow-up plan. It must not overwrite the canonical
R12 full run:

- Canonical R12 full: `reward-k8-beta004-r12-full-001`
- Canonical report: `artifacts/reports/reward-k8-beta004-r12-full-001-clean`

All tuning runs must use new run ids under `r12-full-autotune-*`.

## External guidance used

- Hugging Face TRL documents GRPO as generating `G` completions per prompt and
  lists `num_generations=8` as the default GRPO setting.
- The TRL GRPO docs also note that recent practice often uses `beta=0.0` or a
  very small KL term; nonzero `beta` remains available when KL regularization is
  desired.
- TRL now exposes length-bias-aware losses such as DAPO / Dr.GRPO, but this
  repository's Tunix path currently exposes the practical knobs `BETA`,
  `EPSILON`, `LEARNING_RATE`, `RANK`, `ALPHA`, and checkpoint selection. Do not
  change loss formulation unless it is implemented and tested separately.

References:

- https://huggingface.co/docs/trl/en/grpo_trainer
- https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py
- https://arxiv.org/abs/2402.03300

## Queue trigger

Do not start this queue until `reward-only-r12-full-complete-001` has finished,
been fetched locally, and its clean plots / three-line evidence package have
been rebuilt.

## Budget and rule

- Wall-clock budget: about 14 hours after reward-only completion.
- Only one TPU training tmux should run at a time.
- Check each candidate once checkpoint `64` exists.
- If checkpoint `64` already shows severe health problems, stop that candidate
  and switch to the next one.
- Severe health gate:
  - repeated `zero_reward_std_spike` near 1.0, and
  - repeated `extracted_none_spike` at or above 0.5, and
  - no obvious improving signal in recent logs.
- If a candidate survives the 64-step gate, let it continue to its configured
  pilot horizon, fetch it, and compare checkpoint evals against the current R12
  full baseline.

## Candidate queue

1. Near-zero KL, lower LR:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\cloud\submit_tpu_job.ps1 submit-k8-r14-public-beta-only -RunId r12-full-autotune-beta000-lr1e6-001
   ```

   Expected config: `K=8`, `BETA=0.0`, `LEARNING_RATE=1e-6`, max 256-step
   pilot. This directly tests the TRL-style "KL often unnecessary" direction.

2. Tiny KL, lower LR:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\cloud\submit_tpu_job.ps1 submit-k8-r13-public-beta-only -RunId r12-full-autotune-beta0001-lr1e6-001
   ```

   Expected config: `K=8`, `BETA=0.001`, `LEARNING_RATE=1e-6`, max 256-step
   pilot.

3. Higher-rank, moderated LR:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\cloud\submit_tpu_job.ps1 submit-r12-high-rank-alpha64-only -RunId r12-full-autotune-r128-a64-lr2e6-001
   ```

   Expected config: `K=8`, `BETA=0.04`, `LEARNING_RATE=2e-6`,
   `RANK=128`, `ALPHA=64`, max 256-step pilot.

4. Tail stability from current best R12 checkpoint 512, if time remains:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\cloud\submit_tpu_job.ps1 submit-r12-tail-stability -RunId r12-full-autotune-tail512-001
   ```

   Expected config: seed from
   `reward-k8-beta004-r12-full-001/runs/R12_gsm8k_verifiable_simple/ckpts/actor/512`,
   run `LR=1e-6` continuations with `BETA=0.04` and `BETA=0.06`.

## Evidence requirements

For any candidate that reaches its pilot horizon:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\cloud\submit_tpu_job.ps1 fetch-k8-pilot -RunId <RUN_ID>
py -3 scripts\build_reward_sweep_clean_plots.py --input-dir artifacts\cloud\<RUN_ID> --output-dir artifacts\reports\<RUN_ID>-clean --rolling-window 64
```

Keep the canonical three-line package unchanged unless a candidate beats the
existing R12 full line on checkpoint eval or a later large eval. If a candidate
does beat it, create a separate `r12-full-autotune-evidence-*` package first;
do not silently replace `reward-k8-beta004-r12-full-001`.

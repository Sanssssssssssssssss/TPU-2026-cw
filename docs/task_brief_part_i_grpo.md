# Part I GRPO Task Brief And Run Ledger

This file preserves the original working brief and the current interpretation
against the exam PDF, so future work can be compared against the same target.
Secrets are intentionally omitted.

## Source References

- Exam PDF: `C:/Users/X/Downloads/exam_2026_v2_questions.pdf`
- Extracted Part I text: `artifacts/pdf_part_i_excerpt.txt`
- Baseline repository: `https://github.com/borisbolliet/tpu-2026`
- Local workspace: `D:/GPT_Project/TPU-2026-cw`

## Original User Instruction Snapshot

The user first asked to plan the front practical part of Part I, focused on
GRPO, by reading the exam PDF and the `borisbolliet/tpu-2026` repository.

The initial implementation plan requested:

- Cover Part I front practical work: I.1 baseline reproduction and I.2 baseline
  understanding.
- Defer I.3 improvement experiments, while leaving an extension path.
- Patch `scripts/evaluate.py` with `--ckpt-dir`, `--step 0/latest`,
  `--no-restore`, and `--output-json`, reusing Orbax/LoRA restore logic.
- Patch `scripts/run_tmux.sh` to remove hard-coded user paths and support a
  configurable virtualenv.
- Add `scripts/export_baseline_artifacts.py` to export scalar tables and
  report-ready reward/KL plots.
- Use `greedy`, `tfds`, and `NUM_TEST_BATCHES=64` by default unless TPU/course
  constraints require otherwise.

The cloud setup plan requested:

- Use the user's own Google Cloud account/project, not Boris's project.
- Local Windows should orchestrate TPU VM creation, code upload, tmux
  submission, status checks, result fetches, and TPU stop/delete.
- Training, evaluation, TensorBoard, checkpoints, and artifacts run on the TPU
  VM.
- Use TPU VM + IAP SSH/SCP orchestration.
- Store secrets only in ignored local/remote files such as `.env`; never commit
  or log token values.
- Add Cloud Storage sync so artifacts, checkpoints, TensorBoard logs, and model
  cache survive TPU stop/delete.

## Exam PDF I.1 Requirements

From the exam PDF, section I.1 "Reproducing the baseline" requires:

- Run `scripts/train.py` end-to-end on `v6e-1` with default
  `scripts/config.py`.
- Run `scripts/evaluate.py` on the resulting checkpoint.
- Report:
  - exact commit,
  - wall-clock time,
  - number of GRPO steps actually completed,
  - GSM8K accuracy for base `google/gemma-3-1b-it`,
  - GSM8K accuracy for the LoRA-finetuned checkpoint,
  - held-out split and fixed seed,
  - baseline mean reward vs GRPO step,
  - KL divergence over training.
- If the baseline did not run as shipped, document fixes in one "baseline
  patches" section.

## Current Run Status

Run ID: `baseline-full-001`

- TPU project: `grpo-tpu-play-20260603-z9s5`
- TPU VM: `grpo-play-v6e`
- Zone: `us-east5-b`
- Accelerator/runtime: `v6e-1`, `v2-alpha-tpuv6e`
- TPU final state after manual stop: `STOPPED`, health `HEALTHY`
- Local outputs: `artifacts/cloud/baseline-full-001/`
- GCS run prefix:
  `gs://grpo-tpu-play-20260603-z9s5-tpu-artifacts/tpu-runs/baseline-full-001`
- GCS HF cache prefix:
  `gs://grpo-tpu-play-20260603-z9s5-tpu-artifacts/cache/huggingface/hub`

Base eval result over `NUM_TEST_BATCHES=64`:

- accuracy: `51.5625`
- partial accuracy: `53.125`
- format accuracy: `6.25`
- correct/total: `33/64`

Training was manually stopped on `Thu Jun 4 00:44:02 UTC 2026` after late-stage
collapse:

- latest train score mean: `-2.5` at step `2943`
- latest eval score mean: `-1.034425139427185` at step `2880`
- peak eval score mean: `6.845253944396973` at step `448`
- first post-peak eval score <= 2: step `1152`
- first post-peak eval score <= 0: step `1856`
- synced checkpoints: `1`, `500`, `1000`, `1500`, `2000`, `2500`

Diagnostics:

- `artifacts/cloud/baseline-full-001/analysis/training_diagnostics.json`
- `artifacts/cloud/baseline-full-001/analysis/training_diagnostics.png`
- `artifacts/cloud/baseline-full-001/analysis/training_diagnostics.pdf`

## Compliance Judgment For I.1

Current status is a strong baseline attempt and useful diagnostic run, but it is
not yet a complete I.1 submission because:

- the run did not finish end-to-end;
- there is no `baseline_lora_eval.json` for a restored LoRA checkpoint;
- the report still lacks an evaluated checkpoint accuracy table for the LoRA
  model on the same held-out split/seed as the base model;
- wall-clock time and exact completed GRPO steps can be reported, but must be
  described as a manually early-stopped run rather than a completed default
  baseline.

Minimum recovery for I.1:

- Restart TPU or use the synced checkpoint data.
- Restore and evaluate at least checkpoint `500`, and ideally `1000`, `1500`,
  `2000`, and `2500`, using the same held-out split/seed and `greedy` preset.
- Select and report the best checkpoint by held-out accuracy, while honestly
  documenting that the later training collapsed and the run was manually
  stopped.
- Keep the reward/KL curves and diagnostics as evidence and explanation.

## I.2 Anchors To Preserve

- `pi_theta`: current actor policy with trainable LoRA adapter parameters.
- `pi_ref`: frozen base model/reference policy.
- `pi_old`: rollout/update old policy snapshot or stored old log-probabilities.
- Baseline group size: `scripts/config.py::NUM_GENERATIONS = 2`.
- Tunix advantage estimator: default `advantage_estimator="grpo"` in Tunix,
  using group mean/std normalization, not leave-one-out.
- Reward shaping terms live in `scripts/rewards.py`; numeric correctness is the
  task-success signal, while format and approximate terms shape exploration.
- PPO-style clipped surrogate lives in Tunix GRPO loss; `EPSILON = 0.2`
  corresponds to ratio clip `[0.8, 1.2]`.

## Guardrails

- Do not include secrets in committed files, logs, reports, or prompts.
- Do not claim I.1 is complete until a LoRA checkpoint has been evaluated.
- Do not present the latest checkpoint as the best checkpoint without
  checkpoint-wise evaluation; late-stage metrics show collapse.
- Use GitLab as the final repository link in the report, per the PDF.

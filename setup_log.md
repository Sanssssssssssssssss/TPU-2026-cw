# Google Cloud TPU Playground Setup Log

Started: 2026-06-03

## Planned playground resources

- PROJECT_ID: `grpo-tpu-play-20260603-z9s5`
- TPU_NAME: `grpo-play-v6e`
- ZONE: `us-east5-b`
- REGION: `us-east5`
- ACCELERATOR_TYPE: `v6e-1`
- Preferred VERSION: `v2-alpha-tpuv6e`

## Progress

- Confirmed `gcloud` was not on PATH.
- Confirmed `winget` is available.
- Installed Google Cloud SDK `571.0.0` with `winget install --id Google.CloudSDK --exact --source winget --accept-package-agreements --accept-source-agreements`.
- Logged in with `gcloud auth login` as `sansxu1009@gmail.com`.
- `gcloud auth application-default login` was attempted twice, but Google did not consent the required `cloud-platform` scope. CLI auth is sufficient for the setup steps completed here.
- Created project `grpo-tpu-play-20260603-z9s5` named `GRPO TPU Playground`.
- Set active gcloud project to `grpo-tpu-play-20260603-z9s5`.
- Linked billing account `01B8C9-C8FE04-A1AA35`; billing is enabled.
- Enabled APIs: `compute.googleapis.com`, `tpu.googleapis.com`, `iap.googleapis.com`.
- Enabled Private Google Access on the default subnet in `us-east5`.
- Created Cloud Router `nat-router` and NAT `nat-config` in `us-east5`.
- Created firewall rule `allow-iap-ssh` for source range `35.235.240.0/20` allowing `tcp:22`.
- Confirmed `v6e-1` is available in `us-east5-b`.
- Confirmed runtime version `v2-alpha-tpuv6e` is available in `us-east5-b`.
- Created TPU VM `grpo-play-v6e`.
- TPU status: `READY`, health: `HEALTHY`, accelerator type: `v6e-1`, internal IP: `10.202.0.2`.
- Installed gcloud `alpha` component to access `gcloud alpha compute tpus tpu-vm ssh --tunnel-through-iap`.
- Direct `gcloud alpha compute tpus tpu-vm ssh ... --tunnel-through-iap` used bundled PuTTY/plink on Windows and timed out.
- Verified IAP tunnel works by starting a local tunnel to the TPU VM worker instance `t1v-n-9ce97077-w-0`.
- Verified SSH connectivity through the local IAP tunnel with Windows OpenSSH:
  - Remote output included `CONNECTED`.
  - Hostname: `t1v-n-9ce97077-w-0`.
  - Kernel: `Linux t1v-n-9ce97077-w-0 6.8.0-1015-gcp ... x86_64 GNU/Linux`.
  - Python: `Python 3.10.12`.
  - Remote user: `X`.
- Wrote local ignored config `cloud/tpu_config.local.ps1` pointing to this playground project and TPU.
- Submitted tiny smoke run `smoke-003` through IAP + OpenSSH.
  - Remote submit wrapper tmux session: `submit-smoke-003`.
  - Remote pipeline tmux session: `tpu-smoke-003`.
  - Remote run directory: `/home/X/tpu-runs/smoke-003`.
  - Repository bootstrap completed successfully.
  - TPU backend check passed: `jax.default_backend()` returned `tpu`.
  - Pipeline reached base model evaluation, then stopped on Hugging Face gated repo access for `google/gemma-3-1b-it` with HTTP 403.
  - Hugging Face auth was verified on the TPU VM as user `Sanssss133`; the remaining action is model access approval for that account/token.
  - Earlier smoke attempts found and fixed two local-to-Linux packaging issues: portable ZIP entry separators and CRLF shell script line endings.
  - A local git-ignored `.env` and run-local remote `.env` files were created for Hugging Face auth; secret values are intentionally not logged.
- Completed tiny smoke run `smoke-008`.
  - Confirmed base eval, tiny GRPO training, LoRA checkpoint restore/eval, TensorBoard scalar export, and PNG/PDF curve export all work.
  - Local fetched outputs are under `artifacts/cloud/smoke-008/`.
  - Actual TensorBoard tags used for report curves are `rewards/train/score/mean` and `actor/train/kl`.
- Submitted full baseline run `baseline-full-001`.
  - Remote tmux session: `tpu-baseline-full-001`.
  - Remote run directory: `/home/X/tpu-runs/baseline-full-001`.
  - Status observed after submission: base evaluation running on the TPU VM.
- Full baseline status update on 2026-06-03:
  - `tmux` is still running.
  - Base evaluation completed and wrote `artifacts/base_eval.json`.
  - Base held-out metrics over `NUM_TEST_BATCHES=64`: accuracy `51.5625`, partial accuracy `53.125`, format accuracy `6.25`.
  - Training has started; TensorBoard showed `actor/train/*` through step `125`, and `actor/eval/*` through step `64`.
  - First actor checkpoint exists at `ckpts/actor/1`.
  - Expected full training target is `MAX_STEPS=3364`; observed early throughput suggests roughly 5-6 more hours, allowing for eval/checkpoint overhead.
- Added Cloud Storage backup/cache support.
  - Bucket: `gs://grpo-tpu-play-20260603-z9s5-tpu-artifacts`.
  - Bucket location/class: `us-east5`, `STANDARD`.
  - Current run backup prefix: `gs://grpo-tpu-play-20260603-z9s5-tpu-artifacts/tpu-runs/baseline-full-001`.
  - Hugging Face model cache prefix: `gs://grpo-tpu-play-20260603-z9s5-tpu-artifacts/cache/huggingface/hub`.
  - Synced report artifacts, TensorBoard, checkpoints, logs, and non-token `google/gemma-3-1b-it` cache.
  - Current synced storage usage: run output about `374 MB`, Hugging Face cache about `4.08 GB`.
  - Later status showed training through `actor/train` step `593`, eval step `576`, and checkpoint `ckpts/actor/500`.
- Manually stopped `baseline-full-001` on 2026-06-04 because reward and response quality collapsed in the later GRPO steps.
  - Stop marker: `Thu Jun  4 00:44:02 UTC 2026`.
  - Latest curve data before stop: train score mean `-2.5` at step `2943`; eval score mean `-1.034425139427185` at step `2880`.
  - Peak eval score mean was `6.845253944396973` at step `448`.
  - First post-peak eval score <= 2 occurred at step `1152`; first post-peak eval score <= 0 occurred at step `1856`.
  - Checkpoints synced to GCS: `1`, `500`, `1000`, `1500`, `2000`, `2500`.
  - Final synced run usage: about `1.36 GB`; Hugging Face cache usage: about `4.08 GB`.
  - Local fetched outputs: `artifacts/cloud/baseline-full-001/`.
  - Diagnostics: `artifacts/cloud/baseline-full-001/analysis/training_diagnostics.json` and `training_diagnostics.png/pdf`.
  - TPU VM `grpo-play-v6e` was stopped successfully; final state confirmed as `STOPPED`, health `HEALTHY`.

## Useful commands

Open a local IAP tunnel, then SSH with Windows OpenSSH:

```powershell
gcloud alpha compute start-iap-tunnel t1v-n-9ce97077-w-0 22 `
  --local-host-port=127.0.0.1:2225 `
  --project=grpo-tpu-play-20260603-z9s5 `
  --zone=us-east5-b

ssh.exe -4 -p 2225 `
  -i "$env:USERPROFILE\.ssh\google_compute_engine" `
  -o StrictHostKeyChecking=no `
  -o UserKnownHostsFile="$env:USERPROFILE\.ssh\google_compute_known_hosts" `
  X@127.0.0.1
```

Check TPU status:

```powershell
gcloud compute tpus tpu-vm list `
  --project=grpo-tpu-play-20260603-z9s5 `
  --zone=us-east5-b
```

Delete the TPU VM when finished, but this has not been run:

```powershell
gcloud compute tpus tpu-vm delete grpo-play-v6e `
  --project=grpo-tpu-play-20260603-z9s5 `
  --zone=us-east5-b
```

Delete the playground project when finished, but this has not been run:

```powershell
gcloud projects delete grpo-tpu-play-20260603-z9s5
```

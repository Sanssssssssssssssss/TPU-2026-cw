# Cloud TPU orchestration

This folder lets a local Windows machine submit the GRPO baseline workflow to a
Google TPU VM. Local code is bundled and uploaded; training and evaluation run
remotely in `tmux`; artifacts are fetched back under `artifacts/cloud/<RUN_ID>/`.

## 1. Local prerequisites

Install Google Cloud CLI, then authenticate:

```powershell
gcloud auth login
gcloud config set project tpu-2026
```

Create a local config only if you need to override defaults:

```powershell
Copy-Item cloud\tpu_config.example.ps1 cloud\tpu_config.local.ps1
```

Put secrets in a local `.env` file at the repo root. This file is ignored by
git and is uploaded only to the TPU VM run directory:

```text
WANDB_API_KEY=...
WANDB_PROJECT=grpo-tpu-2026
WANDB_ENTITY=...   # optional; omit to use the authenticated default entity
HF_TOKEN=...
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

## 2. First-time TPU setup

```powershell
.\cloud\submit_tpu_job.ps1 preflight
.\cloud\submit_tpu_job.ps1 ensure-tpu
.\cloud\submit_tpu_job.ps1 ensure-storage
.\cloud\submit_tpu_job.ps1 bootstrap -RunId setup-001
```

Use `-DryRun` on any command to print the `gcloud` calls without touching cloud
resources.

## 3. Submit baseline

For a tiny smoke test:

```powershell
.\cloud\submit_tpu_job.ps1 submit-baseline -RunId smoke-001 -TinySmoke
```

For the real baseline run:

```powershell
.\cloud\submit_tpu_job.ps1 submit-baseline -RunId baseline-001
```

The remote pipeline writes observability outputs under
`~/tpu-runs/<RUN_ID>/artifacts`: `run_manifest.json`, `rollout_traces/*.jsonl`,
W&B/TensorBoard metrics, checkpoint eval summaries, and report-ready plots.

To evaluate existing checkpoints without re-training, start the TPU and submit
only the checkpoint eval pipeline:

```powershell
.\cloud\submit_tpu_job.ps1 start-tpu
.\cloud\submit_tpu_job.ps1 eval-checkpoints -RunId baseline-full-001
```

## 4. Monitor and fetch

```powershell
.\cloud\submit_tpu_job.ps1 status -RunId baseline-001
.\cloud\submit_tpu_job.ps1 fetch -RunId baseline-001
```

Fetched outputs include evaluation JSON, baseline plots, logs, and git metadata.
Checkpoints remain on the TPU VM under `~/tpu-runs/<RUN_ID>/ckpts` until you
sync them to Cloud Storage or delete the VM.

## 5. Back up to Cloud Storage

Use the project-local bucket for durable run outputs and model cache:

```powershell
.\cloud\submit_tpu_job.ps1 ensure-storage
.\cloud\submit_tpu_job.ps1 sync-storage -RunId baseline-001
```

This syncs report artifacts, TensorBoard events, checkpoints, logs, and the
Hugging Face `google/gemma-3-1b-it` model cache. It intentionally does not sync
`.env`, Hugging Face token files, or the full source tree.

If you recreate the TPU VM later, restore the cached model before submitting:

```powershell
.\cloud\submit_tpu_job.ps1 restore-cache
```

## 6. Stop or delete TPU when done

Stop the TPU to stop TPU charges while keeping the VM configuration/software:

```powershell
.\cloud\submit_tpu_job.ps1 stop-tpu
```

Start it again later:

```powershell
.\cloud\submit_tpu_job.ps1 start-tpu
```

Delete the TPU VM when you no longer need it:

```powershell
.\cloud\submit_tpu_job.ps1 delete-tpu
```

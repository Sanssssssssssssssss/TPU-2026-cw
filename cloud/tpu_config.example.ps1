# Copy this file to cloud/tpu_config.local.ps1 and edit values if needed.
# cloud/tpu_config.local.ps1 is ignored by git.

$ProjectId = "grpo-tpu-play-20260603-z9s5"
$TpuName = "grpo-play-v6e"
$Zone = "us-east5-b"
$Region = "us-east5"
$AcceleratorType = "v6e-1"
$RuntimeVersion = "v2-alpha-tpuv6e"

# TPU VM access. Internal IP TPUs require IAP tunnelling from local machines.
$UseIapTunnel = $true

# Set this to $true on Windows if `gcloud tpu-vm ssh/scp` hangs in PuTTY/plink.
# IapTargetName is the TPU worker hostname shown by a successful SSH smoke test,
# for example `t1v-...-w-0`.
$UseOpenSshIap = $false
$IapTargetName = ""
$SshUser = $env:USERNAME
$SshKeyPath = "$env:USERPROFILE\.ssh\google_compute_engine"
$SshKnownHostsPath = "$env:USERPROFILE\.ssh\google_compute_known_hosts"

# Remote paths on the TPU VM.
$RemoteRoot = "~/tpu-runs"
$RemoteToolsDir = "~/tpu-runs/_tools"
$RemoteIncomingDir = "~/tpu-runs/_incoming"
$RemoteVenv = "~/venvs/tunix"

# Local secrets file to upload for training/evaluation. This file should contain
# WANDB_API_KEY, HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY, and optional WANDB_* vars.
$SecretsFile = ".env"

# Local destination for fetched run outputs.
$LocalArtifactsRoot = "artifacts/cloud"

# Cloud Storage backup/cache. Set UseStorage to false when working on a shared
# course project where you should not create or depend on buckets. With
# UseStorage true, leaving StorageBucket empty defaults to
# <project-id>-tpu-artifacts.
$UseStorage = $true
$StorageBucket = ""
$StorageLocation = $Region
$StorageClass = "STANDARD"
$StoragePrefix = "tpu-runs"
$StorageCachePrefix = "cache"

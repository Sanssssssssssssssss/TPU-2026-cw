# Cloud TPU Orchestration

This folder contains the Windows and TPU-VM helpers used to submit, monitor,
stop, and fetch GRPO runs. The final coursework evidence is already fetched and
packaged under `artifacts/reports/`; these commands are here for reproducibility
and audit.

## Local Setup

```powershell
gcloud auth login
gcloud config set project tpu-2026
Copy-Item cloud\tpu_config.example.ps1 cloud\tpu_config.local.ps1
```

Put private tokens in a local root `.env` file. That file is ignored by Git.

## Official Final Runs

The final submission uses only these lines:

| Line | Submit action | Run id |
|---|---|---|
| R0 | `submit-baseline-rollout320-full` | `baseline-rollout320-full-001` |
| R1 | `submit-r1-format-rollout320-full` | `r1-format-rollout320-full-001` |
| R2 | `submit-r2-k8-beta004-rollout320-full` | `r2-k8-beta004-rollout320-full-001` |
| R3 | `submit-r3-loo-advantage-rollout320-full` | `r3-loo-advantage-rollout320-full-001` |
| R4 | `submit-r4-rollout320-lr3e6-format-full` | `r4-r12-format-rollout320-lr3e6-001` |
| R5 | `submit-r5-lora-r16-rollout320-full` | `r5-lora-r16-rollout320-full-001` |
| R6 | `submit-r6-lora-r32-rollout320-full` | `r6-lora-r32-rollout320-full-002` |

Submit one run at a time:

```powershell
.\cloud\submit_tpu_job.ps1 <submit-action> -RunId <run-id>
```

Monitor and fetch:

```powershell
.\cloud\submit_tpu_job.ps1 status-k8-pilot -RunId <run-id>
$env:SKIP_CHECKPOINT_EXTRACT='1'
.\cloud\submit_tpu_job.ps1 fetch-k8-pilot -RunId <run-id>
```

## Final Packaging

After fetching, verify and rebuild:

```powershell
py -3 scripts\verify_experiment_package.py artifacts\cloud\<run-id>
py -3 scripts\build_rollout320_official_comparison_package.py
py -3 scripts\build_rollout320_report_figures.py
```

Final submission-facing outputs:

- `artifacts/reports/final-comparison`
- `artifacts/reports/final-figures`
- `artifacts/reports/final-verification.json`

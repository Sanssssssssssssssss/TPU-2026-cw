#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/home/ext_sansxu1009_gmail_com/tpu-runs/course-baseline-001"
SRC_DIR="/home/ext_sansxu1009_gmail_com/tpu-runs/course-baseline-001/src"
VENV="/home/ext_sansxu1009_gmail_com/venvs/tunix"
ARTIFACT_DIR="/home/ext_sansxu1009_gmail_com/tpu-runs/course-baseline-001/artifacts"
REMOTE_ROOT="/home/ext_sansxu1009_gmail_com/tpu-runs"
PROJECT_ID="tpu-2026"
STORAGE_BUCKET=""
STORAGE_PREFIX="tpu-runs"
STORAGE_CACHE_PREFIX="cache"

cd "$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "$VENV/bin/activate"

export RUN_ID="course-baseline-001"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CKPT_DIR="$RUN_DIR/ckpts"
export INTERMEDIATE_CKPT_DIR="$RUN_DIR/intermediate_ckpt"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
export TRAIN_DATA_DIR="$RUN_DIR/data/train"
export TEST_DATA_DIR="$RUN_DIR/data/test"
export WANDB_RUN_ID="${WANDB_RUN_ID:-course-baseline-001}"
export WANDB_PROJECT="${WANDB_PROJECT:-grpo-tpu-2026}"
export OBS_OUTPUT_DIR="$ARTIFACT_DIR/observability"
export OBS_TRACE_DIR="$ARTIFACT_DIR/rollout_traces"
export OBS_RUN_MANIFEST="$ARTIFACT_DIR/run_manifest.json"
export OBS_TRACE_EVERY_N_STEPS="${OBS_TRACE_EVERY_N_STEPS:-64}"
export OBS_TRACE_MAX_ROWS="${OBS_TRACE_MAX_ROWS:-32}"
if [[ -f "$RUN_DIR/meta/git_commit.txt" ]]; then
  export GIT_COMMIT="$(cat "$RUN_DIR/meta/git_commit.txt")"
fi
if [[ -f "$RUN_DIR/meta/git_status.txt" ]]; then
  export GIT_STATUS_SHORT="$(cat "$RUN_DIR/meta/git_status.txt")"
fi
mkdir -p "$ARTIFACT_DIR" "$CKPT_DIR" "$INTERMEDIATE_CKPT_DIR" "$TENSORBOARD_DIR" "$OBS_OUTPUT_DIR" "$OBS_TRACE_DIR"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "==> W&B enabled: project=$WANDB_PROJECT entity=${WANDB_ENTITY:-<default>}"
else
  if [[ -n "$STORAGE_BUCKET" ]]; then
    echo "==> W&B disabled: WANDB_API_KEY is not set; using TensorBoard + GCS."
  else
    echo "==> W&B disabled: WANDB_API_KEY is not set; using TensorBoard + local artifacts."
  fi
fi

sync_on_exit() {
  local status=$?
  if [[ -n "$STORAGE_BUCKET" ]]; then
    echo "==> Sync outputs to Cloud Storage on exit status $status"
    bash "$REMOTE_ROOT/_tools/remote_tpu_runner.sh" sync-storage \
      --run-id "$RUN_ID" \
      --remote-root "$REMOTE_ROOT" \
      --venv "$VENV" \
      --project-id "$PROJECT_ID" \
      --storage-bucket "$STORAGE_BUCKET" \
      --storage-prefix "$STORAGE_PREFIX" \
      --storage-cache-prefix "$STORAGE_CACHE_PREFIX" || true
  fi
  exit "$status"
}
trap sync_on_exit EXIT

if [[ "0" == "1" ]]; then
  export NUM_BATCHES=4
  export NUM_TEST_BATCHES=2
  export MAX_STEPS=2
  export EVAL_EVERY_N_STEPS=1
  export SAVE_INTERVAL_STEPS=1
  export TOTAL_GENERATION_STEPS=96
  export MAX_PROMPT_LENGTH=128
  export OBS_TRACE_EVERY_N_STEPS=1
fi

echo "==> Base model evaluation"
python -u evaluate.py --no-restore --preset greedy --output-json "$ARTIFACT_DIR/base_eval.json"

echo "==> GRPO baseline training"
python -u train.py

echo "==> LoRA checkpoint evaluation"
python -u evaluate.py --ckpt-dir "$CKPT_DIR/actor" --step 0 --preset greedy \
  --output-json "$ARTIFACT_DIR/baseline_lora_eval.json"

echo "==> Checkpoint-wise evaluation"
python -u evaluate_checkpoints.py \
  --ckpt-dir "$CKPT_DIR/actor" \
  --steps auto \
  --include-base \
  --preset greedy \
  --output-dir "$ARTIFACT_DIR/checkpoint_eval" \
  --skip-existing

echo "==> Export baseline report artifacts"
python -u export_baseline_artifacts.py \
  --tensorboard-dir "$TENSORBOARD_DIR" \
  --output-dir "$ARTIFACT_DIR/baseline" \
  --eval-json "$ARTIFACT_DIR/base_eval.json" \
  --eval-json "$ARTIFACT_DIR/baseline_lora_eval.json"

echo "==> Export GRPO diagnostics"
python -u analyze_grpo_run.py --run-dir "$RUN_DIR" --output-dir "$ARTIFACT_DIR/analysis" || true

echo "==> Baseline pipeline complete"

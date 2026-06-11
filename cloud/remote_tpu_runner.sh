#!/usr/bin/env bash
# Remote TPU VM runner. This script is uploaded and invoked by
# cloud/submit_tpu_job.ps1; it is also usable by hand over SSH.

set -euo pipefail

COMMAND="${1:-}"
if [[ -n "$COMMAND" ]]; then
  shift
fi

RUN_ID=""
BUNDLE=""
SECRETS_FILE=""
REMOTE_ROOT="~/tpu-runs"
REMOTE_VENV="~/venvs/tunix"
TINY_SMOKE=0
PROJECT_ID=""
STORAGE_BUCKET=""
STORAGE_PREFIX="tpu-runs"
STORAGE_CACHE_PREFIX="cache"

usage() {
  cat <<'USAGE'
Usage:
  remote_tpu_runner.sh bootstrap --run-id RUN --bundle /path/code.zip [--secrets /path/.env]
  remote_tpu_runner.sh submit-baseline --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-reward-sweep --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-reward-continuation --run-id RUN --bundle /path/code.zip [--secrets /path/.env]
  remote_tpu_runner.sh submit-candidate-eval --run-id RUN --bundle /path/code.zip [--secrets /path/.env]
  remote_tpu_runner.sh submit-reward-dense --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-r7-large-eval --run-id RUN --bundle /path/code.zip [--secrets /path/.env]
  remote_tpu_runner.sh submit-reward-r9 --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-reward-r10 --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-k8-pilot --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-k8-r10-only --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-k8-r11-fallback-only --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-k8-r12-simple-only --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-k8-r12-simple-full --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-reward-only-r12-full --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-k8-public-beta --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-k8-r13-public-beta-only --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh submit-k8-r14-public-beta-only --run-id RUN --bundle /path/code.zip [--secrets /path/.env] [--tiny-smoke]
  remote_tpu_runner.sh eval-checkpoints --run-id RUN --bundle /path/code.zip [--secrets /path/.env]
  remote_tpu_runner.sh status-sweep --run-id RUN
  remote_tpu_runner.sh status-continuation --run-id RUN
  remote_tpu_runner.sh status-candidate-eval --run-id RUN
  remote_tpu_runner.sh status-reward-dense --run-id RUN
  remote_tpu_runner.sh status-r7-large-eval --run-id RUN
  remote_tpu_runner.sh status-reward-r9 --run-id RUN
  remote_tpu_runner.sh status-reward-r10 --run-id RUN
  remote_tpu_runner.sh status-k8-pilot --run-id RUN
  remote_tpu_runner.sh resume-k8-pilot --run-id RUN
  remote_tpu_runner.sh stop-reward-r10 --run-id RUN
  remote_tpu_runner.sh stop-k8-pilot --run-id RUN
  remote_tpu_runner.sh status --run-id RUN

Options:
  --remote-root PATH   Default: ~/tpu-runs
  --venv PATH          Default: ~/venvs/tunix
  --project-id ID      Google Cloud project used for Cloud Storage commands
  --storage-bucket B   Cloud Storage bucket name, without gs://
  --storage-prefix P   Run output prefix. Default: tpu-runs
  --storage-cache-prefix P
                      Cache prefix. Default: cache
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --bundle)
      BUNDLE="$2"
      shift 2
      ;;
    --secrets)
      SECRETS_FILE="$2"
      shift 2
      ;;
    --remote-root)
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --venv)
      REMOTE_VENV="$2"
      shift 2
      ;;
    --tiny-smoke)
      TINY_SMOKE=1
      shift
      ;;
    --project-id)
      PROJECT_ID="$2"
      shift 2
      ;;
    --storage-bucket)
      STORAGE_BUCKET="$2"
      shift 2
      ;;
    --storage-prefix)
      STORAGE_PREFIX="$2"
      shift 2
      ;;
    --storage-cache-prefix)
      STORAGE_CACHE_PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

expand_path() {
  local path="$1"
  if [[ "$path" == "~" ]]; then
    printf '%s\n' "$HOME"
  elif [[ "$path" == "~/"* ]]; then
    printf '%s/%s\n' "$HOME" "${path#~/}"
  else
    printf '%s\n' "$path"
  fi
}

require_run_id() {
  if [[ -z "$RUN_ID" ]]; then
    echo "--run-id is required" >&2
    exit 2
  fi
  if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_ID may only contain letters, numbers, dot, underscore, and dash." >&2
    exit 2
  fi
}

require_bundle() {
  if [[ -z "$BUNDLE" ]]; then
    echo "--bundle is required for $COMMAND" >&2
    exit 2
  fi
  BUNDLE="$(expand_path "$BUNDLE")"
  if [[ ! -f "$BUNDLE" ]]; then
    echo "Bundle not found: $BUNDLE" >&2
    exit 1
  fi
}

prepare_paths() {
  REMOTE_ROOT="$(expand_path "$REMOTE_ROOT")"
  REMOTE_VENV="$(expand_path "$REMOTE_VENV")"
  RUN_DIR="$REMOTE_ROOT/$RUN_ID"
  SRC_DIR="$RUN_DIR/src"
  ARTIFACT_DIR="$RUN_DIR/artifacts"
  mkdir -p "$RUN_DIR" "$ARTIFACT_DIR"
}

ensure_tmux() {
  if command -v tmux >/dev/null 2>&1; then
    return
  fi
  echo "==> Installing tmux"
  sudo apt-get update
  sudo apt-get install -y tmux
}

normalize_source_scripts() {
  echo "==> Normalizing shell script line endings"
  python3 - "$SRC_DIR" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix != ".sh" and path.name != ".env":
        continue
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    data = data.replace(b"\r\n", b"\n")
    path.write_bytes(data)
PY
  find "$SRC_DIR" -type f -name "*.sh" -print0 \
    | xargs -0 -r chmod +x
}

normalize_secret_files() {
  python3 - "$@" <<'PY'
import pathlib
import sys

for name in sys.argv[1:]:
    path = pathlib.Path(name)
    if not path.exists():
        continue
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    data = data.replace(b"\r\n", b"\n")
    path.write_bytes(data)
PY
}

unpack_bundle() {
  require_bundle
  prepare_paths
  echo "==> Unpacking source bundle into $RUN_DIR"
  rm -rf "$RUN_DIR/unpack" "$SRC_DIR"
  mkdir -p "$RUN_DIR/unpack"
  python3 -m zipfile -e "$BUNDLE" "$RUN_DIR/unpack"

  if [[ ! -d "$RUN_DIR/unpack/src" ]]; then
    echo "Bundle must contain a src/ directory." >&2
    exit 1
  fi
  mv "$RUN_DIR/unpack/src" "$SRC_DIR"
  normalize_source_scripts

  rm -rf "$RUN_DIR/meta"
  if [[ -d "$RUN_DIR/unpack/meta" ]]; then
    mv "$RUN_DIR/unpack/meta" "$RUN_DIR/meta"
  fi
  rm -rf "$RUN_DIR/unpack"
}

install_secrets() {
  if [[ -z "$SECRETS_FILE" ]]; then
    return
  fi
  SECRETS_FILE="$(expand_path "$SECRETS_FILE")"
  if [[ ! -f "$SECRETS_FILE" ]]; then
    echo "Secrets file not found: $SECRETS_FILE" >&2
    exit 1
  fi
  echo "==> Installing secrets into run-local .env files"
  cp "$SECRETS_FILE" "$RUN_DIR/.env"
  cp "$SECRETS_FILE" "$SRC_DIR/.env"
  cp "$SECRETS_FILE" "$SRC_DIR/scripts/.env"
  normalize_secret_files "$RUN_DIR/.env" "$SRC_DIR/.env" "$SRC_DIR/scripts/.env"
  chmod 600 "$RUN_DIR/.env" "$SRC_DIR/.env" "$SRC_DIR/scripts/.env"
}

ensure_uv_python() {
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  echo "==> Ensuring Python 3.12 is available"
  uv python install 3.12
}

bootstrap_env() {
  prepare_paths
  if [[ ! -x "$SRC_DIR/bootstrap.sh" ]]; then
    chmod +x "$SRC_DIR/bootstrap.sh"
  fi
  ensure_tmux
  ensure_uv_python
  echo "==> Running repository bootstrap"
  (cd "$SRC_DIR" && VENV="$REMOTE_VENV" ./bootstrap.sh)
}

check_tpu_backend() {
  echo "==> Verifying JAX sees the TPU"
  # shellcheck disable=SC1091
  source "$REMOTE_VENV/bin/activate"
  python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
if jax.default_backend() != "tpu":
    raise SystemExit("JAX backend is not TPU; refusing to start training.")
PY
}

write_baseline_script() {
  local run_script="$RUN_DIR/run_baseline.sh"
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
ARTIFACT_DIR="$ARTIFACT_DIR"
REMOTE_ROOT="$REMOTE_ROOT"
PROJECT_ID="$PROJECT_ID"
STORAGE_BUCKET="$STORAGE_BUCKET"
STORAGE_PREFIX="$STORAGE_PREFIX"
STORAGE_CACHE_PREFIX="$STORAGE_CACHE_PREFIX"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "\$VENV/bin/activate"

export RUN_ID="$RUN_ID"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CKPT_DIR="\$RUN_DIR/ckpts"
export INTERMEDIATE_CKPT_DIR="\$RUN_DIR/intermediate_ckpt"
export TENSORBOARD_DIR="\$RUN_DIR/tensorboard"
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export WANDB_RUN_ID="\${WANDB_RUN_ID:-$RUN_ID}"
export WANDB_PROJECT="\${WANDB_PROJECT:-grpo-tpu-2026}"
export OBS_OUTPUT_DIR="\$ARTIFACT_DIR/observability"
export OBS_TRACE_DIR="\$ARTIFACT_DIR/rollout_traces"
export OBS_RUN_MANIFEST="\$ARTIFACT_DIR/run_manifest.json"
export OBS_TRACE_EVERY_N_STEPS="\${OBS_TRACE_EVERY_N_STEPS:-64}"
export OBS_TRACE_MAX_ROWS="\${OBS_TRACE_MAX_ROWS:-32}"
if [[ -f "\$RUN_DIR/meta/git_commit.txt" ]]; then
  export GIT_COMMIT="\$(cat "\$RUN_DIR/meta/git_commit.txt")"
fi
if [[ -f "\$RUN_DIR/meta/git_status.txt" ]]; then
  export GIT_STATUS_SHORT="\$(cat "\$RUN_DIR/meta/git_status.txt")"
fi
mkdir -p "\$ARTIFACT_DIR" "\$CKPT_DIR" "\$INTERMEDIATE_CKPT_DIR" "\$TENSORBOARD_DIR" "\$OBS_OUTPUT_DIR" "\$OBS_TRACE_DIR"

if [[ -n "\${WANDB_API_KEY:-}" ]]; then
  echo "==> W&B enabled: project=\$WANDB_PROJECT entity=\${WANDB_ENTITY:-<default>}"
else
  if [[ -n "\$STORAGE_BUCKET" ]]; then
    echo "==> W&B disabled: WANDB_API_KEY is not set; using TensorBoard + GCS."
  else
    echo "==> W&B disabled: WANDB_API_KEY is not set; using TensorBoard + local artifacts."
  fi
fi

sync_on_exit() {
  local status=\$?
  if [[ -n "\$STORAGE_BUCKET" ]]; then
    echo "==> Sync outputs to Cloud Storage on exit status \$status"
    bash "\$REMOTE_ROOT/_tools/remote_tpu_runner.sh" sync-storage \\
      --run-id "\$RUN_ID" \\
      --remote-root "\$REMOTE_ROOT" \\
      --venv "\$VENV" \\
      --project-id "\$PROJECT_ID" \\
      --storage-bucket "\$STORAGE_BUCKET" \\
      --storage-prefix "\$STORAGE_PREFIX" \\
      --storage-cache-prefix "\$STORAGE_CACHE_PREFIX" || true
  fi
  exit "\$status"
}
trap sync_on_exit EXIT

if [[ "$TINY_SMOKE" == "1" ]]; then
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
python -u evaluate.py --no-restore --preset greedy --output-json "\$ARTIFACT_DIR/base_eval.json"

echo "==> GRPO baseline training"
python -u train.py

echo "==> LoRA checkpoint evaluation"
python -u evaluate.py --ckpt-dir "\$CKPT_DIR/actor" --step 0 --preset greedy \\
  --output-json "\$ARTIFACT_DIR/baseline_lora_eval.json"

echo "==> Checkpoint-wise evaluation"
python -u evaluate_checkpoints.py \\
  --ckpt-dir "\$CKPT_DIR/actor" \\
  --steps auto \\
  --include-base \\
  --preset greedy \\
  --output-dir "\$ARTIFACT_DIR/checkpoint_eval" \\
  --skip-existing

echo "==> Export baseline report artifacts"
python -u export_baseline_artifacts.py \\
  --tensorboard-dir "\$TENSORBOARD_DIR" \\
  --output-dir "\$ARTIFACT_DIR/baseline" \\
  --eval-json "\$ARTIFACT_DIR/base_eval.json" \\
  --eval-json "\$ARTIFACT_DIR/baseline_lora_eval.json"

echo "==> Export GRPO diagnostics"
python -u analyze_grpo_run.py --run-dir "\$RUN_DIR" --output-dir "\$ARTIFACT_DIR/analysis" || true

echo "==> Baseline pipeline complete"
EOF
  chmod +x "$run_script"
}

submit_baseline() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_baseline_script

  local session="tpu-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_baseline.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- pipeline exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

write_reward_sweep_script() {
  local run_script="$RUN_DIR/run_reward_sweep.sh"
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SWEEP_ID="$RUN_ID"
RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
PARENT_ARTIFACT_DIR="$ARTIFACT_DIR"
REMOTE_ROOT="$REMOTE_ROOT"
PROJECT_ID="$PROJECT_ID"
STORAGE_BUCKET="$STORAGE_BUCKET"
STORAGE_PREFIX="$STORAGE_PREFIX"
STORAGE_CACHE_PREFIX="$STORAGE_CACHE_PREFIX"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "\$VENV/bin/activate"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_PROJECT="\${WANDB_PROJECT:-grpo-tpu-2026}"
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export NUM_TEST_BATCHES="\${SWEEP_NUM_TEST_BATCHES:-64}"
export TOTAL_GENERATION_STEPS="\${SWEEP_TOTAL_GENERATION_STEPS:-768}"
export MAX_PROMPT_LENGTH="\${MAX_PROMPT_LENGTH:-256}"
mkdir -p "\$PARENT_ARTIFACT_DIR" "\$RUN_DIR/runs" "\$TRAIN_DATA_DIR" "\$TEST_DATA_DIR"

sync_on_exit() {
  local status=\$?
  if [[ -n "\$STORAGE_BUCKET" ]]; then
    echo "==> Sync reward sweep outputs to Cloud Storage on exit status \$status"
    bash "\$REMOTE_ROOT/_tools/remote_tpu_runner.sh" sync-storage \\
      --run-id "\$SWEEP_ID" \\
      --remote-root "\$REMOTE_ROOT" \\
      --venv "\$VENV" \\
      --project-id "\$PROJECT_ID" \\
      --storage-bucket "\$STORAGE_BUCKET" \\
      --storage-prefix "\$STORAGE_PREFIX" \\
      --storage-cache-prefix "\$STORAGE_CACHE_PREFIX" || true
  fi
  exit "\$status"
}
trap sync_on_exit EXIT

cat > "\$PARENT_ARTIFACT_DIR/reward_sweep_manifest.json" <<JSON
{
  "sweep_id": "$RUN_ID",
  "max_steps": "\${SWEEP_MAX_STEPS:-768}",
  "lr_schedule_steps": "\${SWEEP_LR_SCHEDULE_STEPS:-3364}",
  "warmup_steps": "\${SWEEP_WARMUP_STEPS:-336.4}",
  "save_interval_steps": "\${SWEEP_SAVE_INTERVAL_STEPS:-256}",
  "max_to_keep": "\${SWEEP_MAX_TO_KEEP:-5}",
  "eval_every_n_steps": "\${SWEEP_EVAL_EVERY_N_STEPS:-64}",
  "num_generations": "\${SWEEP_NUM_GENERATIONS:-2}",
  "total_generation_steps": "\${SWEEP_TOTAL_GENERATION_STEPS:-768}",
  "learning_rate": "\${SWEEP_LEARNING_RATE:-3e-6}",
  "beta": "\${SWEEP_BETA:-0.08}",
  "epsilon": "\${SWEEP_EPSILON:-0.2}",
  "checkpoint_eval_steps": [256, 512, 768],
  "reward_modes": [
    "no_approx",
    "light_format_oldnum",
    "numeric_primary_no_len",
    "numeric_primary_len1200",
    "numeric_primary_answer_only_len1200"
  ]
}
JSON

if [[ "$TINY_SMOKE" == "1" ]]; then
  export SWEEP_MAX_STEPS=2
  export SWEEP_LR_SCHEDULE_STEPS=3364
  export SWEEP_WARMUP_STEPS=336.4
  export SWEEP_SAVE_INTERVAL_STEPS=1
  export SWEEP_MAX_TO_KEEP=5
  export SWEEP_EVAL_EVERY_N_STEPS=1
  export SWEEP_TOTAL_GENERATION_STEPS=96
  export NUM_TEST_BATCHES=2
  export MAX_PROMPT_LENGTH=128
fi

echo "==> Reward sweep base evaluation"
python -u evaluate.py --no-restore --preset greedy --output-json "\$PARENT_ARTIFACT_DIR/base_eval.json"

declare -a SWEEP_RUNS=(
  "R1_no_approx:no_approx"
  "R2_light_format_oldnum:light_format_oldnum"
  "R3_numeric_primary_no_len:numeric_primary_no_len"
  "R4_numeric_primary_len1200:numeric_primary_len1200"
  "R5_numeric_primary_answer_only_len1200:numeric_primary_answer_only_len1200"
)

for spec in "\${SWEEP_RUNS[@]}"; do
  EXP_ID="\${spec%%:*}"
  MODE="\${spec#*:}"
  EXP_DIR="\$RUN_DIR/runs/\$EXP_ID"
  ARTIFACT_DIR="\$EXP_DIR/artifacts"
  mkdir -p "\$EXP_DIR" "\$ARTIFACT_DIR" "\$EXP_DIR/ckpts" "\$EXP_DIR/intermediate_ckpt" "\$EXP_DIR/tensorboard"

  echo
  echo "==> Reward sweep run \$EXP_ID mode=\$MODE"
  export RUN_ID="\$SWEEP_ID-\$EXP_ID"
  export WANDB_RUN_ID="\$SWEEP_ID-\$EXP_ID"
  export REWARD_MODE="\$MODE"
  export CKPT_DIR="\$EXP_DIR/ckpts"
  export INTERMEDIATE_CKPT_DIR="\$EXP_DIR/intermediate_ckpt"
  export TENSORBOARD_DIR="\$EXP_DIR/tensorboard"
  export OBS_OUTPUT_DIR="\$ARTIFACT_DIR/observability"
  export OBS_TRACE_DIR="\$ARTIFACT_DIR/rollout_traces"
  export OBS_RUN_MANIFEST="\$ARTIFACT_DIR/run_manifest.json"
  export OBS_TRACE_EVERY_N_STEPS="\${SWEEP_OBS_TRACE_EVERY_N_STEPS:-1}"
  export OBS_TRACE_MAX_ROWS="\${SWEEP_OBS_TRACE_MAX_ROWS:-4096}"
  export MAX_STEPS="\${SWEEP_MAX_STEPS:-768}"
  export LR_SCHEDULE_STEPS="\${SWEEP_LR_SCHEDULE_STEPS:-3364}"
  export WARMUP_STEPS="\${SWEEP_WARMUP_STEPS:-336.4}"
  export SAVE_INTERVAL_STEPS="\${SWEEP_SAVE_INTERVAL_STEPS:-256}"
  export MAX_TO_KEEP="\${SWEEP_MAX_TO_KEEP:-5}"
  export EVAL_EVERY_N_STEPS="\${SWEEP_EVAL_EVERY_N_STEPS:-64}"
  export NUM_GENERATIONS="\${SWEEP_NUM_GENERATIONS:-2}"
  export LEARNING_RATE="\${SWEEP_LEARNING_RATE:-3e-6}"
  export BETA="\${SWEEP_BETA:-0.08}"
  export EPSILON="\${SWEEP_EPSILON:-0.2}"
  if [[ -f "\$RUN_DIR/meta/git_commit.txt" ]]; then
    export GIT_COMMIT="\$(cat "\$RUN_DIR/meta/git_commit.txt")"
  fi
  if [[ -f "\$RUN_DIR/meta/git_status.txt" ]]; then
    export GIT_STATUS_SHORT="\$(cat "\$RUN_DIR/meta/git_status.txt")"
  fi
  mkdir -p "\$OBS_OUTPUT_DIR" "\$OBS_TRACE_DIR"

  env | sort | grep -E '^(RUN_ID|REWARD_MODE|MAX_STEPS|LR_SCHEDULE_STEPS|WARMUP_STEPS|SAVE_INTERVAL_STEPS|MAX_TO_KEEP|EVAL_EVERY_N_STEPS|NUM_GENERATIONS|TOTAL_GENERATION_STEPS|LEARNING_RATE|BETA|EPSILON|CKPT_DIR|TENSORBOARD_DIR|OBS_)=' > "\$EXP_DIR/run_env.txt"
  printf '%s\n' "\$MODE" > "\$EXP_DIR/reward_mode.txt"

  python -u train.py 2>&1 | tee -a "\$EXP_DIR/train.log"

  if [[ "$TINY_SMOKE" == "1" ]]; then
    CHECKPOINT_STEPS=(1 2)
  else
    CHECKPOINT_STEPS=(256 512 768)
  fi

  python -u evaluate_checkpoints.py \\
    --ckpt-dir "\$CKPT_DIR/actor" \\
    --steps "\${CHECKPOINT_STEPS[@]}" \\
    --preset greedy \\
    --output-dir "\$ARTIFACT_DIR/checkpoint_eval" \\
    --skip-existing \\
    --continue-on-error 2>&1 | tee -a "\$EXP_DIR/eval_checkpoints.log" || true

  python -u analyze_grpo_run.py --run-dir "\$EXP_DIR" --output-dir "\$ARTIFACT_DIR/analysis" || true
done

echo "==> Build reward sweep analysis package"
python -u analyze_reward_sweep.py --input-dir "\$RUN_DIR" --output-dir "\$PARENT_ARTIFACT_DIR/sweep_analysis" || true

echo "==> Reward sweep complete"
EOF
  chmod +x "$run_script"
}

write_reward_continuation_script() {
  local run_script="$RUN_DIR/run_reward_continuation.sh"
cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

CONTINUATION_ID="$RUN_ID"
RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
PARENT_ARTIFACT_DIR="$ARTIFACT_DIR"
REMOTE_ROOT="$REMOTE_ROOT"
PROJECT_ID="$PROJECT_ID"
STORAGE_BUCKET="$STORAGE_BUCKET"
STORAGE_PREFIX="$STORAGE_PREFIX"
STORAGE_CACHE_PREFIX="$STORAGE_CACHE_PREFIX"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f "\$RUN_DIR/.env" ]]; then
  source "\$RUN_DIR/.env"
fi
set +a
source "\$VENV/bin/activate"

export RUN_ROOT="\$RUN_DIR"
export SOURCE_SWEEP_ID="\${CONTINUATION_SOURCE_RUN_ID:-reward-grid-001}"
export SOURCE_SWEEP_DIR="\$REMOTE_ROOT/\$SOURCE_SWEEP_ID"
export PARENT_ARTIFACT_DIR="\$PARENT_ARTIFACT_DIR"
export HF_HOME="\${HF_HOME:-\$HOME/.cache/huggingface}"
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export DATA_SOURCE="\${DATA_SOURCE:-tfds}"
export MAX_PROMPT_LENGTH="\${MAX_PROMPT_LENGTH:-256}"
export TOTAL_GENERATION_STEPS="\${CONT_TOTAL_GENERATION_STEPS:-768}"
mkdir -p "\$PARENT_ARTIFACT_DIR" "\$RUN_DIR/runs" "\$TRAIN_DATA_DIR" "\$TEST_DATA_DIR"

sync_on_exit() {
  local status=\$?
  if [[ -n "$STORAGE_BUCKET" ]]; then
    echo "==> Sync reward continuation outputs to Cloud Storage on exit status \$status"
    bash "\$REMOTE_ROOT/_tools/remote_tpu_runner.sh" sync-storage \\
      --run-id "\$CONTINUATION_ID" \\
      --remote-root "\$REMOTE_ROOT" \\
      --venv "\$VENV" \\
      --project-id "\$PROJECT_ID" \\
      --storage-bucket "\$STORAGE_BUCKET" \\
      --storage-prefix "\$STORAGE_PREFIX" \\
      --storage-cache-prefix "\$STORAGE_CACHE_PREFIX" || true
  fi
  exit "\$status"
}
trap sync_on_exit EXIT

cat > "\$PARENT_ARTIFACT_DIR/reward_continuation_manifest.json" <<JSON
{
  "continuation_id": "\${CONTINUATION_ID}",
  "source_sweep_id": "\${SOURCE_SWEEP_ID}",
  "max_steps": "\${CONT_MAX_STEPS:-1536}",
  "lr_schedule_steps": "\${CONT_LR_SCHEDULE_STEPS:-3364}",
  "warmup_steps": "\${CONT_WARMUP_STEPS:-336.4}",
  "save_interval_steps": "\${CONT_SAVE_INTERVAL_STEPS:-128}",
  "max_to_keep": "\${CONT_MAX_TO_KEEP:-20}",
  "eval_every_n_steps": "\${CONT_EVAL_EVERY_N_STEPS:-64}",
  "num_generations": "\${CONT_NUM_GENERATIONS:-2}",
  "total_generation_steps": "\${CONT_TOTAL_GENERATION_STEPS:-768}",
  "learning_rate": "\${CONT_LEARNING_RATE:-3e-6}",
  "beta": "\${CONT_BETA:-0.08}",
  "epsilon": "\${CONT_EPSILON:-0.2}",
  "runs": [
    {"run_id": "C1_R1_no_approx_from256", "reward_mode": "no_approx", "source_run": "R1_no_approx", "source_step": 256, "eval_steps": [256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536]},
    {"run_id": "C2_R3_numeric_primary_no_len_from768", "reward_mode": "numeric_primary_no_len", "source_run": "R3_numeric_primary_no_len", "source_step": 768, "eval_steps": [768, 896, 1024, 1152, 1280, 1408, 1536]},
    {"run_id": "C3_R5_numeric_primary_answer_only_len1200_from512", "reward_mode": "numeric_primary_answer_only_len1200", "source_run": "R5_numeric_primary_answer_only_len1200", "source_step": 512, "eval_steps": [512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536]}
  ]
}
JSON

if [[ "$TINY_SMOKE" == "1" ]]; then
  export CONT_MAX_STEPS=4
  export CONT_LR_SCHEDULE_STEPS=3364
  export CONT_WARMUP_STEPS=336.4
  export CONT_SAVE_INTERVAL_STEPS=1
  export CONT_MAX_TO_KEEP=20
  export CONT_EVAL_EVERY_N_STEPS=1
  export CONT_TOTAL_GENERATION_STEPS=96
  export NUM_TEST_BATCHES=2
  export MAX_PROMPT_LENGTH=128
fi

if [[ ! -d "\$SOURCE_SWEEP_DIR/runs" ]]; then
  echo "Source sweep directory not found: \$SOURCE_SWEEP_DIR/runs" >&2
  exit 1
fi

declare -a CONT_RUNS=(
  "C1_R1_no_approx_from256:no_approx:R1_no_approx:256:256 384 512 640 768 896 1024 1152 1280 1408 1536"
  "C2_R3_numeric_primary_no_len_from768:numeric_primary_no_len:R3_numeric_primary_no_len:768:768 896 1024 1152 1280 1408 1536"
  "C3_R5_numeric_primary_answer_only_len1200_from512:numeric_primary_answer_only_len1200:R5_numeric_primary_answer_only_len1200:512:512 640 768 896 1024 1152 1280 1408 1536"
)

for spec in "\${CONT_RUNS[@]}"; do
  IFS=':' read -r EXP_ID MODE SOURCE_EXP SOURCE_STEP EVAL_STEPS_STR <<< "\$spec"
  EXP_DIR="$RUN_DIR/runs/\$EXP_ID"
  ARTIFACT_DIR="\$EXP_DIR/artifacts"
  SOURCE_CKPT="\$SOURCE_SWEEP_DIR/runs/\$SOURCE_EXP/ckpts/actor/\$SOURCE_STEP"

  echo
  echo "==> Reward continuation run \$EXP_ID mode=\$MODE source=\$SOURCE_EXP step=\$SOURCE_STEP"
  if [[ ! -d "\$SOURCE_CKPT" ]]; then
    echo "Source checkpoint missing: \$SOURCE_CKPT" >&2
    exit 1
  fi
  if [[ -e "\$EXP_DIR" ]]; then
    echo "Continuation run directory already exists; refusing to overwrite: \$EXP_DIR" >&2
    exit 1
  fi

  mkdir -p "\$EXP_DIR/ckpts/actor" "\$ARTIFACT_DIR" "\$EXP_DIR/intermediate_ckpt" "\$EXP_DIR/tensorboard"
  cp -a "\$SOURCE_CKPT" "\$EXP_DIR/ckpts/actor/\$SOURCE_STEP"

  cat > "\$EXP_DIR/branch_metadata.json" <<JSON
{
  "continuation_id": "$RUN_ID",
  "run_id": "\$EXP_ID",
  "reward_mode": "\$MODE",
  "source_sweep_id": "\$SOURCE_SWEEP_ID",
  "source_run": "\$SOURCE_EXP",
  "source_step": \$SOURCE_STEP,
  "source_checkpoint": "\$SOURCE_CKPT",
  "copied_checkpoint": "\$EXP_DIR/ckpts/actor/\$SOURCE_STEP"
}
JSON

  export RUN_ID="$RUN_ID-\$EXP_ID"
  export WANDB_RUN_ID="$RUN_ID-\$EXP_ID"
  export REWARD_MODE="\$MODE"
  export CKPT_DIR="\$EXP_DIR/ckpts"
  export INTERMEDIATE_CKPT_DIR="\$EXP_DIR/intermediate_ckpt"
  export TENSORBOARD_DIR="\$EXP_DIR/tensorboard"
  export OBS_OUTPUT_DIR="\$ARTIFACT_DIR/observability"
  export OBS_TRACE_DIR="\$ARTIFACT_DIR/rollout_traces"
  export OBS_RUN_MANIFEST="\$ARTIFACT_DIR/run_manifest.json"
  export OBS_TRACE_EVERY_N_STEPS="\${CONT_OBS_TRACE_EVERY_N_STEPS:-1}"
  export OBS_TRACE_MAX_ROWS="\${CONT_OBS_TRACE_MAX_ROWS:-4096}"
  export MAX_STEPS="\${CONT_MAX_STEPS:-1536}"
  export LR_SCHEDULE_STEPS="\${CONT_LR_SCHEDULE_STEPS:-3364}"
  export WARMUP_STEPS="\${CONT_WARMUP_STEPS:-336.4}"
  export SAVE_INTERVAL_STEPS="\${CONT_SAVE_INTERVAL_STEPS:-128}"
  export MAX_TO_KEEP="\${CONT_MAX_TO_KEEP:-20}"
  export EVAL_EVERY_N_STEPS="\${CONT_EVAL_EVERY_N_STEPS:-64}"
  export NUM_GENERATIONS="\${CONT_NUM_GENERATIONS:-2}"
  export LEARNING_RATE="\${CONT_LEARNING_RATE:-3e-6}"
  export BETA="\${CONT_BETA:-0.08}"
  export EPSILON="\${CONT_EPSILON:-0.2}"
  if [[ -f "$RUN_DIR/meta/git_commit.txt" ]]; then
    export GIT_COMMIT="\$(cat "$RUN_DIR/meta/git_commit.txt")"
  fi
  if [[ -f "$RUN_DIR/meta/git_status.txt" ]]; then
    export GIT_STATUS_SHORT="\$(cat "$RUN_DIR/meta/git_status.txt")"
  fi
  mkdir -p "\$OBS_OUTPUT_DIR" "\$OBS_TRACE_DIR"

  env | sort | grep -E '^(RUN_ID|REWARD_MODE|MAX_STEPS|LR_SCHEDULE_STEPS|WARMUP_STEPS|SAVE_INTERVAL_STEPS|MAX_TO_KEEP|EVAL_EVERY_N_STEPS|NUM_GENERATIONS|TOTAL_GENERATION_STEPS|LEARNING_RATE|BETA|EPSILON|CKPT_DIR|TENSORBOARD_DIR|OBS_)=' > "\$EXP_DIR/run_env.txt"
  printf '%s\n' "\$MODE" > "\$EXP_DIR/reward_mode.txt"

  python -u train.py 2>&1 | tee -a "\$EXP_DIR/train.log"

  if [[ "$TINY_SMOKE" == "1" ]]; then
    CHECKPOINT_STEPS=( "\$SOURCE_STEP" "\$MAX_STEPS" )
  else
    read -r -a CHECKPOINT_STEPS <<< "\$EVAL_STEPS_STR"
  fi

  python -u evaluate_checkpoints.py \\
    --ckpt-dir "\$CKPT_DIR/actor" \\
    --steps "\${CHECKPOINT_STEPS[@]}" \\
    --preset greedy \\
    --output-dir "\$ARTIFACT_DIR/checkpoint_eval" \\
    --skip-existing \\
    --continue-on-error 2>&1 | tee -a "\$EXP_DIR/eval_checkpoints.log" || true

  python -u analyze_grpo_run.py --run-dir "\$EXP_DIR" --output-dir "\$ARTIFACT_DIR/analysis" || true
done

echo "==> Build reward continuation analysis package"
python -u analyze_reward_continuation.py --input-dir "$RUN_DIR" --output-dir "\$PARENT_ARTIFACT_DIR/continuation_analysis" || true

echo "==> Reward continuation complete"
EOF
  chmod +x "$run_script"
}

submit_reward_sweep() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_reward_sweep_script

  local session="tpu-sweep-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_reward_sweep.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- reward sweep exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_reward_continuation() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_reward_continuation_script

  local session="tpu-cont-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_reward_continuation.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- reward continuation exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

write_candidate_eval_script() {
  local run_script="$RUN_DIR/run_candidate_eval.sh"
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

CANDIDATE_ID="$RUN_ID"
RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
ARTIFACT_DIR="$ARTIFACT_DIR"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "\$VENV/bin/activate"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DATA_SOURCE=tfds
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export NUM_TEST_BATCHES="\${CANDIDATE_NUM_TEST_BATCHES:-256}"
export TOTAL_GENERATION_STEPS="\${CANDIDATE_TOTAL_GENERATION_STEPS:-768}"
export MAX_PROMPT_LENGTH="\${MAX_PROMPT_LENGTH:-256}"

OUT_DIR="\$ARTIFACT_DIR/candidate_eval"
mkdir -p "\$OUT_DIR" "\$TRAIN_DATA_DIR" "\$TEST_DATA_DIR"

R5_CKPT_ROOT="\${CANDIDATE_R5_CKPT_ROOT:-$REMOTE_ROOT/reward-grid-001/runs/R5_numeric_primary_answer_only_len1200/ckpts/actor}"
R3_CKPT_ROOT="\${CANDIDATE_R3_CKPT_ROOT:-$REMOTE_ROOT/reward-continuation-001/runs/C2_R3_numeric_primary_no_len_from768/ckpts/actor}"

echo "==> Candidate large eval"
echo "NUM_TEST_BATCHES=\$NUM_TEST_BATCHES"
echo "R5_CKPT_ROOT=\$R5_CKPT_ROOT"
echo "R3_CKPT_ROOT=\$R3_CKPT_ROOT"

python -u evaluate.py --no-restore --preset greedy --source tfds --output-json "\$OUT_DIR/base_eval.json"

if [[ ! -d "\$R5_CKPT_ROOT/512" ]]; then
  echo "Missing R5 checkpoint: \$R5_CKPT_ROOT/512" >&2
  exit 1
fi
python -u evaluate.py --ckpt-dir "\$R5_CKPT_ROOT" --step 512 --preset greedy --source tfds --output-json "\$OUT_DIR/R5_step512_eval.json"

if [[ ! -d "\$R3_CKPT_ROOT/1408" ]]; then
  echo "Missing R3 checkpoint: \$R3_CKPT_ROOT/1408" >&2
  exit 1
fi
python -u evaluate.py --ckpt-dir "\$R3_CKPT_ROOT" --step 1408 --preset greedy --source tfds --output-json "\$OUT_DIR/R3_step1408_eval.json"

python - "\$OUT_DIR" <<'PY'
import csv
import json
import os
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
specs = [
    ("base", None, out / "base_eval.json"),
    ("R5_step512", 512, out / "R5_step512_eval.json"),
    ("R3_step1408", 1408, out / "R3_step1408_eval.json"),
]
rows = []
for label, step, path in specs:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    rows.append({
        "label": label,
        "step": step,
        "correct": metrics.get("correct"),
        "total": metrics.get("total"),
        "accuracy": metrics.get("accuracy"),
        "partial_accuracy": metrics.get("partial_accuracy"),
        "format_accuracy": metrics.get("format_accuracy"),
        "file": str(path),
    })
summary = {
    "num_test_batches": int(os.environ.get("NUM_TEST_BATCHES", "256")),
    "reference_label": "R5_step512",
    "reference_accuracy": next(row["accuracy"] for row in rows if row["label"] == "R5_step512"),
    "rows": rows,
}
(out / "candidate_eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with (out / "candidate_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("Wrote candidate eval summary")
PY

echo "==> Candidate eval complete"
EOF
  chmod +x "$run_script"
}

submit_candidate_eval() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_candidate_eval_script

  local session="tpu-candidate-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_candidate_eval.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- candidate eval exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

write_reward_dense_script() {
  local run_script="$RUN_DIR/run_reward_dense.sh"
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

DENSE_ID="$RUN_ID"
RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
PARENT_ARTIFACT_DIR="$ARTIFACT_DIR"
REMOTE_ROOT="$REMOTE_ROOT"
PROJECT_ID="$PROJECT_ID"
STORAGE_BUCKET="$STORAGE_BUCKET"
STORAGE_PREFIX="$STORAGE_PREFIX"
STORAGE_CACHE_PREFIX="$STORAGE_CACHE_PREFIX"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "\$VENV/bin/activate"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DATA_SOURCE=tfds
export WANDB_PROJECT="\${WANDB_PROJECT:-grpo-tpu-2026}"
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export NUM_TEST_BATCHES="\${DENSE_NUM_TEST_BATCHES:-64}"
export TOTAL_GENERATION_STEPS="\${DENSE_TOTAL_GENERATION_STEPS:-768}"
export MAX_PROMPT_LENGTH="\${MAX_PROMPT_LENGTH:-256}"
mkdir -p "\$PARENT_ARTIFACT_DIR" "\$RUN_DIR/runs" "\$TRAIN_DATA_DIR" "\$TEST_DATA_DIR"

sync_on_exit() {
  local status=\$?
  if [[ -n "\$STORAGE_BUCKET" ]]; then
    echo "==> Sync reward dense outputs to Cloud Storage on exit status \$status"
    bash "\$REMOTE_ROOT/_tools/remote_tpu_runner.sh" sync-storage \\
      --run-id "\$DENSE_ID" \\
      --remote-root "\$REMOTE_ROOT" \\
      --venv "\$VENV" \\
      --project-id "\$PROJECT_ID" \\
      --storage-bucket "\$STORAGE_BUCKET" \\
      --storage-prefix "\$STORAGE_PREFIX" \\
      --storage-cache-prefix "\$STORAGE_CACHE_PREFIX" || true
  fi
  exit "\$status"
}
trap sync_on_exit EXIT

cat > "\$PARENT_ARTIFACT_DIR/reward_dense_manifest.json" <<JSON
{
  "sweep_id": "$RUN_ID",
  "max_steps": "\${DENSE_MAX_STEPS:-768}",
  "lr_schedule_steps": "\${DENSE_LR_SCHEDULE_STEPS:-3364}",
  "warmup_steps": "\${DENSE_WARMUP_STEPS:-336.4}",
  "save_interval_steps": "\${DENSE_SAVE_INTERVAL_STEPS:-256}",
  "max_to_keep": "\${DENSE_MAX_TO_KEEP:-5}",
  "eval_every_n_steps": "\${DENSE_EVAL_EVERY_N_STEPS:-64}",
  "num_generations": "\${DENSE_NUM_GENERATIONS:-2}",
  "total_generation_steps": "\${DENSE_TOTAL_GENERATION_STEPS:-768}",
  "learning_rate": "\${DENSE_LEARNING_RATE:-3e-6}",
  "beta": "\${DENSE_BETA:-0.08}",
  "epsilon": "\${DENSE_EPSILON:-0.2}",
  "checkpoint_eval_steps": [256, 512, 768],
  "conditional_extension_max_steps": 1024,
  "runs": [
    {"run_id": "R6_numeric_dense_lastnum", "reward_mode": "numeric_dense_lastnum"},
    {"run_id": "R7_numeric_dense_single_answer", "reward_mode": "numeric_dense_single_answer"},
    {"run_id": "R8_numeric_dense_single_answer_short", "reward_mode": "numeric_dense_single_answer_short"}
  ]
}
JSON

if [[ "$TINY_SMOKE" == "1" ]]; then
  export DENSE_MAX_STEPS=2
  export DENSE_SAVE_INTERVAL_STEPS=1
  export DENSE_MAX_TO_KEEP=5
  export DENSE_EVAL_EVERY_N_STEPS=1
  export DENSE_TOTAL_GENERATION_STEPS=96
  export NUM_TEST_BATCHES=2
  export MAX_PROMPT_LENGTH=128
fi

REFERENCE_ACCURACY="\${DENSE_REFERENCE_ACCURACY:-}"
if [[ -z "\$REFERENCE_ACCURACY" ]]; then
  CANDIDATE_SUMMARY="\$REMOTE_ROOT/\${DENSE_REFERENCE_RUN_ID:-candidate-eval-r3-r5-001}/artifacts/candidate_eval/candidate_eval_summary.json"
  if [[ -f "\$CANDIDATE_SUMMARY" ]]; then
    REFERENCE_ACCURACY="\$(python - "\$CANDIDATE_SUMMARY" <<'PY'
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(payload.get("reference_accuracy") or 57.8125)
PY
)"
  else
    REFERENCE_ACCURACY="57.8125"
  fi
fi
REFERENCE_PARTIAL="\${DENSE_REFERENCE_PARTIAL:-64.0625}"
echo "Reference accuracy: \$REFERENCE_ACCURACY"

declare -a DENSE_RUNS=(
  "R6_numeric_dense_lastnum:numeric_dense_lastnum"
  "R7_numeric_dense_single_answer:numeric_dense_single_answer"
  "R8_numeric_dense_single_answer_short:numeric_dense_single_answer_short"
)

EXTENSION_SPECS="\$RUN_DIR/dense_extension_specs.txt"
: > "\$EXTENSION_SPECS"

run_dense_experiment() {
  local exp_id="\$1"
  local mode="\$2"
  local exp_dir="\$RUN_DIR/runs/\$exp_id"
  local artifact_dir="\$exp_dir/artifacts"
  mkdir -p "\$exp_dir" "\$artifact_dir" "\$exp_dir/ckpts" "\$exp_dir/intermediate_ckpt" "\$exp_dir/tensorboard"

  echo
  echo "==> Reward dense run \$exp_id mode=\$mode"
  export RUN_ID="\$DENSE_ID-\$exp_id"
  export WANDB_RUN_ID="\$DENSE_ID-\$exp_id"
  export REWARD_MODE="\$mode"
  export CKPT_DIR="\$exp_dir/ckpts"
  export INTERMEDIATE_CKPT_DIR="\$exp_dir/intermediate_ckpt"
  export TENSORBOARD_DIR="\$exp_dir/tensorboard"
  export OBS_OUTPUT_DIR="\$artifact_dir/observability"
  export OBS_TRACE_DIR="\$artifact_dir/rollout_traces"
  export OBS_RUN_MANIFEST="\$artifact_dir/run_manifest.json"
  export OBS_TRACE_EVERY_N_STEPS="\${DENSE_OBS_TRACE_EVERY_N_STEPS:-1}"
  export OBS_TRACE_MAX_ROWS="\${DENSE_OBS_TRACE_MAX_ROWS:-4096}"
  export MAX_STEPS="\${DENSE_MAX_STEPS:-768}"
  export LR_SCHEDULE_STEPS="\${DENSE_LR_SCHEDULE_STEPS:-3364}"
  export WARMUP_STEPS="\${DENSE_WARMUP_STEPS:-336.4}"
  export SAVE_INTERVAL_STEPS="\${DENSE_SAVE_INTERVAL_STEPS:-256}"
  export MAX_TO_KEEP="\${DENSE_MAX_TO_KEEP:-5}"
  export EVAL_EVERY_N_STEPS="\${DENSE_EVAL_EVERY_N_STEPS:-64}"
  export NUM_GENERATIONS="\${DENSE_NUM_GENERATIONS:-2}"
  export LEARNING_RATE="\${DENSE_LEARNING_RATE:-3e-6}"
  export BETA="\${DENSE_BETA:-0.08}"
  export EPSILON="\${DENSE_EPSILON:-0.2}"
  mkdir -p "\$OBS_OUTPUT_DIR" "\$OBS_TRACE_DIR"

  env | sort | grep -E '^(RUN_ID|REWARD_MODE|MAX_STEPS|LR_SCHEDULE_STEPS|WARMUP_STEPS|SAVE_INTERVAL_STEPS|MAX_TO_KEEP|EVAL_EVERY_N_STEPS|NUM_GENERATIONS|TOTAL_GENERATION_STEPS|LEARNING_RATE|BETA|EPSILON|CKPT_DIR|TENSORBOARD_DIR|OBS_)=' > "\$exp_dir/run_env.txt"
  printf '%s\n' "\$mode" > "\$exp_dir/reward_mode.txt"

  python -u train.py 2>&1 | tee -a "\$exp_dir/train.log"

  if [[ "$TINY_SMOKE" == "1" ]]; then
    CHECKPOINT_STEPS=(1 2)
  else
    CHECKPOINT_STEPS=(256 512 768)
  fi
  python -u evaluate_checkpoints.py \\
    --ckpt-dir "\$CKPT_DIR/actor" \\
    --steps "\${CHECKPOINT_STEPS[@]}" \\
    --preset greedy \\
    --source tfds \\
    --output-dir "\$artifact_dir/checkpoint_eval" \\
    --skip-existing \\
    --continue-on-error 2>&1 | tee -a "\$exp_dir/eval_checkpoints.log" || true

  python -u analyze_grpo_run.py --run-dir "\$exp_dir" --output-dir "\$artifact_dir/analysis" || true

  if [[ "$TINY_SMOKE" != "1" ]]; then
    python - "\$artifact_dir/checkpoint_eval/checkpoint_eval_summary.json" "\$REFERENCE_ACCURACY" "\$REFERENCE_PARTIAL" "\$exp_id" "\$mode" "\$EXTENSION_SPECS" <<'PY' || true
import json, pathlib, sys
summary = pathlib.Path(sys.argv[1])
reference_acc = float(sys.argv[2])
reference_partial = float(sys.argv[3])
exp_id, mode, out_path = sys.argv[4], sys.argv[5], pathlib.Path(sys.argv[6])
if not summary.exists():
    raise SystemExit(0)
payload = json.loads(summary.read_text(encoding="utf-8"))
best = payload.get("best_lora_checkpoint") or {}
acc = float(best.get("accuracy") or 0.0)
partial = float(best.get("partial_accuracy") or 0.0)
step = int(best.get("step") or 0)
if acc > reference_acc or (abs(acc - reference_acc) < 1e-9 and partial > reference_partial):
    with out_path.open("a", encoding="utf-8") as f:
        f.write(f"{exp_id}:{mode}:{step}\n")
    print(f"Marked for 1024 extension: {exp_id} step={step} acc={acc} partial={partial}")
else:
    print(f"No 1024 extension: {exp_id} best_step={step} acc={acc} partial={partial} reference={reference_acc}")
PY
  fi
}

run_dense_extension() {
  local exp_id="\$1"
  local mode="\$2"
  local source_step="\$3"
  local source_ckpt="\$RUN_DIR/runs/\$exp_id/ckpts/actor/\$source_step"
  if [[ ! -d "\$source_ckpt" ]]; then
    echo "Extension source checkpoint missing: \$source_ckpt" >&2
    return 1
  fi
  local ext_id="E_\${exp_id}_from\${source_step}_to1024"
  local exp_dir="\$RUN_DIR/runs/\$ext_id"
  local artifact_dir="\$exp_dir/artifacts"
  mkdir -p "\$exp_dir/ckpts/actor" "\$artifact_dir" "\$exp_dir/intermediate_ckpt" "\$exp_dir/tensorboard"
  cp -a "\$source_ckpt" "\$exp_dir/ckpts/actor/\$source_step"

  echo
  echo "==> Reward dense extension \$ext_id mode=\$mode source_step=\$source_step"
  export RUN_ID="\$DENSE_ID-\$ext_id"
  export WANDB_RUN_ID="\$DENSE_ID-\$ext_id"
  export REWARD_MODE="\$mode"
  export CKPT_DIR="\$exp_dir/ckpts"
  export INTERMEDIATE_CKPT_DIR="\$exp_dir/intermediate_ckpt"
  export TENSORBOARD_DIR="\$exp_dir/tensorboard"
  export OBS_OUTPUT_DIR="\$artifact_dir/observability"
  export OBS_TRACE_DIR="\$artifact_dir/rollout_traces"
  export OBS_RUN_MANIFEST="\$artifact_dir/run_manifest.json"
  export OBS_TRACE_EVERY_N_STEPS="\${DENSE_OBS_TRACE_EVERY_N_STEPS:-1}"
  export OBS_TRACE_MAX_ROWS="\${DENSE_OBS_TRACE_MAX_ROWS:-4096}"
  export MAX_STEPS="\${DENSE_EXT_MAX_STEPS:-1024}"
  export LR_SCHEDULE_STEPS="\${DENSE_LR_SCHEDULE_STEPS:-3364}"
  export WARMUP_STEPS="\${DENSE_WARMUP_STEPS:-336.4}"
  export SAVE_INTERVAL_STEPS="\${DENSE_EXT_SAVE_INTERVAL_STEPS:-128}"
  export MAX_TO_KEEP="\${DENSE_EXT_MAX_TO_KEEP:-10}"
  export EVAL_EVERY_N_STEPS="\${DENSE_EVAL_EVERY_N_STEPS:-64}"
  export NUM_GENERATIONS="\${DENSE_NUM_GENERATIONS:-2}"
  export LEARNING_RATE="\${DENSE_LEARNING_RATE:-3e-6}"
  export BETA="\${DENSE_BETA:-0.08}"
  export EPSILON="\${DENSE_EPSILON:-0.2}"
  mkdir -p "\$OBS_OUTPUT_DIR" "\$OBS_TRACE_DIR"
  printf '%s\n' "\$mode" > "\$exp_dir/reward_mode.txt"
  env | sort | grep -E '^(RUN_ID|REWARD_MODE|MAX_STEPS|LR_SCHEDULE_STEPS|WARMUP_STEPS|SAVE_INTERVAL_STEPS|MAX_TO_KEEP|EVAL_EVERY_N_STEPS|NUM_GENERATIONS|TOTAL_GENERATION_STEPS|LEARNING_RATE|BETA|EPSILON|CKPT_DIR|TENSORBOARD_DIR|OBS_)=' > "\$exp_dir/run_env.txt"

  python -u train.py 2>&1 | tee -a "\$exp_dir/train.log"
  python -u evaluate_checkpoints.py \\
    --ckpt-dir "\$CKPT_DIR/actor" \\
    --steps auto \\
    --preset greedy \\
    --source tfds \\
    --output-dir "\$artifact_dir/checkpoint_eval" \\
    --skip-existing \\
    --continue-on-error 2>&1 | tee -a "\$exp_dir/eval_checkpoints.log" || true
  python -u analyze_grpo_run.py --run-dir "\$exp_dir" --output-dir "\$artifact_dir/analysis" || true
}

for spec in "\${DENSE_RUNS[@]}"; do
  run_dense_experiment "\${spec%%:*}" "\${spec#*:}"
done

if [[ -s "\$EXTENSION_SPECS" ]]; then
  echo
  echo "==> Conditional 1024 extensions"
  while IFS=: read -r exp_id mode source_step; do
    [[ -n "\$exp_id" ]] || continue
    run_dense_extension "\$exp_id" "\$mode" "\$source_step"
  done < "\$EXTENSION_SPECS"
else
  echo
  echo "==> No dense run met the 1024 extension rule"
fi

python - "\$PARENT_ARTIFACT_DIR/reward_dense_manifest.json" "\$RUN_DIR/runs" <<'PY'
import json, pathlib, sys
manifest_path = pathlib.Path(sys.argv[1])
runs_dir = pathlib.Path(sys.argv[2])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
runs = []
for child in sorted(runs_dir.iterdir()):
    if not child.is_dir():
        continue
    reward_mode_path = child / "reward_mode.txt"
    if not reward_mode_path.exists():
        continue
    runs.append({"run_id": child.name, "reward_mode": reward_mode_path.read_text(encoding="utf-8").strip()})
payload["runs"] = runs
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Updated dense manifest with {len(runs)} runs")
PY

echo "==> Build reward dense analysis package"
python -u analyze_reward_sweep.py --input-dir "\$RUN_DIR" --output-dir "\$PARENT_ARTIFACT_DIR/sweep_analysis" --baseline-dir "" || true
python -u build_reward_sweep_clean_plots.py --input-dir "\$RUN_DIR" --output-dir "\$PARENT_ARTIFACT_DIR/clean_plots" --rolling-window 64 || true

echo "==> Reward dense sweep complete"
EOF
  chmod +x "$run_script"
}

submit_reward_dense() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_reward_dense_script

  local session="tpu-dense-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_reward_dense.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- reward dense exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

write_r7_large_eval_script() {
  local run_script="$RUN_DIR/run_r7_large_eval.sh"
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
ARTIFACT_DIR="$ARTIFACT_DIR"
REMOTE_ROOT="$REMOTE_ROOT"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "\$VENV/bin/activate"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DATA_SOURCE=tfds
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export NUM_TEST_BATCHES="\${R7_LARGE_NUM_TEST_BATCHES:-256}"
export TOTAL_GENERATION_STEPS="\${R7_LARGE_TOTAL_GENERATION_STEPS:-768}"
export MAX_PROMPT_LENGTH="\${MAX_PROMPT_LENGTH:-256}"

OUT_DIR="\$ARTIFACT_DIR/eval"
mkdir -p "\$OUT_DIR" "\$TRAIN_DATA_DIR" "\$TEST_DATA_DIR"

R7_CKPT_ROOT="\${R7_LARGE_R7_CKPT_ROOT:-$REMOTE_ROOT/reward-dense-001/runs/R7_numeric_dense_single_answer/ckpts/actor}"
R5_CKPT_ROOT="\${R7_LARGE_R5_CKPT_ROOT:-$REMOTE_ROOT/reward-grid-001/runs/R5_numeric_primary_answer_only_len1200/ckpts/actor}"
R3_CKPT_ROOT="\${R7_LARGE_R3_CKPT_ROOT:-$REMOTE_ROOT/reward-continuation-001/runs/C2_R3_numeric_primary_no_len_from768/ckpts/actor}"

cat > "\$ARTIFACT_DIR/r7_large_eval_manifest.json" <<JSON
{
  "run_id": "$RUN_ID",
  "num_test_batches": "\${NUM_TEST_BATCHES}",
  "total_generation_steps": "\${TOTAL_GENERATION_STEPS}",
  "preset": "greedy",
  "source": "tfds",
  "evaluations": [
    {"label": "R7_step512", "step": 512, "ckpt_root": "\${R7_CKPT_ROOT}"},
    {"label": "R5_step512", "step": 512, "ckpt_root": "\${R5_CKPT_ROOT}"},
    {"label": "R3_step1408", "step": 1408, "ckpt_root": "\${R3_CKPT_ROOT}"}
  ]
}
JSON

run_one_eval() {
  local label="\$1"
  local ckpt_root="\$2"
  local step="\$3"
  local json_out="\$OUT_DIR/\${label}_large.json"
  local examples_out="\$OUT_DIR/\${label}_large_examples.jsonl"
  if [[ ! -d "\$ckpt_root/\$step" ]]; then
    echo "Missing checkpoint: \$ckpt_root/\$step" >&2
    exit 1
  fi
  echo
  echo "==> Large eval \$label ckpt_root=\$ckpt_root step=\$step"
  python -u evaluate.py \\
    --ckpt-dir "\$ckpt_root" \\
    --step "\$step" \\
    --preset greedy \\
    --source tfds \\
    --output-json "\$json_out" \\
    --output-examples-jsonl "\$examples_out"
}

echo "==> R7/R5/R3 large eval"
echo "NUM_TEST_BATCHES=\$NUM_TEST_BATCHES"
echo "R7_CKPT_ROOT=\$R7_CKPT_ROOT"
echo "R5_CKPT_ROOT=\$R5_CKPT_ROOT"
echo "R3_CKPT_ROOT=\$R3_CKPT_ROOT"

run_one_eval "R7_step512" "\$R7_CKPT_ROOT" 512
run_one_eval "R5_step512" "\$R5_CKPT_ROOT" 512
run_one_eval "R3_step1408" "\$R3_CKPT_ROOT" 1408

python - "\$OUT_DIR" <<'PY'
import csv
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
specs = [
    ("R7_step512", 512, out / "R7_step512_large.json", out / "R7_step512_large_examples.jsonl"),
    ("R5_step512", 512, out / "R5_step512_large.json", out / "R5_step512_large_examples.jsonl"),
    ("R3_step1408", 1408, out / "R3_step1408_large.json", out / "R3_step1408_large_examples.jsonl"),
]
rows = []
for label, step, path, examples_path in specs:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    rows.append({
        "label": label,
        "step": step,
        "correct": metrics.get("correct"),
        "total": metrics.get("total"),
        "accuracy": metrics.get("accuracy"),
        "partial_accuracy": metrics.get("partial_accuracy"),
        "format_accuracy": metrics.get("format_accuracy"),
        "robust_numeric_exact_rate": metrics.get("robust_numeric_exact_rate"),
        "no_close_answer_rate": metrics.get("no_close_answer_rate"),
        "text_after_close_rate": metrics.get("text_after_close_rate"),
        "failure_counts": json.dumps(metrics.get("failure_counts") or {}, sort_keys=True),
        "file": str(path),
        "examples_jsonl": str(examples_path),
    })
reference = next(row for row in rows if row["label"] == "R5_step512")
r7 = next(row for row in rows if row["label"] == "R7_step512")
summary = {
    "reference_label": "R5_step512",
    "reference_accuracy": reference["accuracy"],
    "reference_partial_accuracy": reference["partial_accuracy"],
    "r7_accuracy": r7["accuracy"],
    "r7_partial_accuracy": r7["partial_accuracy"],
    "submit_r9_recommended": (
        float(r7["accuracy"] or 0.0) >= float(reference["accuracy"] or 0.0) - 1.0
        or float(r7["partial_accuracy"] or 0.0) >= float(reference["partial_accuracy"] or 0.0)
    ),
    "rows": rows,
}
(out / "large_eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with (out / "large_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("Wrote large eval summary")
PY

echo "==> R7 large eval complete"
EOF
  chmod +x "$run_script"
}

submit_r7_large_eval() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_r7_large_eval_script

  local session="tpu-r7eval-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_r7_large_eval.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- r7 large eval exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

write_r12_best_large_eval_script() {
  local run_script="$RUN_DIR/run_r12_best_large_eval.sh"
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
ARTIFACT_DIR="$ARTIFACT_DIR"
REMOTE_ROOT="$REMOTE_ROOT"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "\$VENV/bin/activate"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DATA_SOURCE=tfds
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export NUM_TEST_BATCHES="\${R12_LARGE_NUM_TEST_BATCHES:-256}"
export TOTAL_GENERATION_STEPS="\${R12_LARGE_TOTAL_GENERATION_STEPS:-768}"
export MAX_PROMPT_LENGTH="\${MAX_PROMPT_LENGTH:-256}"

OUT_DIR="\$ARTIFACT_DIR/eval"
mkdir -p "\$OUT_DIR" "\$TRAIN_DATA_DIR" "\$TEST_DATA_DIR"

R12_CKPT_ROOT="\${R12_LARGE_CKPT_ROOT:-$REMOTE_ROOT/reward-k8-beta004-r12-full-001/runs/R12_gsm8k_verifiable_simple/ckpts/actor}"

cat > "\$ARTIFACT_DIR/r12_best_large_eval_manifest.json" <<JSON
{
  "run_id": "$RUN_ID",
  "num_test_batches": "\${NUM_TEST_BATCHES}",
  "total_generation_steps": "\${TOTAL_GENERATION_STEPS}",
  "preset": "greedy",
  "source": "tfds",
  "ckpt_root": "\${R12_CKPT_ROOT}",
  "evaluations": [
    {"label": "base", "step": null},
    {"label": "R12_step384", "step": 384},
    {"label": "R12_step512", "step": 512},
    {"label": "R12_step841", "step": 841}
  ]
}
JSON

run_base_eval() {
  echo
  echo "==> Large eval base"
  python -u evaluate.py \\
    --no-restore \\
    --preset greedy \\
    --source tfds \\
    --output-json "\$OUT_DIR/base_large.json" \\
    --output-examples-jsonl "\$OUT_DIR/base_large_examples.jsonl"
}

run_one_eval() {
  local label="\$1"
  local step="\$2"
  local json_out="\$OUT_DIR/\${label}_large.json"
  local examples_out="\$OUT_DIR/\${label}_large_examples.jsonl"
  if [[ ! -d "\$R12_CKPT_ROOT/\$step" ]]; then
    echo "Missing checkpoint: \$R12_CKPT_ROOT/\$step" >&2
    exit 1
  fi
  echo
  echo "==> Large eval \$label ckpt_root=\$R12_CKPT_ROOT step=\$step"
  python -u evaluate.py \\
    --ckpt-dir "\$R12_CKPT_ROOT" \\
    --step "\$step" \\
    --preset greedy \\
    --source tfds \\
    --output-json "\$json_out" \\
    --output-examples-jsonl "\$examples_out"
}

echo "==> R12 best large eval"
echo "NUM_TEST_BATCHES=\$NUM_TEST_BATCHES"
echo "R12_CKPT_ROOT=\$R12_CKPT_ROOT"

run_base_eval
run_one_eval "R12_step384" 384
run_one_eval "R12_step512" 512
run_one_eval "R12_step841" 841

python - "\$OUT_DIR" <<'PY'
import csv
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
specs = [
    ("base", None, out / "base_large.json", out / "base_large_examples.jsonl"),
    ("R12_step384", 384, out / "R12_step384_large.json", out / "R12_step384_large_examples.jsonl"),
    ("R12_step512", 512, out / "R12_step512_large.json", out / "R12_step512_large_examples.jsonl"),
    ("R12_step841", 841, out / "R12_step841_large.json", out / "R12_step841_large_examples.jsonl"),
]
rows = []
for label, step, path, examples_path in specs:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    rows.append({
        "label": label,
        "step": step,
        "correct": metrics.get("correct"),
        "total": metrics.get("total"),
        "accuracy": metrics.get("accuracy"),
        "partial_accuracy": metrics.get("partial_accuracy"),
        "format_accuracy": metrics.get("format_accuracy"),
        "robust_numeric_exact_rate": metrics.get("robust_numeric_exact_rate"),
        "no_close_answer_rate": metrics.get("no_close_answer_rate"),
        "text_after_close_rate": metrics.get("text_after_close_rate"),
        "failure_counts": json.dumps(metrics.get("failure_counts") or {}, sort_keys=True),
        "file": str(path),
        "examples_jsonl": str(examples_path),
    })
lora = [row for row in rows if row["step"] is not None]
best = max(lora, key=lambda row: (float(row["accuracy"] or 0.0), float(row["partial_accuracy"] or 0.0), -int(row["step"] or 0)))
summary = {
    "best_label": best["label"],
    "best_step": best["step"],
    "best_accuracy": best["accuracy"],
    "best_partial_accuracy": best["partial_accuracy"],
    "rows": rows,
}
(out / "large_eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with (out / "large_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("Wrote R12 large eval summary")
PY

echo "==> R12 best large eval complete"
EOF
  chmod +x "$run_script"
}

submit_r12_best_large_eval() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_r12_best_large_eval_script

  local session="tpu-r12eval-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_r12_best_large_eval.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- r12 large eval exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

write_reward_r9_script() {
  local run_script="$RUN_DIR/run_reward_r9.sh"
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

R9_ID="$RUN_ID"
RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
PARENT_ARTIFACT_DIR="$ARTIFACT_DIR"
REMOTE_ROOT="$REMOTE_ROOT"
PROJECT_ID="$PROJECT_ID"
STORAGE_BUCKET="$STORAGE_BUCKET"
STORAGE_PREFIX="$STORAGE_PREFIX"
STORAGE_CACHE_PREFIX="$STORAGE_CACHE_PREFIX"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "\$VENV/bin/activate"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DATA_SOURCE=tfds
export WANDB_PROJECT="\${WANDB_PROJECT:-grpo-tpu-2026}"
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export NUM_TEST_BATCHES="\${R9_NUM_TEST_BATCHES:-64}"
export TOTAL_GENERATION_STEPS="\${R9_TOTAL_GENERATION_STEPS:-768}"
export MAX_PROMPT_LENGTH="\${MAX_PROMPT_LENGTH:-256}"
mkdir -p "\$PARENT_ARTIFACT_DIR" "\$RUN_DIR/runs" "\$TRAIN_DATA_DIR" "\$TEST_DATA_DIR"

sync_on_exit() {
  local status=\$?
  if [[ -n "\$STORAGE_BUCKET" ]]; then
    echo "==> Sync R9 outputs to Cloud Storage on exit status \$status"
    bash "\$REMOTE_ROOT/_tools/remote_tpu_runner.sh" sync-storage \\
      --run-id "\$R9_ID" \\
      --remote-root "\$REMOTE_ROOT" \\
      --venv "\$VENV" \\
      --project-id "\$PROJECT_ID" \\
      --storage-bucket "\$STORAGE_BUCKET" \\
      --storage-prefix "\$STORAGE_PREFIX" \\
      --storage-cache-prefix "\$STORAGE_CACHE_PREFIX" || true
  fi
  exit "\$status"
}
trap sync_on_exit EXIT

cat > "\$PARENT_ARTIFACT_DIR/reward_r9_manifest.json" <<JSON
{
  "run_id": "$RUN_ID",
  "run_name": "R9_closed_answer_minimal",
  "reward_mode": "closed_answer_minimal",
  "max_steps": "\${R9_MAX_STEPS:-768}",
  "lr_schedule_steps": "\${R9_LR_SCHEDULE_STEPS:-3364}",
  "warmup_steps": "\${R9_WARMUP_STEPS:-336.4}",
  "save_interval_steps": "\${R9_SAVE_INTERVAL_STEPS:-256}",
  "max_to_keep": "\${R9_MAX_TO_KEEP:-5}",
  "eval_every_n_steps": "\${R9_EVAL_EVERY_N_STEPS:-64}",
  "num_generations": "\${R9_NUM_GENERATIONS:-2}",
  "total_generation_steps": "\${R9_TOTAL_GENERATION_STEPS:-768}",
  "learning_rate": "\${R9_LEARNING_RATE:-3e-6}",
  "beta": "\${R9_BETA:-0.08}",
  "epsilon": "\${R9_EPSILON:-0.2}",
  "checkpoint_eval_steps": "\${R9_CHECKPOINT_STEPS:-256 512 768}"
}
JSON

if [[ "$TINY_SMOKE" == "1" ]]; then
  export R9_MAX_STEPS=2
  export R9_SAVE_INTERVAL_STEPS=1
  export R9_MAX_TO_KEEP=5
  export R9_EVAL_EVERY_N_STEPS=1
  export R9_TOTAL_GENERATION_STEPS=96
  export NUM_TEST_BATCHES=2
  export MAX_PROMPT_LENGTH=128
fi

EXP_ID="R9_closed_answer_minimal"
EXP_DIR="\$RUN_DIR/runs/\$EXP_ID"
ARTIFACT_DIR="\$EXP_DIR/artifacts"
mkdir -p "\$EXP_DIR" "\$ARTIFACT_DIR" "\$EXP_DIR/ckpts" "\$EXP_DIR/intermediate_ckpt" "\$EXP_DIR/tensorboard"

echo
echo "==> Reward R9 run \$EXP_ID mode=closed_answer_minimal"
export RUN_ID="\$R9_ID-\$EXP_ID"
export WANDB_RUN_ID="\$R9_ID-\$EXP_ID"
export REWARD_MODE="closed_answer_minimal"
export CKPT_DIR="\$EXP_DIR/ckpts"
export INTERMEDIATE_CKPT_DIR="\$EXP_DIR/intermediate_ckpt"
export TENSORBOARD_DIR="\$EXP_DIR/tensorboard"
export OBS_OUTPUT_DIR="\$ARTIFACT_DIR/observability"
export OBS_TRACE_DIR="\$ARTIFACT_DIR/rollout_traces"
export OBS_RUN_MANIFEST="\$ARTIFACT_DIR/run_manifest.json"
export OBS_TRACE_EVERY_N_STEPS="\${R9_OBS_TRACE_EVERY_N_STEPS:-1}"
export OBS_TRACE_MAX_ROWS="\${R9_OBS_TRACE_MAX_ROWS:-4096}"
export MAX_STEPS="\${R9_MAX_STEPS:-768}"
export LR_SCHEDULE_STEPS="\${R9_LR_SCHEDULE_STEPS:-3364}"
export WARMUP_STEPS="\${R9_WARMUP_STEPS:-336.4}"
export SAVE_INTERVAL_STEPS="\${R9_SAVE_INTERVAL_STEPS:-256}"
export MAX_TO_KEEP="\${R9_MAX_TO_KEEP:-5}"
export EVAL_EVERY_N_STEPS="\${R9_EVAL_EVERY_N_STEPS:-64}"
export NUM_GENERATIONS="\${R9_NUM_GENERATIONS:-2}"
export LEARNING_RATE="\${R9_LEARNING_RATE:-3e-6}"
export BETA="\${R9_BETA:-0.08}"
export EPSILON="\${R9_EPSILON:-0.2}"
mkdir -p "\$OBS_OUTPUT_DIR" "\$OBS_TRACE_DIR"

env | sort | grep -E '^(RUN_ID|REWARD_MODE|MAX_STEPS|LR_SCHEDULE_STEPS|WARMUP_STEPS|SAVE_INTERVAL_STEPS|MAX_TO_KEEP|EVAL_EVERY_N_STEPS|NUM_GENERATIONS|TOTAL_GENERATION_STEPS|LEARNING_RATE|BETA|EPSILON|CKPT_DIR|TENSORBOARD_DIR|OBS_)=' > "\$EXP_DIR/run_env.txt"
printf '%s\n' "closed_answer_minimal" > "\$EXP_DIR/reward_mode.txt"

python -u train.py 2>&1 | tee -a "\$EXP_DIR/train.log"

if [[ "$TINY_SMOKE" == "1" ]]; then
  CHECKPOINT_STEPS=(1 2)
elif [[ -n "\${R9_CHECKPOINT_STEPS:-}" ]]; then
  read -r -a CHECKPOINT_STEPS <<< "\$R9_CHECKPOINT_STEPS"
else
  CHECKPOINT_STEPS=(256 512 768)
fi
python -u evaluate_checkpoints.py \\
  --ckpt-dir "\$CKPT_DIR/actor" \\
  --steps "\${CHECKPOINT_STEPS[@]}" \\
  --preset greedy \\
  --source tfds \\
  --output-dir "\$ARTIFACT_DIR/checkpoint_eval" \\
  --output-examples-jsonl-dir "\$ARTIFACT_DIR/checkpoint_eval_examples" \\
  --skip-existing \\
  --continue-on-error 2>&1 | tee -a "\$EXP_DIR/eval_checkpoints.log" || true

python -u analyze_grpo_run.py --run-dir "\$EXP_DIR" --output-dir "\$ARTIFACT_DIR/analysis" || true
python -u analyze_reward_sweep.py --input-dir "\$RUN_DIR" --output-dir "\$PARENT_ARTIFACT_DIR/sweep_analysis" --baseline-dir "" || true
python -u build_reward_sweep_clean_plots.py --input-dir "\$RUN_DIR" --output-dir "\$PARENT_ARTIFACT_DIR/clean_plots" --rolling-window 64 || true

echo "==> Reward R9 complete"
EOF
  chmod +x "$run_script"
}

submit_reward_r9() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_reward_r9_script

  local session="tpu-r9-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_reward_r9.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- reward r9 exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

write_reward_r10_script() {
  local run_script="$RUN_DIR/run_reward_r10.sh"
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

R10_ID="$RUN_ID"
RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
PARENT_ARTIFACT_DIR="$ARTIFACT_DIR"
REMOTE_ROOT="$REMOTE_ROOT"
PROJECT_ID="$PROJECT_ID"
STORAGE_BUCKET="$STORAGE_BUCKET"
STORAGE_PREFIX="$STORAGE_PREFIX"
STORAGE_CACHE_PREFIX="$STORAGE_CACHE_PREFIX"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "\$VENV/bin/activate"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DATA_SOURCE=tfds
export WANDB_PROJECT="\${WANDB_PROJECT:-grpo-tpu-2026}"
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export NUM_TEST_BATCHES="\${R10_NUM_TEST_BATCHES:-64}"
export TOTAL_GENERATION_STEPS="\${R10_TOTAL_GENERATION_STEPS:-768}"
export MAX_PROMPT_LENGTH="\${MAX_PROMPT_LENGTH:-256}"
mkdir -p "\$PARENT_ARTIFACT_DIR" "\$RUN_DIR/runs" "\$TRAIN_DATA_DIR" "\$TEST_DATA_DIR"

sync_on_exit() {
  local status=\$?
  if [[ -n "\$STORAGE_BUCKET" ]]; then
    echo "==> Sync R10 outputs to Cloud Storage on exit status \$status"
    bash "\$REMOTE_ROOT/_tools/remote_tpu_runner.sh" sync-storage \\
      --run-id "\$R10_ID" \\
      --remote-root "\$REMOTE_ROOT" \\
      --venv "\$VENV" \\
      --project-id "\$PROJECT_ID" \\
      --storage-bucket "\$STORAGE_BUCKET" \\
      --storage-prefix "\$STORAGE_PREFIX" \\
      --storage-cache-prefix "\$STORAGE_CACHE_PREFIX" || true
  fi
  exit "\$status"
}
trap sync_on_exit EXIT

cat > "\$PARENT_ARTIFACT_DIR/reward_r10_manifest.json" <<JSON
{
  "run_id": "$RUN_ID",
  "runs": [
    {"run_id": "R10_numeric_guarded", "reward_mode": "numeric_guarded"}
  ],
  "max_steps": "\${R10_MAX_STEPS:-1280}",
  "lr_schedule_steps": "\${R10_LR_SCHEDULE_STEPS:-3364}",
  "warmup_steps": "\${R10_WARMUP_STEPS:-336.4}",
  "save_interval_steps": "\${R10_SAVE_INTERVAL_STEPS:-128}",
  "max_to_keep": "\${R10_MAX_TO_KEEP:-16}",
  "eval_every_n_steps": "\${R10_EVAL_EVERY_N_STEPS:-64}",
  "num_generations": "\${R10_NUM_GENERATIONS:-2}",
  "total_generation_steps": "\${R10_TOTAL_GENERATION_STEPS:-768}",
  "learning_rate": "\${R10_LEARNING_RATE:-3e-6}",
  "beta": "\${R10_BETA:-0.08}",
  "epsilon": "\${R10_EPSILON:-0.2}",
  "checkpoint_eval_steps": "\${R10_CHECKPOINT_STEPS:-128 256 384 512 640 768 896 1024 1152 1280}"
}
JSON

if [[ "$TINY_SMOKE" == "1" ]]; then
  export R10_MAX_STEPS=2
  export R10_SAVE_INTERVAL_STEPS=1
  export R10_MAX_TO_KEEP=5
  export R10_EVAL_EVERY_N_STEPS=1
  export R10_TOTAL_GENERATION_STEPS=96
  export NUM_TEST_BATCHES=2
  export MAX_PROMPT_LENGTH=128
fi

EXP_ID="R10_numeric_guarded"
EXP_DIR="\$RUN_DIR/runs/\$EXP_ID"
ARTIFACT_DIR="\$EXP_DIR/artifacts"
mkdir -p "\$EXP_DIR" "\$ARTIFACT_DIR" "\$EXP_DIR/ckpts" "\$EXP_DIR/intermediate_ckpt" "\$EXP_DIR/tensorboard"

echo
echo "==> Reward R10 run \$EXP_ID mode=numeric_guarded"
export RUN_ID="\$R10_ID-\$EXP_ID"
export WANDB_RUN_ID="\$R10_ID-\$EXP_ID"
export REWARD_MODE="numeric_guarded"
export CKPT_DIR="\$EXP_DIR/ckpts"
export INTERMEDIATE_CKPT_DIR="\$EXP_DIR/intermediate_ckpt"
export TENSORBOARD_DIR="\$EXP_DIR/tensorboard"
export OBS_OUTPUT_DIR="\$ARTIFACT_DIR/observability"
export OBS_TRACE_DIR="\$ARTIFACT_DIR/rollout_traces"
export OBS_RUN_MANIFEST="\$ARTIFACT_DIR/run_manifest.json"
export OBS_TRACE_EVERY_N_STEPS="\${R10_OBS_TRACE_EVERY_N_STEPS:-1}"
export OBS_TRACE_MAX_ROWS="\${R10_OBS_TRACE_MAX_ROWS:-4096}"
export MAX_STEPS="\${R10_MAX_STEPS:-1280}"
export LR_SCHEDULE_STEPS="\${R10_LR_SCHEDULE_STEPS:-3364}"
export WARMUP_STEPS="\${R10_WARMUP_STEPS:-336.4}"
export SAVE_INTERVAL_STEPS="\${R10_SAVE_INTERVAL_STEPS:-128}"
export MAX_TO_KEEP="\${R10_MAX_TO_KEEP:-16}"
export EVAL_EVERY_N_STEPS="\${R10_EVAL_EVERY_N_STEPS:-64}"
export NUM_GENERATIONS="\${R10_NUM_GENERATIONS:-2}"
export LEARNING_RATE="\${R10_LEARNING_RATE:-3e-6}"
export BETA="\${R10_BETA:-0.08}"
export EPSILON="\${R10_EPSILON:-0.2}"
mkdir -p "\$OBS_OUTPUT_DIR" "\$OBS_TRACE_DIR"

env | sort | grep -E '^(RUN_ID|REWARD_MODE|MAX_STEPS|LR_SCHEDULE_STEPS|WARMUP_STEPS|SAVE_INTERVAL_STEPS|MAX_TO_KEEP|EVAL_EVERY_N_STEPS|NUM_GENERATIONS|TOTAL_GENERATION_STEPS|LEARNING_RATE|BETA|EPSILON|CKPT_DIR|TENSORBOARD_DIR|OBS_)=' > "\$EXP_DIR/run_env.txt"
printf '%s\n' "numeric_guarded" > "\$EXP_DIR/reward_mode.txt"

python -u train.py 2>&1 | tee -a "\$EXP_DIR/train.log"

if [[ "$TINY_SMOKE" == "1" ]]; then
  CHECKPOINT_STEPS=(1 2)
elif [[ -n "\${R10_CHECKPOINT_STEPS:-}" ]]; then
  read -r -a CHECKPOINT_STEPS <<< "\$R10_CHECKPOINT_STEPS"
else
  CHECKPOINT_STEPS=(128 256 384 512 640 768 896 1024 1152 1280)
fi
python -u evaluate_checkpoints.py \\
  --ckpt-dir "\$CKPT_DIR/actor" \\
  --steps "\${CHECKPOINT_STEPS[@]}" \\
  --preset greedy \\
  --source tfds \\
  --output-dir "\$ARTIFACT_DIR/checkpoint_eval" \\
  --output-examples-jsonl-dir "\$ARTIFACT_DIR/checkpoint_eval_examples" \\
  --skip-existing \\
  --continue-on-error 2>&1 | tee -a "\$EXP_DIR/eval_checkpoints.log" || true

python -u analyze_grpo_run.py --run-dir "\$EXP_DIR" --output-dir "\$ARTIFACT_DIR/analysis" || true
python -u analyze_reward_sweep.py --input-dir "\$RUN_DIR" --output-dir "\$PARENT_ARTIFACT_DIR/sweep_analysis" --baseline-dir "" || true
python -u build_reward_sweep_clean_plots.py --input-dir "\$RUN_DIR" --output-dir "\$PARENT_ARTIFACT_DIR/clean_plots" --rolling-window 64 || true

echo "==> Reward R10 complete"
EOF
  chmod +x "$run_script"
}

submit_reward_r10() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_reward_r10_script

  local session="tpu-r10-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_reward_r10.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- reward r10 exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

write_k8_pilot_script() {
  local run_script="$RUN_DIR/run_k8_pilot.sh"
  local run_specs="${1:-R9_closed_answer_minimal:closed_answer_minimal R10_numeric_guarded:numeric_guarded}"
  local manifest_runs=""
  local manifest_sep=""
  local spec exp_id mode spec_beta spec_lr spec_rank spec_alpha
  for spec in $run_specs; do
    IFS=':' read -r exp_id mode spec_beta spec_lr spec_rank spec_alpha <<< "$spec"
    manifest_runs+="${manifest_sep}    {\"run_id\": \"$exp_id\", \"reward_mode\": \"$mode\", \"beta_override\": \"${spec_beta:-}\", \"learning_rate_override\": \"${spec_lr:-}\", \"rank_override\": \"${spec_rank:-}\", \"alpha_override\": \"${spec_alpha:-}\"}"
    manifest_sep=$',\n'
  done
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

K8_ID="$RUN_ID"
K8_RUN_SPECS="$run_specs"
RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
PARENT_ARTIFACT_DIR="$ARTIFACT_DIR"
REMOTE_ROOT="$REMOTE_ROOT"
PROJECT_ID="$PROJECT_ID"
STORAGE_BUCKET="$STORAGE_BUCKET"
STORAGE_PREFIX="$STORAGE_PREFIX"
STORAGE_CACHE_PREFIX="$STORAGE_CACHE_PREFIX"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "\$VENV/bin/activate"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DATA_SOURCE=tfds
export WANDB_PROJECT="\${WANDB_PROJECT:-grpo-tpu-2026}"
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export NUM_TEST_BATCHES="\${K8_NUM_TEST_BATCHES:-64}"
export TOTAL_GENERATION_STEPS="\${K8_TOTAL_GENERATION_STEPS:-768}"
export MAX_PROMPT_LENGTH="\${MAX_PROMPT_LENGTH:-256}"
mkdir -p "\$PARENT_ARTIFACT_DIR" "\$RUN_DIR/runs" "\$TRAIN_DATA_DIR" "\$TEST_DATA_DIR"

sync_on_exit() {
  local status=\$?
  if [[ -n "\$STORAGE_BUCKET" ]]; then
    echo "==> Sync K8 outputs to Cloud Storage on exit status \$status"
    bash "\$REMOTE_ROOT/_tools/remote_tpu_runner.sh" sync-storage \\
      --run-id "\$K8_ID" \\
      --remote-root "\$REMOTE_ROOT" \\
      --venv "\$VENV" \\
      --project-id "\$PROJECT_ID" \\
      --storage-bucket "\$STORAGE_BUCKET" \\
      --storage-prefix "\$STORAGE_PREFIX" \\
      --storage-cache-prefix "\$STORAGE_CACHE_PREFIX" || true
  fi
  exit "\$status"
}
trap sync_on_exit EXIT

cat > "\$PARENT_ARTIFACT_DIR/reward_k8_pilot_manifest.json" <<JSON
{
  "run_id": "$RUN_ID",
  "runs": [
$manifest_runs
  ],
  "max_steps": "\${K8_MAX_STEPS:-256}",
  "lr_schedule_steps": "\${K8_LR_SCHEDULE_STEPS:-841}",
  "warmup_steps": "\${K8_WARMUP_STEPS:-84.1}",
  "save_interval_steps": "\${K8_SAVE_INTERVAL_STEPS:-32}",
  "max_to_keep": "\${K8_MAX_TO_KEEP:-12}",
  "eval_every_n_steps": "\${K8_EVAL_EVERY_N_STEPS:-32}",
  "num_generations": "\${K8_NUM_GENERATIONS:-8}",
  "total_generation_steps": "\${K8_TOTAL_GENERATION_STEPS:-768}",
  "learning_rate": "\${K8_LEARNING_RATE:-3e-6}",
  "beta": "\${K8_BETA:-0.04}",
  "epsilon": "\${K8_EPSILON:-0.2}",
  "rank": "\${K8_RANK:-64}",
  "alpha": "\${K8_ALPHA:-64}",
  "checkpoint_eval_steps": "\${K8_CHECKPOINT_STEPS:-32 64 96 128 160 192 224 256}",
  "run_spec_format": "run_id:reward_mode[:beta_override[:learning_rate_override[:rank_override[:alpha_override]]]]"
}
JSON

if [[ "$TINY_SMOKE" == "1" ]]; then
  export K8_MAX_STEPS=2
  export K8_SAVE_INTERVAL_STEPS=1
  export K8_MAX_TO_KEEP=5
  export K8_EVAL_EVERY_N_STEPS=1
  export K8_TOTAL_GENERATION_STEPS=96
  export NUM_TEST_BATCHES=2
  export MAX_PROMPT_LENGTH=128
fi

read -r -a K8_RUNS <<< "\$K8_RUN_SPECS"

for spec in "\${K8_RUNS[@]}"; do
  IFS=':' read -r EXP_ID MODE SPEC_BETA SPEC_LEARNING_RATE SPEC_RANK SPEC_ALPHA <<< "\$spec"
  BRANCH_BETA="\${SPEC_BETA:-\${K8_BETA:-0.04}}"
  BRANCH_LEARNING_RATE="\${SPEC_LEARNING_RATE:-\${K8_LEARNING_RATE:-3e-6}}"
  BRANCH_RANK="\${SPEC_RANK:-\${K8_RANK:-64}}"
  BRANCH_ALPHA="\${SPEC_ALPHA:-\${K8_ALPHA:-64}}"
  EXP_DIR="\$RUN_DIR/runs/\$EXP_ID"
  ARTIFACT_DIR="\$EXP_DIR/artifacts"
  mkdir -p "\$EXP_DIR" "\$ARTIFACT_DIR" "\$EXP_DIR/ckpts" "\$EXP_DIR/intermediate_ckpt" "\$EXP_DIR/tensorboard"

  echo
  echo "==> K8 pilot run \$EXP_ID mode=\$MODE beta=\$BRANCH_BETA lr=\$BRANCH_LEARNING_RATE rank=\$BRANCH_RANK alpha=\$BRANCH_ALPHA"
  export RUN_ID="\$K8_ID-\$EXP_ID"
  export WANDB_RUN_ID="\$K8_ID-\$EXP_ID"
  export REWARD_MODE="\$MODE"
  export CKPT_DIR="\$EXP_DIR/ckpts"
  export INTERMEDIATE_CKPT_DIR="\$EXP_DIR/intermediate_ckpt"
  export TENSORBOARD_DIR="\$EXP_DIR/tensorboard"
  export OBS_OUTPUT_DIR="\$ARTIFACT_DIR/observability"
  export OBS_TRACE_DIR="\$ARTIFACT_DIR/rollout_traces"
  export OBS_RUN_MANIFEST="\$ARTIFACT_DIR/run_manifest.json"
  export OBS_TRACE_EVERY_N_STEPS="\${K8_OBS_TRACE_EVERY_N_STEPS:-1}"
  export OBS_TRACE_MAX_ROWS="\${K8_OBS_TRACE_MAX_ROWS:-8192}"
  export MAX_STEPS="\${K8_MAX_STEPS:-256}"
  export LR_SCHEDULE_STEPS="\${K8_LR_SCHEDULE_STEPS:-841}"
  export WARMUP_STEPS="\${K8_WARMUP_STEPS:-84.1}"
  export SAVE_INTERVAL_STEPS="\${K8_SAVE_INTERVAL_STEPS:-32}"
  export MAX_TO_KEEP="\${K8_MAX_TO_KEEP:-12}"
  export EVAL_EVERY_N_STEPS="\${K8_EVAL_EVERY_N_STEPS:-32}"
  export NUM_GENERATIONS="\${K8_NUM_GENERATIONS:-8}"
  export LEARNING_RATE="\$BRANCH_LEARNING_RATE"
  export BETA="\$BRANCH_BETA"
  export EPSILON="\${K8_EPSILON:-0.2}"
  export RANK="\$BRANCH_RANK"
  export ALPHA="\$BRANCH_ALPHA"
  mkdir -p "\$OBS_OUTPUT_DIR" "\$OBS_TRACE_DIR"

  env | sort | grep -E '^(RUN_ID|REWARD_MODE|MAX_STEPS|LR_SCHEDULE_STEPS|WARMUP_STEPS|SAVE_INTERVAL_STEPS|MAX_TO_KEEP|EVAL_EVERY_N_STEPS|NUM_GENERATIONS|TOTAL_GENERATION_STEPS|LEARNING_RATE|BETA|EPSILON|RANK|ALPHA|CKPT_DIR|TENSORBOARD_DIR|OBS_)=' > "\$EXP_DIR/run_env.txt"
  printf '%s\n' "\$MODE" > "\$EXP_DIR/reward_mode.txt"

  python -u train.py 2>&1 | tee -a "\$EXP_DIR/train.log"

  if [[ "$TINY_SMOKE" == "1" ]]; then
    CHECKPOINT_STEPS=(1 2)
  elif [[ -n "\${K8_CHECKPOINT_STEPS:-}" ]]; then
    read -r -a CHECKPOINT_STEPS <<< "\$K8_CHECKPOINT_STEPS"
  else
    CHECKPOINT_STEPS=(32 64 96 128 160 192 224 256)
  fi
  python -u evaluate_checkpoints.py \\
    --ckpt-dir "\$CKPT_DIR/actor" \\
    --steps "\${CHECKPOINT_STEPS[@]}" \\
    --preset greedy \\
    --source tfds \\
    --output-dir "\$ARTIFACT_DIR/checkpoint_eval" \\
    --output-examples-jsonl-dir "\$ARTIFACT_DIR/checkpoint_eval_examples" \\
    --skip-existing \\
    --continue-on-error 2>&1 | tee -a "\$EXP_DIR/eval_checkpoints.log" || true

  python -u analyze_grpo_run.py --run-dir "\$EXP_DIR" --output-dir "\$ARTIFACT_DIR/analysis" || true
done

python -u analyze_reward_sweep.py --input-dir "\$RUN_DIR" --output-dir "\$PARENT_ARTIFACT_DIR/sweep_analysis" --baseline-dir "" || true
python -u build_reward_sweep_clean_plots.py --input-dir "\$RUN_DIR" --output-dir "\$PARENT_ARTIFACT_DIR/clean_plots" --rolling-window 64 || true

echo "==> K8 pilot complete"
EOF
  chmod +x "$run_script"
}

submit_k8_pilot() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 pilot exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_k8_r10_only() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script "R10_numeric_guarded:numeric_guarded"

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 pilot exited (\$status) ---\"; exec bash"
  echo "Started R10-only K8 pilot. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_k8_r11_fallback_only() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script "R11_numeric_guarded_fallback:numeric_guarded_fallback"

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "K8_MAX_STEPS=128 K8_CHECKPOINT_STEPS='32 64 96 128' K8_MAX_TO_KEEP=8 K8_SAVE_INTERVAL_STEPS=32 K8_EVAL_EVERY_N_STEPS=32 bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 pilot exited (\$status) ---\"; exec bash"
  echo "Started R11 fallback-only K8 pilot. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_k8_r12_simple_only() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script "R12_gsm8k_verifiable_simple:gsm8k_verifiable_simple"

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "K8_MAX_STEPS=256 K8_CHECKPOINT_STEPS='32 64 96 128 160 192 224 256' K8_MAX_TO_KEEP=12 K8_SAVE_INTERVAL_STEPS=32 K8_EVAL_EVERY_N_STEPS=32 bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 pilot exited (\$status) ---\"; exec bash"
  echo "Started R12 simple-verifiable K8 pilot. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_k8_r12_simple_full() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script "R12_gsm8k_verifiable_simple:gsm8k_verifiable_simple"

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "K8_MAX_STEPS=841 K8_LR_SCHEDULE_STEPS=841 K8_WARMUP_STEPS=84.1 K8_CHECKPOINT_STEPS='128 256 384 512 640 768 841' K8_MAX_TO_KEEP=16 K8_SAVE_INTERVAL_STEPS=128 K8_EVAL_EVERY_N_STEPS=64 bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 pilot exited (\$status) ---\"; exec bash"
  echo "Started R12 simple-verifiable K8 equivalent-full run. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_reward_only_r12_full() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script "R12_reward_only_baseline_kkl:gsm8k_verifiable_simple"

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "K8_MAX_STEPS=3364 K8_LR_SCHEDULE_STEPS=3364 K8_WARMUP_STEPS=336.4 K8_CHECKPOINT_STEPS='500 1000 1500 2000 2500 3000 3364' K8_MAX_TO_KEEP=16 K8_SAVE_INTERVAL_STEPS=500 K8_EVAL_EVERY_N_STEPS=64 K8_NUM_GENERATIONS=2 K8_BETA=0.08 K8_LEARNING_RATE=3e-6 bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 pilot exited (\$status) ---\"; exec bash"
  echo "Started R12 reward-only baseline-K/KL full run. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_r12_non_r64_pilot() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script "R12_rank16_alpha32_beta004_lr3e-6:gsm8k_verifiable_simple:0.04:3e-6:16:32"

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "K8_MAX_STEPS=256 K8_CHECKPOINT_STEPS='32 64 96 128 160 192 224 256' K8_MAX_TO_KEEP=12 K8_SAVE_INTERVAL_STEPS=32 K8_EVAL_EVERY_N_STEPS=32 bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 non-r64 pilot exited (\$status) ---\"; exec bash"
  echo "Started R12 non-R64 K8 pilot. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_r12_lora_public_tuning() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script "R12_rank32_alpha32_beta004_lr3e-6:gsm8k_verifiable_simple:0.04:3e-6:32:32 R12_rank16_alpha32_beta001_lr1e-6:gsm8k_verifiable_simple:0.001:1e-6:16:32 R12_rank16_alpha32_beta000_lr1e-6:gsm8k_verifiable_simple:0.0:1e-6:16:32"

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "K8_MAX_STEPS=256 K8_CHECKPOINT_STEPS='32 64 96 128 160 192 224 256' K8_MAX_TO_KEEP=12 K8_SAVE_INTERVAL_STEPS=32 K8_EVAL_EVERY_N_STEPS=32 bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 lora tuning pilot exited (\$status) ---\"; exec bash"
  echo "Started R12 LoRA/LR public tuning pilot. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_k8_public_beta() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script "R13_gsm8k_simple_beta0001:gsm8k_verifiable_simple:0.001:1e-6 R14_gsm8k_simple_beta000:gsm8k_verifiable_simple:0.0:1e-6"

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "K8_MAX_STEPS=256 K8_CHECKPOINT_STEPS='32 64 96 128 160 192 224 256' K8_MAX_TO_KEEP=12 K8_SAVE_INTERVAL_STEPS=32 K8_EVAL_EVERY_N_STEPS=32 bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 pilot exited (\$status) ---\"; exec bash"
  echo "Started R13/R14 public-beta K8 pilot. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_k8_r13_public_beta_only() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script "R13_gsm8k_simple_beta0001:gsm8k_verifiable_simple:0.001:1e-6"

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "K8_MAX_STEPS=256 K8_CHECKPOINT_STEPS='32 64 96 128 160 192 224 256' K8_MAX_TO_KEEP=12 K8_SAVE_INTERVAL_STEPS=32 K8_EVAL_EVERY_N_STEPS=32 bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 pilot exited (\$status) ---\"; exec bash"
  echo "Started R13 public-beta K8 pilot. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

submit_k8_r14_public_beta_only() {
  require_run_id
  unpack_bundle
  install_secrets
  bootstrap_env
  check_tpu_backend
  write_k8_pilot_script "R14_gsm8k_simple_beta000:gsm8k_verifiable_simple:0.0:1e-6"

  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "K8_MAX_STEPS=256 K8_CHECKPOINT_STEPS='32 64 96 128 160 192 224 256' K8_MAX_TO_KEEP=12 K8_SAVE_INTERVAL_STEPS=32 K8_EVAL_EVERY_N_STEPS=32 bash '$RUN_DIR/run_k8_pilot.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 pilot exited (\$status) ---\"; exec bash"
  echo "Started R14 public-beta K8 pilot. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

write_eval_checkpoints_script() {
  local run_script="$RUN_DIR/run_eval_checkpoints.sh"
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$RUN_DIR"
SRC_DIR="$SRC_DIR"
VENV="$REMOTE_VENV"
ARTIFACT_DIR="$ARTIFACT_DIR"
REMOTE_ROOT="$REMOTE_ROOT"
PROJECT_ID="$PROJECT_ID"
STORAGE_BUCKET="$STORAGE_BUCKET"
STORAGE_PREFIX="$STORAGE_PREFIX"
STORAGE_CACHE_PREFIX="$STORAGE_CACHE_PREFIX"

cd "\$SRC_DIR/scripts"
set -a
if [[ -f .env ]]; then
  source .env
fi
set +a
source "\$VENV/bin/activate"

export RUN_ID="$RUN_ID"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CKPT_DIR="\$RUN_DIR/ckpts"
export INTERMEDIATE_CKPT_DIR="\$RUN_DIR/intermediate_ckpt"
export TENSORBOARD_DIR="\$RUN_DIR/tensorboard"
export TRAIN_DATA_DIR="\$RUN_DIR/data/train"
export TEST_DATA_DIR="\$RUN_DIR/data/test"
export WANDB_RUN_ID="\${WANDB_RUN_ID:-$RUN_ID-checkpoint-eval}"
export WANDB_PROJECT="\${WANDB_PROJECT:-grpo-tpu-2026}"
export OBS_OUTPUT_DIR="\$ARTIFACT_DIR/observability"
export OBS_TRACE_DIR="\$ARTIFACT_DIR/rollout_traces"
export OBS_RUN_MANIFEST="\$ARTIFACT_DIR/run_manifest.json"
mkdir -p "\$ARTIFACT_DIR/checkpoint_eval" "\$OBS_OUTPUT_DIR" "\$OBS_TRACE_DIR"

read -r -a CHECKPOINT_STEPS <<< "\${CHECKPOINT_EVAL_STEPS:-auto}"

echo "==> Checkpoint-wise evaluation only"
python -u evaluate_checkpoints.py \\
  --ckpt-dir "\$CKPT_DIR/actor" \\
  --steps "\${CHECKPOINT_STEPS[@]}" \\
  --include-base \\
  --preset "\${CHECKPOINT_EVAL_PRESET:-greedy}" \\
  --output-dir "\$ARTIFACT_DIR/checkpoint_eval" \\
  --skip-existing

echo "==> Export GRPO diagnostics"
python -u analyze_grpo_run.py --run-dir "\$RUN_DIR" --output-dir "\$ARTIFACT_DIR/analysis" || true

if [[ -n "\$STORAGE_BUCKET" ]]; then
  echo "==> Sync checkpoint eval outputs to Cloud Storage"
  bash "\$REMOTE_ROOT/_tools/remote_tpu_runner.sh" sync-storage \\
    --run-id "\$RUN_ID" \\
    --remote-root "\$REMOTE_ROOT" \\
    --venv "\$VENV" \\
    --project-id "\$PROJECT_ID" \\
    --storage-bucket "\$STORAGE_BUCKET" \\
    --storage-prefix "\$STORAGE_PREFIX" \\
    --storage-cache-prefix "\$STORAGE_CACHE_PREFIX" || true
fi

echo "==> Checkpoint eval pipeline complete"
EOF
  chmod +x "$run_script"
}

eval_checkpoints() {
  require_run_id
  if [[ -n "$BUNDLE" ]]; then
    unpack_bundle
    install_secrets
    bootstrap_env
  else
    prepare_paths
  fi
  check_tpu_backend
  if [[ -n "$PROJECT_ID" && -n "$STORAGE_BUCKET" ]]; then
    restore_run_from_storage || true
  fi
  write_eval_checkpoints_script

  local session="tpu-eval-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi

  echo "==> Starting tmux session $session"
  tmux new-session -d -s "$session" "bash '$RUN_DIR/run_eval_checkpoints.sh' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- checkpoint eval exited (\$status) ---\"; exec bash"
  echo "Started. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

status_run() {
  require_run_id
  prepare_paths
  local session="tpu-${RUN_ID//./-}"
  local eval_session="tpu-eval-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux: running ($session)"
  else
    echo "tmux: not running ($session)"
  fi
  if tmux has-session -t "$eval_session" 2>/dev/null; then
    echo "tmux eval: running ($eval_session)"
  else
    echo "tmux eval: not running ($eval_session)"
  fi
  echo
  echo "Run directory: $RUN_DIR"
  echo
  if [[ -f "$RUN_DIR/pipeline.log" ]]; then
    echo "Pipeline markers:"
    grep -nE '^(==>|--- pipeline exited|--- checkpoint eval exited)' "$RUN_DIR/pipeline.log" | tail -n 20 || true
    echo
    echo "Last 80 log lines:"
    tail -n 80 "$RUN_DIR/pipeline.log"
  else
    echo "No pipeline.log found yet."
  fi
  echo
  echo "Checkpoints:"
  find "$RUN_DIR/ckpts" "$RUN_DIR/intermediate_ckpt" -maxdepth 4 -type f 2>/dev/null | sort | tail -n 40 || true
  echo
  echo "TensorBoard latest scalars:"
  if compgen -G "$RUN_DIR/tensorboard/events.out.tfevents*" >/dev/null; then
    (
      source "$REMOTE_VENV/bin/activate"
      python - "$RUN_DIR/tensorboard" <<'PY'
import pathlib
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:
    print(f"tensorboard read failed: {exc}")
    raise SystemExit(0)

tb_dir = pathlib.Path(sys.argv[1])
seen = False
for event_file in sorted(tb_dir.glob("events.out.tfevents*")):
    acc = EventAccumulator(str(event_file))
    acc.Reload()
    tags = sorted(acc.Tags().get("scalars", []))
    if not tags:
        continue
    seen = True
    print(event_file.name)
    for tag in tags:
        values = acc.Scalars(tag)
        if values:
            latest = values[-1]
            print(f"  {tag}: count={len(values)} latest_step={latest.step} latest_value={latest.value}")
if not seen:
    print("no scalar tags found yet")
PY
    ) || true
  else
    echo "No TensorBoard event files found yet."
  fi
  echo
  echo "Artifacts:"
  find "$ARTIFACT_DIR" -maxdepth 3 -type f 2>/dev/null | sort || true
}

status_sweep() {
  require_run_id
  prepare_paths
  local session="tpu-sweep-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux sweep: running ($session)"
  else
    echo "tmux sweep: not running ($session)"
  fi
  echo
  echo "Sweep directory: $RUN_DIR"
  echo
  if [[ -f "$RUN_DIR/pipeline.log" ]]; then
    echo "Sweep markers:"
    grep -nE '^(==>|--- reward sweep exited)' "$RUN_DIR/pipeline.log" | tail -n 40 || true
    echo
    echo "Last 100 log lines:"
    tail -n 100 "$RUN_DIR/pipeline.log"
  else
    echo "No pipeline.log found yet."
  fi
  echo
  echo "Per-run summaries:"
  if [[ -d "$RUN_DIR/runs" ]]; then
    for child in "$RUN_DIR"/runs/*; do
      [[ -d "$child" ]] || continue
      echo
      echo "--- $(basename "$child") ---"
      if [[ -f "$child/reward_mode.txt" ]]; then
        echo "reward_mode: $(cat "$child/reward_mode.txt")"
      fi
      find "$child/ckpts/actor" -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sed 's#.*/##' | sort -n | tail -n 10 | xargs -r echo "checkpoints:"
      if [[ -f "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" ]]; then
        python - "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" <<'PY' || true
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
best = payload.get("best_lora_checkpoint") or {}
rows = payload.get("rows") or []
print(f"eval_rows: {len(rows)}")
if best:
    print(f"best_eval: step={best.get('step')} accuracy={best.get('accuracy')} partial={best.get('partial_accuracy')} format={best.get('format_accuracy')}")
failures = payload.get("failures") or []
if failures:
    print(f"failures: {failures}")
PY
      fi
    done
  else
    echo "No runs/ directory found yet."
  fi
  echo
  echo "Sweep artifacts:"
  find "$ARTIFACT_DIR" -maxdepth 5 -type f 2>/dev/null | sort | tail -n 80 || true
}

status_continuation() {
  require_run_id
  prepare_paths
  local session="tpu-cont-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux continuation: running ($session)"
  else
    echo "tmux continuation: not running ($session)"
  fi
  echo
  echo "Continuation directory: $RUN_DIR"
  echo
  if [[ -f "$RUN_DIR/pipeline.log" ]]; then
    echo "Continuation markers:"
    grep -nE '^(==>|--- reward continuation exited)' "$RUN_DIR/pipeline.log" | tail -n 60 || true
    echo
    echo "Last 100 log lines:"
    tail -n 100 "$RUN_DIR/pipeline.log"
  else
    echo "No pipeline.log found yet."
  fi
  echo
  echo "Per-run summaries:"
  if [[ -d "$RUN_DIR/runs" ]]; then
    for child in "$RUN_DIR"/runs/*; do
      [[ -d "$child" ]] || continue
      echo
      echo "--- $(basename "$child") ---"
      if [[ -f "$child/reward_mode.txt" ]]; then
        echo "reward_mode: $(cat "$child/reward_mode.txt")"
      fi
      if [[ -f "$child/branch_metadata.json" ]]; then
        python - "$child/branch_metadata.json" <<'PY' || true
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(f"source: {payload.get('source_run')}@{payload.get('source_step')}")
PY
      fi
      find "$child/ckpts/actor" -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sed 's#.*/##' | sort -n | tail -n 20 | xargs -r echo "checkpoints:"
      if [[ -f "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" ]]; then
        python - "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" <<'PY' || true
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
best = payload.get("best_lora_checkpoint") or {}
rows = payload.get("rows") or []
print(f"eval_rows: {len(rows)}")
if best:
    print(f"best_eval: step={best.get('step')} accuracy={best.get('accuracy')} partial={best.get('partial_accuracy')} format={best.get('format_accuracy')}")
failures = payload.get("failures") or []
if failures:
    print(f"failures: {failures}")
PY
      fi
    done
  else
    echo "No runs/ directory found yet."
  fi
  echo
  echo "Continuation artifacts:"
  find "$ARTIFACT_DIR" -maxdepth 5 -type f 2>/dev/null | sort | tail -n 100 || true
}

status_candidate_eval() {
  require_run_id
  prepare_paths
  local session="tpu-candidate-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux candidate eval: running ($session)"
  else
    echo "tmux candidate eval: not running ($session)"
  fi
  echo
  echo "Candidate eval directory: $RUN_DIR"
  echo
  if [[ -f "$RUN_DIR/pipeline.log" ]]; then
    echo "Candidate eval markers:"
    grep -nE '^(==>|--- candidate eval exited)' "$RUN_DIR/pipeline.log" | tail -n 40 || true
    echo
    echo "Last 100 log lines:"
    tail -n 100 "$RUN_DIR/pipeline.log"
  else
    echo "No pipeline.log found yet."
  fi
  echo
  echo "Candidate eval summary:"
  if [[ -f "$ARTIFACT_DIR/candidate_eval/candidate_eval_summary.json" ]]; then
    python - "$ARTIFACT_DIR/candidate_eval/candidate_eval_summary.json" <<'PY' || true
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
for row in payload.get("rows") or []:
    print(f"{row.get('label')}: acc={row.get('accuracy')} partial={row.get('partial_accuracy')} format={row.get('format_accuracy')} total={row.get('total')}")
print(f"reference_accuracy={payload.get('reference_accuracy')}")
PY
  else
    echo "No candidate_eval_summary.json yet."
  fi
  echo
  echo "Candidate artifacts:"
  find "$ARTIFACT_DIR" -maxdepth 5 -type f 2>/dev/null | sort | tail -n 100 || true
}

status_reward_dense() {
  require_run_id
  prepare_paths
  local session="tpu-dense-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux reward dense: running ($session)"
  else
    echo "tmux reward dense: not running ($session)"
  fi
  echo
  echo "Reward dense directory: $RUN_DIR"
  echo
  if [[ -f "$RUN_DIR/pipeline.log" ]]; then
    echo "Reward dense markers:"
    grep -nE '^(==>|--- reward dense exited)' "$RUN_DIR/pipeline.log" | tail -n 80 || true
    echo
    echo "Last 100 log lines:"
    tail -n 100 "$RUN_DIR/pipeline.log"
  else
    echo "No pipeline.log found yet."
  fi
  echo
  echo "Per-run summaries:"
  if [[ -d "$RUN_DIR/runs" ]]; then
    for child in "$RUN_DIR"/runs/*; do
      [[ -d "$child" ]] || continue
      echo
      echo "--- $(basename "$child") ---"
      if [[ -f "$child/reward_mode.txt" ]]; then
        echo "reward_mode: $(cat "$child/reward_mode.txt")"
      fi
      find "$child/ckpts/actor" -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sed 's#.*/##' | sort -n | tail -n 20 | xargs -r echo "checkpoints:"
      if [[ -f "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" ]]; then
        python - "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" <<'PY' || true
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
best = payload.get("best_lora_checkpoint") or {}
rows = payload.get("rows") or []
print(f"eval_rows: {len(rows)}")
if best:
    print(f"best_eval: step={best.get('step')} accuracy={best.get('accuracy')} partial={best.get('partial_accuracy')} format={best.get('format_accuracy')}")
failures = payload.get("failures") or []
if failures:
    print(f"failures: {failures}")
PY
      fi
    done
  else
    echo "No runs/ directory found yet."
  fi
  echo
  echo "Dense artifacts:"
  find "$ARTIFACT_DIR" -maxdepth 5 -type f 2>/dev/null | sort | tail -n 100 || true
}

status_r7_large_eval() {
  require_run_id
  prepare_paths
  local session="tpu-r7eval-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux r7 large eval: running ($session)"
  else
    echo "tmux r7 large eval: not running ($session)"
  fi
  echo
  echo "R7 large eval directory: $RUN_DIR"
  echo
  if [[ -f "$RUN_DIR/pipeline.log" ]]; then
    echo "R7 large eval markers:"
    grep -nE '^(==>|--- r7 large eval exited)' "$RUN_DIR/pipeline.log" | tail -n 60 || true
    echo
    echo "Last 100 log lines:"
    tail -n 100 "$RUN_DIR/pipeline.log"
  else
    echo "No pipeline.log found yet."
  fi
  echo
  echo "Large eval summary:"
  if [[ -f "$ARTIFACT_DIR/eval/large_eval_summary.json" ]]; then
    python - "$ARTIFACT_DIR/eval/large_eval_summary.json" <<'PY' || true
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
for row in payload.get("rows") or []:
    print(
        f"{row.get('label')}: acc={row.get('accuracy')} partial={row.get('partial_accuracy')} "
        f"format={row.get('format_accuracy')} robust_exact={row.get('robust_numeric_exact_rate')} total={row.get('total')}"
    )
print(f"reference_accuracy={payload.get('reference_accuracy')}")
print(f"submit_r9_recommended={payload.get('submit_r9_recommended')}")
PY
  else
    echo "No large_eval_summary.json yet."
  fi
  echo
  echo "R7 large eval artifacts:"
  find "$ARTIFACT_DIR" -maxdepth 5 -type f 2>/dev/null | sort | tail -n 100 || true
}

status_reward_r9() {
  require_run_id
  prepare_paths
  local session="tpu-r9-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux reward r9: running ($session)"
  else
    echo "tmux reward r9: not running ($session)"
  fi
  echo
  echo "Reward R9 directory: $RUN_DIR"
  echo
  if [[ -f "$RUN_DIR/pipeline.log" ]]; then
    echo "Reward R9 markers:"
    grep -nE '^(==>|--- reward r9 exited)' "$RUN_DIR/pipeline.log" | tail -n 80 || true
    echo
    echo "Last 100 log lines:"
    tail -n 100 "$RUN_DIR/pipeline.log"
  else
    echo "No pipeline.log found yet."
  fi
  echo
  echo "Per-run summaries:"
  if [[ -d "$RUN_DIR/runs" ]]; then
    for child in "$RUN_DIR"/runs/*; do
      [[ -d "$child" ]] || continue
      echo
      echo "--- $(basename "$child") ---"
      if [[ -f "$child/reward_mode.txt" ]]; then
        echo "reward_mode: $(cat "$child/reward_mode.txt")"
      fi
      find "$child/ckpts/actor" -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sed 's#.*/##' | sort -n | tail -n 20 | xargs -r echo "checkpoints:"
      if [[ -f "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" ]]; then
        python - "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" <<'PY' || true
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
best = payload.get("best_lora_checkpoint") or {}
rows = payload.get("rows") or []
print(f"eval_rows: {len(rows)}")
if best:
    print(
        f"best_eval: step={best.get('step')} accuracy={best.get('accuracy')} "
        f"partial={best.get('partial_accuracy')} format={best.get('format_accuracy')} "
        f"robust_exact={best.get('robust_numeric_exact_rate')}"
    )
failures = payload.get("failures") or []
if failures:
    print(f"failures: {failures}")
PY
      fi
    done
  else
    echo "No runs/ directory found yet."
  fi
  echo
  echo "R9 artifacts:"
  find "$ARTIFACT_DIR" -maxdepth 5 -type f 2>/dev/null | sort | tail -n 100 || true
}

status_reward_r10() {
  require_run_id
  prepare_paths
  local session="tpu-r10-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux reward r10: running ($session)"
  else
    echo "tmux reward r10: not running ($session)"
  fi
  echo
  echo "Reward R10 directory: $RUN_DIR"
  echo
  if [[ -f "$RUN_DIR/pipeline.log" ]]; then
    echo "Reward R10 markers:"
    grep -nE '^(==>|--- reward r10 exited)' "$RUN_DIR/pipeline.log" | tail -n 80 || true
    echo
    echo "Last 100 log lines:"
    tail -n 100 "$RUN_DIR/pipeline.log"
  else
    echo "No pipeline.log found yet."
  fi
  echo
  echo "Per-run summaries:"
  if [[ -d "$RUN_DIR/runs" ]]; then
    for child in "$RUN_DIR"/runs/*; do
      [[ -d "$child" ]] || continue
      echo
      echo "--- $(basename "$child") ---"
      if [[ -f "$child/reward_mode.txt" ]]; then
        echo "reward_mode: $(cat "$child/reward_mode.txt")"
      fi
      find "$child/ckpts/actor" -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sed 's#.*/##' | sort -n | tail -n 30 | xargs -r echo "checkpoints:"
      local trace_file
      trace_file="$(find "$child/artifacts/rollout_traces" -maxdepth 1 -type f -name '*.jsonl' 2>/dev/null | sort | tail -n 1 || true)"
      if [[ -n "$trace_file" && -f "$trace_file" ]]; then
        python - "$trace_file" <<'PY' || true
import collections
import json
import pathlib
import statistics
import sys

path = pathlib.Path(sys.argv[1])
rows = []
for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-512:]:
    if line.strip():
        rows.append(json.loads(line))
if not rows:
    raise SystemExit
latest_call = max(int(row.get("call_index") or 0) for row in rows)
latest = [row for row in rows if int(row.get("call_index") or 0) == latest_call]
def rate(key):
    return sum(1 for row in latest if bool(row.get(key))) / len(latest)
def missing(key):
    return sum(1 for row in latest if row.get(key) in (None, "")) / len(latest)
rewards = [row.get("reward_total") for row in latest if isinstance(row.get("reward_total"), (int, float))]
std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
buckets = collections.Counter(round(float(value), 2) for value in rewards).most_common(8)
print(
    "latest_trace: "
    f"call={latest_call} rows={len(latest)} reward_mean={(sum(rewards)/len(rewards) if rewards else None)} "
    f"reward_std={std} buckets={buckets}"
)
print(
    "latest_rates: "
    f"official_missing={missing('extracted_number'):.3f} robust_missing={missing('robust_extracted_number'):.3f} "
    f"fallback_used={rate('fallback_number_used'):.3f} fallback_exact={rate('fallback_numeric_exact'):.3f} "
    f"robust_exact={rate('robust_numeric_exact'):.3f} no_close={rate('no_close_answer'):.3f} "
    f"single_number={rate('answer_single_number'):.3f}"
)
PY
      fi
      if [[ -f "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" ]]; then
        python - "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" <<'PY' || true
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
best = payload.get("best_lora_checkpoint") or {}
rows = payload.get("rows") or []
print(f"eval_rows: {len(rows)}")
if best:
    print(
        f"best_eval: step={best.get('step')} accuracy={best.get('accuracy')} "
        f"partial={best.get('partial_accuracy')} format={best.get('format_accuracy')} "
        f"robust_exact={best.get('robust_numeric_exact_rate')}"
    )
failures = payload.get("failures") or []
if failures:
    print(f"failures: {failures}")
PY
      fi
    done
  else
    echo "No runs/ directory found yet."
  fi
  echo
  echo "R10 artifacts:"
  find "$ARTIFACT_DIR" -maxdepth 5 -type f 2>/dev/null | sort | tail -n 100 || true
}

stop_reward_r10() {
  require_run_id
  prepare_paths
  local session="tpu-r10-${RUN_ID//./-}"
  echo "Reward R10 directory: $RUN_DIR"
  if [[ -f "$RUN_DIR/pipeline.log" ]] && grep -q 'USER REQUESTED STOP' "$RUN_DIR/pipeline.log"; then
    echo "Run already marked as user-stopped."
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Stopping tmux reward r10 session: $session"
    mkdir -p "$RUN_DIR"
    {
      echo
      echo "==> USER REQUESTED STOP $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >> "$RUN_DIR/pipeline.log"
    tmux send-keys -t "$session" C-c || true
    sleep 10
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux send-keys -t "$session" C-c || true
      sleep 5
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session" || true
    fi
  else
    echo "tmux reward r10: not running ($session)"
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux reward r10: still running ($session)"
    exit 1
  fi
  echo "tmux reward r10: stopped ($session)"
  echo
  echo "Preserved checkpoints:"
  find "$RUN_DIR/runs/R10_numeric_guarded/ckpts/actor" -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sed 's#.*/##' | sort -n | xargs -r echo "checkpoints:"
  echo
  df -h /
}

stop_k8_pilot() {
  require_run_id
  prepare_paths
  local session="tpu-k8-${RUN_ID//./-}"
  echo "K8 pilot directory: $RUN_DIR"
  if [[ -f "$RUN_DIR/pipeline.log" ]] && grep -q 'USER REQUESTED STOP' "$RUN_DIR/pipeline.log"; then
    echo "Run already marked as user-stopped."
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Stopping tmux K8 pilot session: $session"
    mkdir -p "$RUN_DIR"
    {
      echo
      echo "==> USER REQUESTED STOP $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >> "$RUN_DIR/pipeline.log"
    tmux send-keys -t "$session" C-c || true
    sleep 10
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux send-keys -t "$session" C-c || true
      sleep 5
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session" || true
    fi
  else
    echo "tmux K8 pilot: not running ($session)"
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux K8 pilot: still running ($session)"
    exit 1
  fi
  echo "tmux K8 pilot: stopped ($session)"
  echo
  echo "Preserved checkpoints:"
  if [[ -d "$RUN_DIR/runs" ]]; then
    for child in "$RUN_DIR"/runs/*; do
      [[ -d "$child" ]] || continue
      echo "--- $(basename "$child") ---"
      find "$child/ckpts/actor" -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sed 's#.*/##' | sort -n | xargs -r echo "checkpoints:"
    done
  fi
  echo
  df -h /
}

resume_k8_pilot() {
  require_run_id
  prepare_paths
  local session="tpu-k8-${RUN_ID//./-}"
  local run_script="$RUN_DIR/run_k8_pilot.sh"
  if [[ ! -x "$run_script" ]]; then
    echo "Missing executable run script: $run_script" >&2
    exit 1
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; not starting a duplicate." >&2
    exit 1
  fi
  echo "==> Resuming tmux session $session"
  tmux new-session -d -s "$session" "bash '$run_script' 2>&1 | tee -a '$RUN_DIR/pipeline.log'; status=\${PIPESTATUS[0]}; echo; echo \"--- k8 pilot exited (\$status) ---\"; exec bash"
  echo "Resumed. Attach with: tmux attach -t $session"
  echo "Log: $RUN_DIR/pipeline.log"
}

status_k8_pilot() {
  require_run_id
  prepare_paths
  local session="tpu-k8-${RUN_ID//./-}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux k8 pilot: running ($session)"
  else
    echo "tmux k8 pilot: not running ($session)"
  fi
  echo
  echo "K8 pilot directory: $RUN_DIR"
  echo
  if [[ -f "$RUN_DIR/pipeline.log" ]]; then
    echo "K8 pilot markers:"
    grep -nE '^(==>|--- k8 pilot exited)' "$RUN_DIR/pipeline.log" | tail -n 100 || true
    echo
    echo "Last 100 log lines:"
    tail -n 100 "$RUN_DIR/pipeline.log"
  else
    echo "No pipeline.log found yet."
  fi
  echo
  echo "Per-run summaries:"
  if [[ -d "$RUN_DIR/runs" ]]; then
    for child in "$RUN_DIR"/runs/*; do
      [[ -d "$child" ]] || continue
      echo
      echo "--- $(basename "$child") ---"
      if [[ -f "$child/reward_mode.txt" ]]; then
        echo "reward_mode: $(cat "$child/reward_mode.txt")"
      fi
      find "$child/ckpts/actor" -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sed 's#.*/##' | sort -n | tail -n 30 | xargs -r echo "checkpoints:"
      if [[ -f "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" ]]; then
        python - "$child/artifacts/checkpoint_eval/checkpoint_eval_summary.json" <<'PY' || true
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
best = payload.get("best_lora_checkpoint") or {}
rows = payload.get("rows") or []
print(f"eval_rows: {len(rows)}")
if best:
    print(
        f"best_eval: step={best.get('step')} accuracy={best.get('accuracy')} "
        f"partial={best.get('partial_accuracy')} format={best.get('format_accuracy')} "
        f"robust_exact={best.get('robust_numeric_exact_rate')}"
    )
failures = payload.get("failures") or []
if failures:
    print(f"failures: {failures}")
PY
      fi
    done
  else
    echo "No runs/ directory found yet."
  fi
  echo
  echo "K8 artifacts:"
  find "$ARTIFACT_DIR" -maxdepth 5 -type f 2>/dev/null | sort | tail -n 100 || true
}

require_storage() {
  if [[ -z "$PROJECT_ID" ]]; then
    echo "--project-id is required for storage commands." >&2
    exit 2
  fi
  if [[ -z "$STORAGE_BUCKET" ]]; then
    echo "--storage-bucket is required for storage commands." >&2
    exit 2
  fi
}

storage_run_dest() {
  printf 'gs://%s/%s/%s\n' "$STORAGE_BUCKET" "$STORAGE_PREFIX" "$RUN_ID"
}

storage_cache_dest() {
  printf 'gs://%s/%s/huggingface/hub\n' "$STORAGE_BUCKET" "$STORAGE_CACHE_PREFIX"
}

sync_dir_to_storage() {
  local src="$1"
  local dst="$2"
  if [[ -d "$src" ]]; then
    echo "rsync $src -> $dst"
    gcloud storage rsync --recursive "$src" "$dst" --project="$PROJECT_ID"
  fi
}

sync_hf_cache_to_storage() {
  local src="$1"
  local dst="$2"
  if [[ -d "$src" ]]; then
    echo "rsync HF cache $src -> $dst"
    gcloud storage rsync --recursive --no-ignore-symlinks "$src" "$dst" --project="$PROJECT_ID"
  fi
}

copy_file_to_storage() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    echo "cp $src -> $dst"
    gcloud storage cp "$src" "$dst" --project="$PROJECT_ID"
  fi
}

sync_storage() {
  require_run_id
  require_storage
  prepare_paths

  if [[ ! -d "$RUN_DIR" ]]; then
    echo "Run directory not found: $RUN_DIR" >&2
    exit 1
  fi

  local run_dest
  run_dest="$(storage_run_dest)"
  local cache_dest
  cache_dest="$(storage_cache_dest)"

  sync_dir_to_storage "$RUN_DIR/artifacts" "$run_dest/artifacts"
  sync_dir_to_storage "$RUN_DIR/meta" "$run_dest/meta"
  sync_dir_to_storage "$RUN_DIR/tensorboard" "$run_dest/tensorboard"
  sync_dir_to_storage "$RUN_DIR/ckpts" "$run_dest/ckpts"
  sync_dir_to_storage "$RUN_DIR/intermediate_ckpt" "$run_dest/intermediate_ckpt"
  sync_dir_to_storage "$RUN_DIR/runs" "$run_dest/runs"
  copy_file_to_storage "$RUN_DIR/pipeline.log" "$run_dest/pipeline.log"
  copy_file_to_storage "$RUN_DIR/run_baseline.sh" "$run_dest/run_baseline.sh"
  copy_file_to_storage "$RUN_DIR/run_reward_sweep.sh" "$run_dest/run_reward_sweep.sh"
  copy_file_to_storage "$RUN_DIR/run_reward_continuation.sh" "$run_dest/run_reward_continuation.sh"
  copy_file_to_storage "$RUN_DIR/run_candidate_eval.sh" "$run_dest/run_candidate_eval.sh"
  copy_file_to_storage "$RUN_DIR/run_reward_dense.sh" "$run_dest/run_reward_dense.sh"

  # Do not sync HF_HOME wholesale: token files can live there. Only sync model
  # repository cache directories, which are enough to avoid re-downloading weights.
  local cache_dir
  for cache_dir in \
    "$HOME/.cache/huggingface/hub/models--google--gemma-3-1b-it" \
    "$HOME/tpu-runs/_hf_hub_cache/models--google--gemma-3-1b-it"; do
    if [[ -d "$cache_dir" ]]; then
      sync_hf_cache_to_storage "$cache_dir" "$cache_dest/$(basename "$cache_dir")"
    fi
  done

  echo "Cloud Storage usage:"
  gcloud storage du --summarize "$run_dest" --project="$PROJECT_ID" || true
  gcloud storage du --summarize "$cache_dest" --project="$PROJECT_ID" || true
}

restore_cache() {
  require_storage
  prepare_paths

  local cache_src
  cache_src="$(storage_cache_dest)/models--google--gemma-3-1b-it"
  local cache_dst="$HOME/.cache/huggingface/hub/models--google--gemma-3-1b-it"
  mkdir -p "$(dirname "$cache_dst")"

  if gcloud storage ls "$cache_src/**" --project="$PROJECT_ID" >/dev/null 2>&1; then
    echo "rsync $cache_src -> $cache_dst"
    gcloud storage rsync --recursive "$cache_src" "$cache_dst" --project="$PROJECT_ID"
    du -sh "$cache_dst" || true
  else
    echo "No cached Gemma model found at $cache_src"
  fi
}

sync_storage_to_dir() {
  local src="$1"
  local dst="$2"
  if gcloud storage ls "$src/**" --project="$PROJECT_ID" >/dev/null 2>&1; then
    mkdir -p "$dst"
    echo "rsync $src -> $dst"
    gcloud storage rsync --recursive "$src" "$dst" --project="$PROJECT_ID"
  else
    echo "No storage objects found at $src"
  fi
}

restore_run_from_storage() {
  require_run_id
  require_storage
  prepare_paths

  local run_src
  run_src="$(storage_run_dest)"
  sync_storage_to_dir "$run_src/ckpts" "$RUN_DIR/ckpts"
  sync_storage_to_dir "$run_src/intermediate_ckpt" "$RUN_DIR/intermediate_ckpt"
  sync_storage_to_dir "$run_src/tensorboard" "$RUN_DIR/tensorboard"
  sync_storage_to_dir "$run_src/artifacts" "$RUN_DIR/artifacts"
  sync_storage_to_dir "$run_src/meta" "$RUN_DIR/meta"
  sync_storage_to_dir "$run_src/runs" "$RUN_DIR/runs"
}

case "$COMMAND" in
  bootstrap)
    require_run_id
    unpack_bundle
    install_secrets
    bootstrap_env
    check_tpu_backend
    ;;
  submit-baseline)
    submit_baseline
    ;;
  submit-reward-sweep)
    submit_reward_sweep
    ;;
  submit-reward-continuation)
    submit_reward_continuation
    ;;
  submit-candidate-eval)
    submit_candidate_eval
    ;;
  submit-reward-dense)
    submit_reward_dense
    ;;
  submit-r7-large-eval)
    submit_r7_large_eval
    ;;
  submit-r12-best-large-eval)
    submit_r12_best_large_eval
    ;;
  submit-reward-r9)
    submit_reward_r9
    ;;
  submit-reward-r10)
    submit_reward_r10
    ;;
  submit-k8-pilot)
    submit_k8_pilot
    ;;
  submit-k8-r10-only)
    submit_k8_r10_only
    ;;
  submit-k8-r11-fallback-only)
    submit_k8_r11_fallback_only
    ;;
  submit-k8-r12-simple-only)
    submit_k8_r12_simple_only
    ;;
  submit-k8-r12-simple-full)
    submit_k8_r12_simple_full
    ;;
  submit-reward-only-r12-full)
    submit_reward_only_r12_full
    ;;
  submit-r12-non-r64-pilot)
    submit_r12_non_r64_pilot
    ;;
  submit-r12-lora-public-tuning)
    submit_r12_lora_public_tuning
    ;;
  submit-k8-public-beta)
    submit_k8_public_beta
    ;;
  submit-k8-r13-public-beta-only)
    submit_k8_r13_public_beta_only
    ;;
  submit-k8-r14-public-beta-only)
    submit_k8_r14_public_beta_only
    ;;
  eval-checkpoints)
    eval_checkpoints
    ;;
  status)
    status_run
    ;;
  status-sweep)
    status_sweep
    ;;
  status-continuation)
    status_continuation
    ;;
  status-candidate-eval)
    status_candidate_eval
    ;;
  status-reward-dense)
    status_reward_dense
    ;;
  status-r7-large-eval)
    status_r7_large_eval
    ;;
  status-r12-best-large-eval)
    status_r7_large_eval
    ;;
  status-reward-r9)
    status_reward_r9
    ;;
  status-reward-r10)
    status_reward_r10
    ;;
  status-k8-pilot)
    status_k8_pilot
    ;;
  stop-reward-r10)
    stop_reward_r10
    ;;
  stop-k8-pilot)
    stop_k8_pilot
    ;;
  resume-k8-pilot)
    resume_k8_pilot
    ;;
  sync-storage)
    sync_storage
    ;;
  restore-cache)
    restore_cache
    ;;
  ""|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage >&2
    exit 2
    ;;
esac

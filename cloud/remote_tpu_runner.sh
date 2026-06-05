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
  remote_tpu_runner.sh eval-checkpoints --run-id RUN --bundle /path/code.zip [--secrets /path/.env]
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
mkdir -p "\$ARTIFACT_DIR" "\$CKPT_DIR" "\$INTERMEDIATE_CKPT_DIR" "\$TENSORBOARD_DIR" "\$OBS_OUTPUT_DIR" "\$OBS_TRACE_DIR"

if [[ -n "\${WANDB_API_KEY:-}" ]]; then
  echo "==> W&B enabled: project=\$WANDB_PROJECT entity=\${WANDB_ENTITY:-<default>}"
else
  echo "==> W&B disabled: WANDB_API_KEY is not set; using TensorBoard + GCS."
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
  copy_file_to_storage "$RUN_DIR/pipeline.log" "$run_dest/pipeline.log"
  copy_file_to_storage "$RUN_DIR/run_baseline.sh" "$run_dest/run_baseline.sh"

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
  eval-checkpoints)
    eval_checkpoints
    ;;
  status)
    status_run
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

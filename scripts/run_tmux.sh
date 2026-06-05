#!/usr/bin/env bash
# Launch training inside a detached tmux session so closing your shell does
# not kill the run. Re-run this script and it attaches to the existing session.
#
#   ./run_tmux.sh
#   WANDB_RUN_ID=<run-id> ./run_tmux.sh resume
#   TUNIX_VENV=$HOME/venvs/tunix ./run_tmux.sh
#   tmux attach -t tunix
#   tmux kill-session -t tunix

set -euo pipefail

SESSION="${TUNIX_SESSION:-tunix}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TUNIX_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="${TUNIX_VENV:-$HOME/venvs/tunix}"
PYTHON_BIN="${TUNIX_PYTHON:-python}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
LOG="${TUNIX_LOG:-$REPO/scripts/train.log}"
RUN_LABEL="${WANDB_RUN_ID:-manual}"

export OBS_OUTPUT_DIR="${OBS_OUTPUT_DIR:-$REPO/artifacts/local/$RUN_LABEL/observability}"
export OBS_TRACE_DIR="${OBS_TRACE_DIR:-$REPO/artifacts/local/$RUN_LABEL/rollout_traces}"
export OBS_RUN_MANIFEST="${OBS_RUN_MANIFEST:-$REPO/artifacts/local/$RUN_LABEL/run_manifest.json}"

if [[ ! -d "$REPO/scripts" ]]; then
  echo "Could not find scripts directory under REPO=$REPO" >&2
  echo "Set TUNIX_REPO=/path/to/tpu-2026 and try again." >&2
  exit 1
fi

ACTIVATE=""
if [[ -f "$VENV/bin/activate" ]]; then
  ACTIVATE="source $(printf "%q" "$VENV/bin/activate") && "
elif [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "No virtualenv found at VENV=$VENV and VIRTUAL_ENV is not active." >&2
  echo "Set TUNIX_VENV=/path/to/venv or activate the venv before running." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists - attaching."
  exec tmux attach -t "$SESSION"
fi

REPO_Q="$(printf "%q" "$REPO")"
LOG_Q="$(printf "%q" "$LOG")"

INNER="cd $REPO_Q/scripts && ${ACTIVATE}$PYTHON_BIN -u train.py"
if [[ "${1:-}" == "resume" ]]; then
  if [[ -z "$WANDB_RUN_ID" ]]; then
    echo "resume requested, but WANDB_RUN_ID is not set." >&2
    echo "Run: WANDB_RUN_ID=<run-id> ./run_tmux.sh resume" >&2
    exit 1
  fi
  RUN_ID_Q="$(printf "%q" "$WANDB_RUN_ID")"
  INNER="cd $REPO_Q/scripts && ${ACTIVATE}WANDB_RUN_ID=$RUN_ID_Q $PYTHON_BIN -u train.py --wandb-run-id $RUN_ID_Q"
fi

# Keep the shell alive after success/failure so the final output stays readable.
CMD="$INNER 2>&1 | tee -a $LOG_Q; status=\${PIPESTATUS[0]}; echo; echo \"--- process exited (\$status) ---\"; exec bash"
TMUX_CMD="$(printf "%q " bash -lc "$CMD")"

tmux new-session -d -s "$SESSION" "$TMUX_CMD"
echo "Started tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
echo "Log file:                                tail -f $LOG"

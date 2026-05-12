#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/root/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="selectivecl"
PHYSICAL_GPU="3"
LOGICAL_GPU="0"
DATA_ROOT="/root/workspace/andycho/CV/AGD20K"

CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"
VISUAL_RUNS_DIR="${ROOT_DIR}/visual_runs"
TIMESTAMP="${SELECTIVECL_VIS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${SELECTIVECL_VIS_RUN_DIR:-${VISUAL_RUNS_DIR}/${TIMESTAMP}}"
ORCHESTRATOR_LOG="${RUN_DIR}/orchestrator.log"

FOREGROUND=0
MAX_SAVE_IMAGES=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--foreground] [--max-save-images N]

Default behavior starts a background nohup visualization run on physical GPU ${PHYSICAL_GPU}.
Use --foreground for direct execution. Omit --max-save-images to save all test images.
USAGE
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --max-save-images)
      [[ "$#" -ge 2 ]] || {
        echo "--max-save-images requires a value" >&2
        exit 2
      }
      MAX_SAVE_IMAGES="$2"
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

mkdir -p "$RUN_DIR"
trap 'status=$?; printf "[%s] ERROR line %s status %s command: %s\n" "$(date "+%F %T")" "$LINENO" "$status" "$BASH_COMMAND" >> "$ORCHESTRATOR_LOG"; exit "$status"' ERR

if [[ "$FOREGROUND" -eq 0 ]]; then
  args=(--foreground)
  if [[ -n "$MAX_SAVE_IMAGES" ]]; then
    args+=(--max-save-images "$MAX_SAVE_IMAGES")
  fi

  export SELECTIVECL_VIS_RUN_DIR="$RUN_DIR"
  export SELECTIVECL_VIS_TIMESTAMP="$TIMESTAMP"
  nohup bash "$0" "${args[@]}" > "${RUN_DIR}/orchestrator.out" 2>&1 &
  pid=$!
  echo "$pid" > "${RUN_DIR}/pid"

  echo "Started SelectiveCL visualization test in background."
  echo "PID: ${pid}"
  echo "Run dir: ${RUN_DIR}"
  echo "Main log: ${RUN_DIR}/orchestrator.log"
  echo "Nohup output: ${RUN_DIR}/orchestrator.out"
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ORCHESTRATOR_LOG"
}

fail() {
  log "ERROR: $*"
  exit 1
}

run_step() {
  local name="$1"
  local logfile="$2"
  shift 2
  local cmd=("$@")

  log "START ${name}"
  log "Log: ${logfile}"
  printf '[%s] Command:' "$(date '+%F %T')" > "$logfile"
  for arg in "${cmd[@]}"; do
    printf ' %q' "$arg" >> "$logfile"
  done
  printf '\n' >> "$logfile"

  local status=0
  "${cmd[@]}" >> "$logfile" 2>&1 || status=$?
  if [[ "$status" -ne 0 ]]; then
    log "FAILED ${name} with exit code ${status}"
    exit "$status"
  fi
  log "DONE ${name}"
}

setup_environment() {
  [[ -f "$CONDA_SH" ]] || fail "Conda activation script not found: ${CONDA_SH}"
  set +u
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENV_NAME"
  set -u

  export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
  cd "$ROOT_DIR"

  log "Run dir: ${RUN_DIR}"
  log "Root dir: ${ROOT_DIR}"
  log "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; test --gpu ${LOGICAL_GPU}"
  python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())" \
    >> "$ORCHESTRATOR_LOG" 2>&1
}

append_max_save_args() {
  if [[ -n "$MAX_SAVE_IMAGES" ]]; then
    printf '%s\n' --max_save_images "$MAX_SAVE_IMAGES"
  fi
}

main() {
  log "SelectiveCL visualization test started"
  echo "$$" > "${RUN_DIR}/pid"

  setup_environment

  [[ -d "${DATA_ROOT}/Seen" ]] || fail "Missing AGD20K Seen data: ${DATA_ROOT}/Seen"
  [[ -d "${DATA_ROOT}/Unseen" ]] || fail "Missing AGD20K Unseen data: ${DATA_ROOT}/Unseen"
  [[ -s "${CHECKPOINT_DIR}/agd20k_seen.pth" ]] || fail "Missing checkpoint: ${CHECKPOINT_DIR}/agd20k_seen.pth"
  [[ -s "${CHECKPOINT_DIR}/agd20k_unseen.pth" ]] || fail "Missing checkpoint: ${CHECKPOINT_DIR}/agd20k_unseen.pth"

  mapfile -t max_save_args < <(append_max_save_args)

  run_step "test_seen_visual" "${RUN_DIR}/test_seen_visual.log" \
    python test.py \
      --data_root "$DATA_ROOT" \
      --model_file "${CHECKPOINT_DIR}/agd20k_seen.pth" \
      --divide Seen \
      --gpu "$LOGICAL_GPU" \
      --test_batch_size 1 \
      --test_num_workers 0 \
      --save_visuals \
      --save_heatmaps \
      --save_path "${RUN_DIR}/seen_visuals" \
      "${max_save_args[@]}"

  run_step "test_unseen_visual" "${RUN_DIR}/test_unseen_visual.log" \
    python test.py \
      --data_root "$DATA_ROOT" \
      --model_file "${CHECKPOINT_DIR}/agd20k_unseen.pth" \
      --divide Unseen \
      --gpu "$LOGICAL_GPU" \
      --test_batch_size 1 \
      --test_num_workers 0 \
      --save_visuals \
      --save_heatmaps \
      --save_path "${RUN_DIR}/unseen_visuals" \
      "${max_save_args[@]}"

  log "SelectiveCL visualization test completed"
}

main

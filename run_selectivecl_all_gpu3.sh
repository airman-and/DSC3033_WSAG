#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/root/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="selectivecl"
PHYSICAL_GPU="3"
LOGICAL_GPU="0"
DATA_ROOT="/root/workspace/andycho/CV/AGD20K"

CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"
FULL_RUNS_DIR="${ROOT_DIR}/full_runs"
TIMESTAMP="${SELECTIVECL_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${SELECTIVECL_RUN_DIR:-${FULL_RUNS_DIR}/${TIMESTAMP}}"
ORCHESTRATOR_LOG="${RUN_DIR}/orchestrator.log"
DOWNLOAD_LOG="${RUN_DIR}/download.log"

FOREGROUND=0
DOWNLOAD_ONLY=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--foreground] [--download-only]

Default behavior starts a background nohup run on physical GPU ${PHYSICAL_GPU}.
Use --foreground for direct execution, and --download-only to fetch checkpoints
without launching training/test.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --foreground)
      FOREGROUND=1
      ;;
    --download-only)
      DOWNLOAD_ONLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$RUN_DIR"
trap 'status=$?; printf "[%s] ERROR line %s status %s command: %s\n" "$(date "+%F %T")" "$LINENO" "$status" "$BASH_COMMAND" >> "$ORCHESTRATOR_LOG"; exit "$status"' ERR

if [[ "$FOREGROUND" -eq 0 ]]; then
  args=(--foreground)
  if [[ "$DOWNLOAD_ONLY" -eq 1 ]]; then
    args+=(--download-only)
  fi

  export SELECTIVECL_RUN_DIR="$RUN_DIR"
  export SELECTIVECL_TIMESTAMP="$TIMESTAMP"
  nohup bash "$0" "${args[@]}" > "${RUN_DIR}/orchestrator.out" 2>&1 &
  pid=$!
  echo "$pid" > "${RUN_DIR}/pid"

  echo "Started SelectiveCL full run in background."
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
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; train/test --gpu ${LOGICAL_GPU}"
  python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())" \
    >> "$ORCHESTRATOR_LOG" 2>&1
}

ensure_gdown() {
  if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('gdown') else 1)" >/dev/null 2>&1; then
    log "gdown already installed"
    return
  fi

  log "Installing gdown<5 into conda env ${ENV_NAME}"
  python -m pip install "gdown<5" >> "$DOWNLOAD_LOG" 2>&1
}

download_checkpoint() {
  local label="$1"
  local file_id="$2"
  local output="$3"
  local url="https://drive.google.com/file/d/${file_id}/view?usp=sharing"

  if [[ -s "$output" ]]; then
    log "Checkpoint exists, skipping ${label}: ${output}"
    printf '[%s] skip %s %s\n' "$(date '+%F %T')" "$label" "$output" >> "$DOWNLOAD_LOG"
    return
  fi

  log "Downloading ${label} checkpoint to ${output}"
  gdown --fuzzy "$url" -O "$output" >> "$DOWNLOAD_LOG" 2>&1
  [[ -s "$output" ]] || fail "Downloaded checkpoint is missing or empty: ${output}"
}

download_all_checkpoints() {
  mkdir -p "$CHECKPOINT_DIR"
  : > "$DOWNLOAD_LOG"

  ensure_gdown
  download_checkpoint "AGD20K-Seen" "1cYC2PBEjhLntySyP51R46J7i8f1Cf1NT" "${CHECKPOINT_DIR}/agd20k_seen.pth"
  download_checkpoint "AGD20K-Unseen" "1YojVtXtl4gCiqDRDOpHn59vdIPSIIgdt" "${CHECKPOINT_DIR}/agd20k_unseen.pth"
  download_checkpoint "HICO-IIF" "1fOIarlqETEpY7JrqUWjgzvHtwCzRfeGb" "${CHECKPOINT_DIR}/hico_iif.pth"
}

main() {
  log "SelectiveCL full run started"
  echo "$$" > "${RUN_DIR}/pid"

  setup_environment
  download_all_checkpoints

  if [[ "$DOWNLOAD_ONLY" -eq 1 ]]; then
    log "Download-only mode complete"
    exit 0
  fi

  [[ -d "${DATA_ROOT}/Seen" ]] || fail "Missing AGD20K Seen data: ${DATA_ROOT}/Seen"
  [[ -d "${DATA_ROOT}/Unseen" ]] || fail "Missing AGD20K Unseen data: ${DATA_ROOT}/Unseen"

  log "HICO-IIF checkpoint was downloaded, but HICO execution is skipped: /DATA/HICO-IIF is absent and current test.py supports AGD20K Seen/Unseen only."

  run_step "train_seen" "${RUN_DIR}/train_seen.log" \
    python train.py \
      --data_root "$DATA_ROOT" \
      --divide Seen \
      --exp_name SelectiveCL \
      --gpu "$LOGICAL_GPU" \
      --save_root "${RUN_DIR}/train_seen_models"

  run_step "train_unseen" "${RUN_DIR}/train_unseen.log" \
    python train.py \
      --data_root "$DATA_ROOT" \
      --divide Unseen \
      --exp_name SelectiveCL \
      --gpu "$LOGICAL_GPU" \
      --save_root "${RUN_DIR}/train_unseen_models"

  run_step "test_seen_official" "${RUN_DIR}/test_seen_official.log" \
    python test.py \
      --data_root "$DATA_ROOT" \
      --model_file "${CHECKPOINT_DIR}/agd20k_seen.pth" \
      --divide Seen \
      --gpu "$LOGICAL_GPU"

  run_step "test_unseen_official" "${RUN_DIR}/test_unseen_official.log" \
    python test.py \
      --data_root "$DATA_ROOT" \
      --model_file "${CHECKPOINT_DIR}/agd20k_unseen.pth" \
      --divide Unseen \
      --gpu "$LOGICAL_GPU"

  log "SelectiveCL full run completed"
}

main

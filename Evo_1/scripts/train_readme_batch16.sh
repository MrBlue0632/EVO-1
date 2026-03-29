#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/train_readme_batch16.sh [all|stage1|stage2|resume|validate]
# Defaults to `all` (stage1 + stage2), with both stage batch sizes locked to 16.

MODE="${1:-all}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep defaults aligned with the original README training setup.
export REPO_DIR
export RUN_ROOT="${RUN_ROOT:-/root/evo1_runs}"
export RUN_NAME="${RUN_NAME:-version_1.0_stage_12}"
export STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-version_1.0_stage_12}"

export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export USE_DEEPSPEED="${USE_DEEPSPEED:-1}"

# Use README's original deepspeed config file path semantics.
export DS_CONFIG="${DS_CONFIG:-./ds_config.json}"

export STAGE1_STEPS="${STAGE1_STEPS:-5000}"
export STAGE2_STEPS="${STAGE2_STEPS:-80000}"

# Lock both stage batch sizes to 16 as requested.
export BATCH_SIZE=16
export STAGE2_BATCH_SIZE=16
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-1}"

# Match README stage2 behavior: resume from stage1 and reset optimizer/scheduler states.
export STAGE2_INIT_FROM_SCRATCH=0
export USE_AUGMENTATION="${USE_AUGMENTATION:-1}"
export AUGMENTATION_MODE="${AUGMENTATION_MODE:-legacy_mix}"
export AUGMENTATION_PROB="${AUGMENTATION_PROB:-0.5}"
export USE_CACHED_DATASET="${USE_CACHED_DATASET:-1}"
export CACHE_IMAGE_CODEC="${CACHE_IMAGE_CODEC:-png}"

# README used --disable_wandb; keep swanlab disabled by default for parity.
export DISABLE_SWANLAB="${DISABLE_SWANLAB:-1}"

dataset_paths_exist() {
  local cfg="$1"
  if [[ ! -f "$cfg" ]]; then
    return 1
  fi
  local path_line
  local found_any=0
  while IFS= read -r path_line; do
    found_any=1
    if [[ ! -d "$path_line" ]]; then
      return 1
    fi
  done < <(awk -F': ' '/^[[:space:]]*path:[[:space:]]*/ {print $2}' "$cfg")
  [[ "$found_any" -eq 1 ]]
}

DEFAULT_DATASET_CONFIG="$REPO_DIR/dataset/config.yaml"
FALLBACK_DATASET_CONFIG="$REPO_DIR/dataset/config_lerobotv21_click_alarmclock.yaml"
if [[ -z "${DATASET_CONFIG:-}" ]]; then
  DATASET_CONFIG="$DEFAULT_DATASET_CONFIG"
  if ! dataset_paths_exist "$DATASET_CONFIG"; then
    if dataset_paths_exist "$FALLBACK_DATASET_CONFIG"; then
      echo "[train_readme_batch16] WARN: $DEFAULT_DATASET_CONFIG points to unavailable dataset paths."
      echo "[train_readme_batch16] Using fallback: $FALLBACK_DATASET_CONFIG"
      DATASET_CONFIG="$FALLBACK_DATASET_CONFIG"
    else
      echo "[train_readme_batch16] ERROR: No valid dataset config found." >&2
      echo "Checked: $DEFAULT_DATASET_CONFIG and $FALLBACK_DATASET_CONFIG" >&2
      echo "Please set DATASET_CONFIG=/your/path/config.yaml" >&2
      exit 1
    fi
  fi
else
  if ! dataset_paths_exist "$DATASET_CONFIG"; then
    echo "[train_readme_batch16] ERROR: DATASET_CONFIG=$DATASET_CONFIG has unavailable dataset path(s)." >&2
    exit 1
  fi
fi
export DATASET_CONFIG

cd "$REPO_DIR"
./scripts/train_lerobotv21_pipeline.sh "$MODE"

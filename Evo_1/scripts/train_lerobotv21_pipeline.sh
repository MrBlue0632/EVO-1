#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"

REPO_DIR="${REPO_DIR:-/root/EVO-1/Evo_1}"
DATASET_DIR="${DATASET_DIR:-/root/click_alarmclock_25_10_08_v2.1}"
DATASET_CONFIG="${DATASET_CONFIG:-$REPO_DIR/dataset/config_lerobotv21_click_alarmclock.yaml}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python3}"
RUN_ROOT="${RUN_ROOT:-/root/evo1_runs}"
CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/cache}"
CKPT_ROOT="${CKPT_ROOT:-$RUN_ROOT/checkpoints}"
DS_CONFIG="${DS_CONFIG:-$REPO_DIR/ds_config.json}"
NUM_MACHINES="${NUM_MACHINES:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
STAGE1_STEPS="${STAGE1_STEPS:-5000}"
STAGE2_STEPS="${STAGE2_STEPS:-80000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
DISABLE_SWANLAB="${DISABLE_SWANLAB:-0}"
DISABLE_PIN_MEMORY="${DISABLE_PIN_MEMORY:-0}"
DISABLE_PERSISTENT_WORKERS="${DISABLE_PERSISTENT_WORKERS:-0}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
RUN_NAME="${RUN_NAME:-evo1_test}"
VLM_BATCHED_FORWARD="${VLM_BATCHED_FORWARD:-1}"
EMBEDDER_TENSOR_FASTPATH="${EMBEDDER_TENSOR_FASTPATH:-1}"
ALLOW_TF32="${ALLOW_TF32:-1}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
VIDEO_CHECK_PYTHON="${VIDEO_CHECK_PYTHON:-python3}"
SKIP_VIDEO_CHECK="${SKIP_VIDEO_CHECK:-0}"
VIDEO_CHECK_FAIL_FAST="${VIDEO_CHECK_FAIL_FAST:-0}"
VIDEO_CHECK_REPORT="${VIDEO_CHECK_REPORT:-$RUN_ROOT/video_validation_report.json}"
AUTO_EXPORT_CACHED_DATASET="${AUTO_EXPORT_CACHED_DATASET:-1}"
ALLOW_DIM_MISMATCH="${ALLOW_DIM_MISMATCH:-0}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "REPO_DIR not found: $REPO_DIR" >&2
  exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "DATASET_DIR not found: $DATASET_DIR" >&2
  exit 1
fi
if [[ ! -f "$DATASET_CONFIG" ]]; then
  echo "DATASET_CONFIG not found: $DATASET_CONFIG" >&2
  exit 1
fi

if ! command -v "$TRAIN_PYTHON" >/dev/null 2>&1; then
  echo "TRAIN_PYTHON not found in PATH: $TRAIN_PYTHON" >&2
  exit 1
fi

"$TRAIN_PYTHON" - <<'PY'
import importlib.util as iu
required = ["torch", "transformers", "accelerate", "pyarrow"]
missing = [m for m in required if iu.find_spec(m) is None]
if missing:
    raise SystemExit("Missing python packages for training: " + ", ".join(missing))
PY

mkdir -p "$CACHE_DIR" "$CKPT_ROOT/stage1" "$CKPT_ROOT/stage2"

PARQUET_SAMPLE="$(find "$DATASET_DIR/data" -type f -name '*.parquet' -print -quit 2>/dev/null || true)"
if [[ -z "$PARQUET_SAMPLE" ]]; then
  echo "No parquet files found under $DATASET_DIR/data" >&2
  exit 1
fi
if head -n 1 "$PARQUET_SAMPLE" | grep -q "https://git-lfs.github.com/spec/v1"; then
  echo "Detected Git LFS pointer files in dataset (not real parquet/video content)." >&2
  if git lfs version >/dev/null 2>&1; then
    echo "Run these commands first:" >&2
    echo "  cd \"$DATASET_DIR\" && git lfs install && git lfs pull" >&2
  else
    echo "git-lfs is not installed. Install git-lfs, then run:" >&2
    echo "  cd \"$DATASET_DIR\" && git lfs install && git lfs pull" >&2
  fi
  exit 2
fi

NUM_PROCESSES="${NUM_PROCESSES:-$("$TRAIN_PYTHON" -c "import importlib.util as iu; n=1; t=iu.find_spec('torch'); \
print(max(1, __import__('torch').cuda.device_count()) if t else n)")}"

RAW_DIMS="$("$TRAIN_PYTHON" -c "import json, pathlib; p=pathlib.Path('$DATASET_DIR')/'meta'/'episodes_stats.jsonl'; \
q=pathlib.Path('$DATASET_DIR')/'meta'/'stats.json'; s=8; a=7; \
obj=None; \
obj=json.loads(p.open('r', encoding='utf-8').readline())['stats'] if p.exists() else (json.loads(q.read_text(encoding='utf-8')) if q.exists() else None); \
print(len(obj['observation.state']['min']) if obj else s, len(obj['action']['min']) if obj else a)")"
STATE_DIM_AUTO="$(echo "$RAW_DIMS" | awk '{print $1}')"
PER_ACTION_DIM_AUTO="$(echo "$RAW_DIMS" | awk '{print $2}')"
CFG_MAX_STATE_DIM="$(awk -F: '/^[[:space:]]*max_state_dim[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$DATASET_CONFIG")"
CFG_MAX_ACTION_DIM="$(awk -F: '/^[[:space:]]*max_action_dim[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$DATASET_CONFIG")"

if [[ -z "${STATE_DIM:-}" ]]; then
  if [[ -n "$CFG_MAX_STATE_DIM" ]]; then
    STATE_DIM="$CFG_MAX_STATE_DIM"
  else
    STATE_DIM="$STATE_DIM_AUTO"
  fi
fi

if [[ -z "${PER_ACTION_DIM:-}" ]]; then
  if [[ -n "$CFG_MAX_ACTION_DIM" ]]; then
    PER_ACTION_DIM="$CFG_MAX_ACTION_DIM"
  else
    PER_ACTION_DIM="$PER_ACTION_DIM_AUTO"
  fi
fi

if [[ -n "$CFG_MAX_STATE_DIM" && "$STATE_DIM" != "$CFG_MAX_STATE_DIM" ]]; then
  echo "STATE_DIM=$STATE_DIM does not match max_state_dim=$CFG_MAX_STATE_DIM from $DATASET_CONFIG" >&2
  if [[ "$ALLOW_DIM_MISMATCH" != "1" ]]; then
    echo "Set ALLOW_DIM_MISMATCH=1 only if you intentionally know what you are doing." >&2
    exit 1
  fi
fi

if [[ -n "$CFG_MAX_ACTION_DIM" && "$PER_ACTION_DIM" != "$CFG_MAX_ACTION_DIM" ]]; then
  echo "PER_ACTION_DIM=$PER_ACTION_DIM does not match max_action_dim=$CFG_MAX_ACTION_DIM from $DATASET_CONFIG" >&2
  if [[ "$ALLOW_DIM_MISMATCH" != "1" ]]; then
    echo "Set ALLOW_DIM_MISMATCH=1 only if you intentionally know what you are doing." >&2
    exit 1
  fi
fi

LAUNCH_CMD=("$TRAIN_PYTHON" -m accelerate.commands.launch)

echo "MODE=$MODE"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "STATE_DIM=$STATE_DIM PER_ACTION_DIM=$PER_ACTION_DIM"
echo "DATASET_CONFIG=$DATASET_CONFIG"
echo "CACHE_DIR=$CACHE_DIR"
echo "CKPT_ROOT=$CKPT_ROOT"
echo "BATCH_SIZE(stage1)=$BATCH_SIZE BATCH_SIZE(stage2)=$STAGE2_BATCH_SIZE"
echo "VLM_BATCHED_FORWARD=$VLM_BATCHED_FORWARD EMBEDDER_TENSOR_FASTPATH=$EMBEDDER_TENSOR_FASTPATH"
echo "ALLOW_TF32=$ALLOW_TF32 CUDNN_BENCHMARK=$CUDNN_BENCHMARK"

cd "$REPO_DIR"

run_video_check() {
  if [[ "$SKIP_VIDEO_CHECK" == "1" ]]; then
    echo "Skipping video validation because SKIP_VIDEO_CHECK=1"
    return 0
  fi

  local checker_args=(
    scripts/check_lerobot_videos.py
    --dataset-dir "$DATASET_DIR"
    --report-path "$VIDEO_CHECK_REPORT"
  )
  if [[ "$VIDEO_CHECK_FAIL_FAST" == "1" ]]; then
    checker_args+=(--fail-fast)
  fi

  echo "Validating dataset videos before training..."
  "$VIDEO_CHECK_PYTHON" "${checker_args[@]}"
}

COMMON_ARGS=(
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  scripts/train.py
  --action_head flowmatching
  --use_augmentation
  --lr 1e-5
  --dropout 0.2
  --weight_decay 1e-3
  --batch_size "$BATCH_SIZE"
  --image_size 448
  --log_interval 10
  --ckpt_interval 2500
  --warmup_steps 1000
  --grad_clip_norm 1.0
  --num_layers 8
  --horizon 50
  --disable_wandb
  --disable_swanlab
  --vlm_name OpenGVLab/InternVL3-1B
  --dataset_config_path "$DATASET_CONFIG"
  --cache_dir "$CACHE_DIR"
  --dataset_num_workers "$DATASET_NUM_WORKERS"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --per_action_dim "$PER_ACTION_DIM"
  --state_dim "$STATE_DIM"
)

if [[ "$USE_DEEPSPEED" == "1" ]]; then
  COMMON_ARGS=(--deepspeed_config_file "$DS_CONFIG" "${COMMON_ARGS[@]}")
fi

if [[ "$DISABLE_SWANLAB" != "1" ]]; then
  FILTERED_ARGS=()
  for arg in "${COMMON_ARGS[@]}"; do
    if [[ "$arg" != "--disable_swanlab" ]]; then
      FILTERED_ARGS+=("$arg")
    fi
  done
  COMMON_ARGS=("${FILTERED_ARGS[@]}")
fi

if [[ "$AUTO_EXPORT_CACHED_DATASET" != "1" ]]; then
  COMMON_ARGS+=(--no_auto_export_cached_dataset)
fi

if [[ "$DISABLE_PIN_MEMORY" == "1" ]]; then
  COMMON_ARGS+=(--disable_pin_memory)
fi

if [[ "$DISABLE_PERSISTENT_WORKERS" == "1" ]]; then
  COMMON_ARGS+=(--disable_persistent_workers)
fi

if [[ "$VLM_BATCHED_FORWARD" != "1" ]]; then
  COMMON_ARGS+=(--no_vlm_batched_forward)
fi

if [[ "$EMBEDDER_TENSOR_FASTPATH" != "1" ]]; then
  COMMON_ARGS+=(--no_embedder_tensor_fastpath)
fi

if [[ "$ALLOW_TF32" != "1" ]]; then
  COMMON_ARGS+=(--disable_tf32)
fi

if [[ "$CUDNN_BENCHMARK" != "1" ]]; then
  COMMON_ARGS+=(--disable_cudnn_benchmark)
fi

if [[ "$DISABLE_TQDM" == "1" ]]; then
  COMMON_ARGS+=(--disable_tqdm)
fi

run_stage1() {
  "${LAUNCH_CMD[@]}" "${COMMON_ARGS[@]}" \
    --run_name "$RUN_NAME" \
    --max_steps "$STAGE1_STEPS" \
    --finetune_action_head \
    --save_dir "$CKPT_ROOT/stage1"
}

resolve_stage1_resume_path() {
  if [[ -n "${STAGE2_RESUME_PATH:-}" ]]; then
    echo "$STAGE2_RESUME_PATH"
    return 0
  fi

  local preferred_path="$CKPT_ROOT/stage1/step_${STAGE1_STEPS}"
  if [[ -f "$preferred_path/checkpoint.pt" ]]; then
    echo "$preferred_path"
    return 0
  fi

  local final_path="$CKPT_ROOT/stage1/step_final"
  if [[ -f "$final_path/checkpoint.pt" ]]; then
    echo "$final_path"
    return 0
  fi

  local latest_step_num=-1
  local latest_step_path=""
  while IFS= read -r step_dir; do
    local step_name step_suffix
    step_name="$(basename "$step_dir")"
    step_suffix="${step_name#step_}"
    if [[ "$step_suffix" =~ ^[0-9]+$ ]] && [[ -f "$step_dir/checkpoint.pt" ]]; then
      if (( step_suffix > latest_step_num )); then
        latest_step_num="$step_suffix"
        latest_step_path="$step_dir"
      fi
    fi
  done < <(find "$CKPT_ROOT/stage1" -maxdepth 1 -mindepth 1 -type d -name 'step_*' 2>/dev/null)

  if [[ -n "$latest_step_path" ]]; then
    echo "$latest_step_path"
    return 0
  fi

  echo "No usable stage1 checkpoint found under $CKPT_ROOT/stage1" >&2
  echo "Expected one of: step_${STAGE1_STEPS}, step_final, or any step_<N>/checkpoint.pt" >&2
  echo "You can also set STAGE2_RESUME_PATH explicitly." >&2
  return 1
}

run_stage2() {
  local stage2_resume_path
  stage2_resume_path="$(resolve_stage1_resume_path)"
  echo "Stage2 resume checkpoint: $stage2_resume_path"

  "${LAUNCH_CMD[@]}" "${COMMON_ARGS[@]}" \
    --batch_size "$STAGE2_BATCH_SIZE" \
    --run_name "$RUN_NAME" \
    --max_steps "$STAGE2_STEPS" \
    --finetune_vlm \
    --finetune_action_head \
    --save_dir "$CKPT_ROOT/stage2" \
    --resume \
    --resume_pretrain \
    --resume_path "$stage2_resume_path"
}

run_resume() {
  if [[ -z "${RESUME_PATH:-}" ]]; then
    echo "Please set RESUME_PATH for resume mode." >&2
    exit 1
  fi
  "${LAUNCH_CMD[@]}" "${COMMON_ARGS[@]}" \
    --batch_size "$STAGE2_BATCH_SIZE" \
    --run_name "$RUN_NAME" \
    --max_steps "$STAGE2_STEPS" \
    --finetune_vlm \
    --finetune_action_head \
    --save_dir "$CKPT_ROOT/stage2" \
    --resume \
    --resume_path "$RESUME_PATH"
}

case "$MODE" in
  validate)
    run_video_check
    ;;
  stage1)
    run_video_check
    run_stage1
    ;;
  stage2)
    run_video_check
    run_stage2
    ;;
  resume)
    run_video_check
    run_resume
    ;;
  all)
    run_video_check
    run_stage1
    run_stage2
    ;;
  *)
    echo "Unknown mode: $MODE (use validate|stage1|stage2|resume|all)" >&2
    exit 1
    ;;
esac

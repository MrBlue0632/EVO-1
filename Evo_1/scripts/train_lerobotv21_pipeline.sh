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
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-}"
SAME_NETWORK="${SAME_NETWORK:-1}"
RDZV_BACKEND="${RDZV_BACKEND:-}"
RDZV_CONF="${RDZV_CONF:-}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-}"
ENABLE_CPU_AFFINITY="${ENABLE_CPU_AFFINITY:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
STAGE1_STEPS="${STAGE1_STEPS:-5000}"
STAGE2_STEPS="${STAGE2_STEPS:-80000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-$GRAD_ACCUM_STEPS}"
STAGE2_TARGET_GLOBAL_BATCH="${STAGE2_TARGET_GLOBAL_BATCH:-}"
DISABLE_SWANLAB="${DISABLE_SWANLAB:-0}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-evo1-pro}"
USE_AUGMENTATION="${USE_AUGMENTATION:-1}"
AUGMENTATION_MODE="${AUGMENTATION_MODE:-legacy_mix}"
AUGMENTATION_PROB="${AUGMENTATION_PROB:-0.5}"
USE_CACHED_DATASET="${USE_CACHED_DATASET:-1}"
DISABLE_PIN_MEMORY="${DISABLE_PIN_MEMORY:-0}"
DISABLE_PERSISTENT_WORKERS="${DISABLE_PERSISTENT_WORKERS:-0}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
DEEPSPEED_PRESET="${DEEPSPEED_PRESET:-zero2}"
DEEPSPEED_HOSTFILE="${DEEPSPEED_HOSTFILE:-}"
DEEPSPEED_MULTINODE_LAUNCHER="${DEEPSPEED_MULTINODE_LAUNCHER:-}"
RUN_NAME="${RUN_NAME:-evo1_test}"
STAGE2_RUN_NAME_PREFIX="${STAGE2_RUN_NAME_PREFIX:-stage2_test_speedup}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-}"
SEED="${SEED:-42}"
CACHE_FORMAT="${CACHE_FORMAT:-indexed_v2}"
CACHE_SHARD_SIZE_MB="${CACHE_SHARD_SIZE_MB:-512}"
CACHE_IMAGE_CODEC="${CACHE_IMAGE_CODEC:-png}"
STAGE2_INIT_FROM_SCRATCH="${STAGE2_INIT_FROM_SCRATCH:-0}"
VLM_BATCHED_FORWARD="${VLM_BATCHED_FORWARD:-0}"
EMBEDDER_TENSOR_FASTPATH="${EMBEDDER_TENSOR_FASTPATH:-0}"
ALLOW_TF32="${ALLOW_TF32:-0}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
FUSED_ADAMW="${FUSED_ADAMW:-0}"
VIDEO_CHECK_PYTHON="${VIDEO_CHECK_PYTHON:-python3}"
SKIP_VIDEO_CHECK="${SKIP_VIDEO_CHECK:-0}"
VIDEO_CHECK_FAIL_FAST="${VIDEO_CHECK_FAIL_FAST:-0}"
VIDEO_CHECK_REPORT="${VIDEO_CHECK_REPORT:-$RUN_ROOT/video_validation_report.json}"
AUTO_EXPORT_CACHED_DATASET="${AUTO_EXPORT_CACHED_DATASET:-1}"
ALLOW_DIM_MISMATCH="${ALLOW_DIM_MISMATCH:-0}"
TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ -z "$STAGE2_RUN_NAME" ]]; then
  STAGE2_RUN_NAME="${STAGE2_RUN_NAME_PREFIX}_$(date +%Y%m%d_%H%M%S)"
fi

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
if (( NUM_MACHINES > 1 )); then
  if [[ -z "$MAIN_PROCESS_IP" || -z "$MAIN_PROCESS_PORT" ]]; then
    echo "For multi-node launch (NUM_MACHINES>1), MAIN_PROCESS_IP and MAIN_PROCESS_PORT must be set." >&2
    exit 1
  fi
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

if [[ -n "$STAGE2_TARGET_GLOBAL_BATCH" ]]; then
  if ! [[ "$STAGE2_TARGET_GLOBAL_BATCH" =~ ^[0-9]+$ ]] || (( STAGE2_TARGET_GLOBAL_BATCH <= 0 )); then
    echo "STAGE2_TARGET_GLOBAL_BATCH must be a positive integer, got: $STAGE2_TARGET_GLOBAL_BATCH" >&2
    exit 1
  fi
  stage2_global_per_step=$(( STAGE2_BATCH_SIZE * NUM_PROCESSES ))
  if (( stage2_global_per_step <= 0 )); then
    echo "Invalid stage2 global-per-step batch: $stage2_global_per_step" >&2
    exit 1
  fi
  STAGE2_GRAD_ACCUM_STEPS=$(( (STAGE2_TARGET_GLOBAL_BATCH + stage2_global_per_step - 1) / stage2_global_per_step ))
  if (( STAGE2_GRAD_ACCUM_STEPS < 1 )); then
    STAGE2_GRAD_ACCUM_STEPS=1
  fi
fi

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

export TORCH_NCCL_AVOID_RECORD_STREAMS
export NCCL_ASYNC_ERROR_HANDLING

LAUNCH_CMD=(
  "$TRAIN_PYTHON" -m accelerate.commands.launch
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
)
if (( NUM_PROCESSES > 1 )); then
  LAUNCH_CMD+=(--multi_gpu)
fi
if (( NUM_MACHINES > 1 )); then
  LAUNCH_CMD+=(--machine_rank "$MACHINE_RANK")
  if [[ "$SAME_NETWORK" == "1" ]]; then
    LAUNCH_CMD+=(--same_network)
  fi
  if [[ -n "$MAIN_PROCESS_IP" ]]; then
    LAUNCH_CMD+=(--main_process_ip "$MAIN_PROCESS_IP")
  fi
  if [[ -n "$MAIN_PROCESS_PORT" ]]; then
    LAUNCH_CMD+=(--main_process_port "$MAIN_PROCESS_PORT")
  fi
fi
if [[ -n "$RDZV_BACKEND" ]]; then
  LAUNCH_CMD+=(--rdzv_backend "$RDZV_BACKEND")
fi
if [[ -n "$RDZV_CONF" ]]; then
  LAUNCH_CMD+=(--rdzv_conf "$RDZV_CONF")
fi
if [[ -n "$NUM_CPU_THREADS_PER_PROCESS" ]]; then
  LAUNCH_CMD+=(--num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS")
fi
if [[ "$ENABLE_CPU_AFFINITY" == "1" ]]; then
  LAUNCH_CMD+=(--enable_cpu_affinity)
fi

echo "MODE=$MODE"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "NUM_MACHINES=$NUM_MACHINES MACHINE_RANK=$MACHINE_RANK"
echo "STATE_DIM=$STATE_DIM PER_ACTION_DIM=$PER_ACTION_DIM"
echo "DATASET_CONFIG=$DATASET_CONFIG"
echo "CACHE_DIR=$CACHE_DIR"
echo "CKPT_ROOT=$CKPT_ROOT"
echo "BATCH_SIZE(stage1)=$BATCH_SIZE BATCH_SIZE(stage2)=$STAGE2_BATCH_SIZE"
echo "GRAD_ACCUM(stage1)=$GRAD_ACCUM_STEPS GRAD_ACCUM(stage2)=$STAGE2_GRAD_ACCUM_STEPS"
echo "STAGE2_TARGET_GLOBAL_BATCH=${STAGE2_TARGET_GLOBAL_BATCH:-<unset>}"
echo "SWANLAB_PROJECT=$SWANLAB_PROJECT STAGE2_RUN_NAME=$STAGE2_RUN_NAME"
echo "CACHE_FORMAT=$CACHE_FORMAT CACHE_SHARD_SIZE_MB=$CACHE_SHARD_SIZE_MB"
echo "CACHE_IMAGE_CODEC=$CACHE_IMAGE_CODEC"
echo "USE_AUGMENTATION=$USE_AUGMENTATION AUGMENTATION_MODE=$AUGMENTATION_MODE AUGMENTATION_PROB=$AUGMENTATION_PROB"
echo "USE_CACHED_DATASET=$USE_CACHED_DATASET"
echo "VLM_BATCHED_FORWARD=$VLM_BATCHED_FORWARD EMBEDDER_TENSOR_FASTPATH=$EMBEDDER_TENSOR_FASTPATH"
echo "ALLOW_TF32=$ALLOW_TF32 CUDNN_BENCHMARK=$CUDNN_BENCHMARK"
echo "FUSED_ADAMW=$FUSED_ADAMW"
echo "TORCH_NCCL_AVOID_RECORD_STREAMS=$TORCH_NCCL_AVOID_RECORD_STREAMS NCCL_ASYNC_ERROR_HANDLING=$NCCL_ASYNC_ERROR_HANDLING"

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
  scripts/train.py
  --action_head flowmatching
  --augmentation_mode "$AUGMENTATION_MODE"
  --augmentation_prob "$AUGMENTATION_PROB"
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
  --cache_format "$CACHE_FORMAT"
  --cache_shard_size_mb "$CACHE_SHARD_SIZE_MB"
  --cache_image_codec "$CACHE_IMAGE_CODEC"
  --dataset_num_workers "$DATASET_NUM_WORKERS"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
  --seed "$SEED"
  --swanlab_project "$SWANLAB_PROJECT"
  --per_action_dim "$PER_ACTION_DIM"
  --state_dim "$STATE_DIM"
)

if [[ "$USE_AUGMENTATION" == "1" ]]; then
  COMMON_ARGS+=(--use_augmentation)
fi

if [[ "$USE_CACHED_DATASET" != "1" ]]; then
  COMMON_ARGS+=(--no_use_cached_dataset)
fi

if [[ "$USE_DEEPSPEED" == "1" ]]; then
  if [[ "$DS_CONFIG" == "$REPO_DIR/ds_config.json" ]]; then
    case "$DEEPSPEED_PRESET" in
      zero2) DS_CONFIG="$REPO_DIR/config/deepspeed/zero2.json" ;;
      zero3) DS_CONFIG="$REPO_DIR/config/deepspeed/zero3.json" ;;
      zero2_offload) DS_CONFIG="$REPO_DIR/config/deepspeed/zero2_offload.json" ;;
      zero3_offload) DS_CONFIG="$REPO_DIR/config/deepspeed/zero3_offload.json" ;;
      *) echo "Unknown DEEPSPEED_PRESET=$DEEPSPEED_PRESET" >&2; exit 1 ;;
    esac
  fi
  COMMON_ARGS=(--deepspeed_config_file "$DS_CONFIG" "${COMMON_ARGS[@]}")
  if [[ -n "$DEEPSPEED_HOSTFILE" ]]; then
    COMMON_ARGS=(--deepspeed_hostfile "$DEEPSPEED_HOSTFILE" "${COMMON_ARGS[@]}")
  fi
  if [[ -n "$DEEPSPEED_MULTINODE_LAUNCHER" ]]; then
    COMMON_ARGS=(--deepspeed_multinode_launcher "$DEEPSPEED_MULTINODE_LAUNCHER" "${COMMON_ARGS[@]}")
  fi
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

if [[ "$FUSED_ADAMW" != "1" ]]; then
  COMMON_ARGS+=(--no_fused_adamw)
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
  local stage2_args=()
  if [[ "$STAGE2_INIT_FROM_SCRATCH" == "1" ]]; then
    echo "Stage2 starts from random initialization (STAGE2_INIT_FROM_SCRATCH=1)"
  else
    local stage2_resume_path
    stage2_resume_path="$(resolve_stage1_resume_path)"
    echo "Stage2 resume checkpoint: $stage2_resume_path"
    stage2_args+=(--resume --resume_pretrain --resume_path "$stage2_resume_path")
  fi

  "${LAUNCH_CMD[@]}" "${COMMON_ARGS[@]}" \
    --batch_size "$STAGE2_BATCH_SIZE" \
    --gradient_accumulation_steps "$STAGE2_GRAD_ACCUM_STEPS" \
    --run_name "$STAGE2_RUN_NAME" \
    --max_steps "$STAGE2_STEPS" \
    --finetune_vlm \
    --finetune_action_head \
    --save_dir "$CKPT_ROOT/stage2" \
    "${stage2_args[@]}"
}

run_resume() {
  if [[ -z "${RESUME_PATH:-}" ]]; then
    echo "Please set RESUME_PATH for resume mode." >&2
    exit 1
  fi
  "${LAUNCH_CMD[@]}" "${COMMON_ARGS[@]}" \
    --batch_size "$STAGE2_BATCH_SIZE" \
    --gradient_accumulation_steps "$STAGE2_GRAD_ACCUM_STEPS" \
    --run_name "$STAGE2_RUN_NAME" \
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

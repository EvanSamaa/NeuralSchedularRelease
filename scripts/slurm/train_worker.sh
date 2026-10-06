#!/bin/bash
# Per-node launcher for multi-node training. Started on every node by
# train_multinode.sh through srun; do not submit it directly.
#
# Required environment:
#   TRAIN_FILE        training JSONL file(s), comma-separated
#   VALIDATION_FILE   validation JSONL file(s), comma-separated
#   OUTPUT_DIR        where checkpoints and logs go (resumes from the last checkpoint found there)
#   MASTER_ADDR       host name of the first node (set by train_multinode.sh)
# Optional environment:
#   ENV_SETUP         script sourced on each node to load modules / activate the virtualenv
#   INIT_FROM         weights to start from (default THUDM/glm-4-voice-9b)
#   DEEPSPEED_CONFIG  DeepSpeed config (default configs/deepspeed_zero3.json)
#   CACHE_DIR         datasets cache directory
#   GPUS_PER_NODE     processes per node (default 4)
#   MASTER_PORT       rendezvous port (default 29500)
#   NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE, SAVE_STEPS, EVAL_STEPS
set -euo pipefail

: "${TRAIN_FILE:?set TRAIN_FILE}"
: "${VALIDATION_FILE:?set VALIDATION_FILE}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${MASTER_ADDR:?set MASTER_ADDR}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ -n "${ENV_SETUP:-}" ]; then
    # shellcheck disable=SC1090
    source "$ENV_SETUP"
fi
cd "$REPO_ROOT"

# Triton compiles kernels at start-up; give every node its own cache.
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_cache_${SLURM_JOB_ID:-local}_${SLURM_NODEID:-0}"
mkdir -p "$TRITON_CACHE_DIR"
trap 'rm -rf "$TRITON_CACHE_DIR"' EXIT

CACHE_ARGS=()
if [ -n "${CACHE_DIR:-}" ]; then
    CACHE_ARGS=(--cache_dir "$CACHE_DIR")
fi

python -m torch.distributed.run \
    --nnodes="${SLURM_NNODES:-1}" \
    --nproc_per_node="${GPUS_PER_NODE:-4}" \
    --node_rank="${SLURM_NODEID:-0}" \
    --master_addr="$MASTER_ADDR" \
    --master_port="${MASTER_PORT:-29500}" \
    -m neural_scheduler.train.train \
    --deepspeed "${DEEPSPEED_CONFIG:-$REPO_ROOT/configs/deepspeed_zero3.json}" \
    --model_name_or_path "${INIT_FROM:-THUDM/glm-4-voice-9b}" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VALIDATION_FILE" \
    "${CACHE_ARGS[@]}" \
    --num_train_epochs "${NUM_EPOCHS:-2}" \
    --per_device_train_batch_size "${BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "${GRAD_ACCUM:-2}" \
    --learning_rate "${LEARNING_RATE:-5e-5}" \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.01 \
    --gradient_checkpointing \
    --optim "adamw_torch" \
    --seed 42 \
    --bf16 \
    --report_to "tensorboard" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-100}" \
    --eval_strategy "steps" \
    --eval_steps "${EVAL_STEPS:-100}" \
    --do_train \
    --do_eval \
    --weight_decay 0.00 \
    --max_grad_norm 1.0 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --no_remove_unused_columns \
    --metric_for_best_model "eval_loss" \
    --dataloader_num_workers 8 \
    --use_duration_prediction \
    --num_duration_classes 32 \
    --duration_loss_weight 0.5

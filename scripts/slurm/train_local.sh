#!/bin/bash
# Single-process training run for debugging: one GPU, no DeepSpeed, a handful of samples.
#
#   TRAIN_FILE=train.jsonl VALIDATION_FILE=val.jsonl OUTPUT_DIR=/scratch/$USER/debug_run \
#       bash scripts/slurm/train_local.sh
set -euo pipefail

: "${TRAIN_FILE:?set TRAIN_FILE}"
: "${VALIDATION_FILE:?set VALIDATION_FILE}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ -n "${ENV_SETUP:-}" ]; then
    # shellcheck disable=SC1090
    source "$ENV_SETUP"
fi
cd "$REPO_ROOT"

python -m neural_scheduler.train.train \
    --model_name_or_path "${INIT_FROM:-THUDM/glm-4-voice-9b}" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VALIDATION_FILE" \
    --max_train_samples "${MAX_TRAIN_SAMPLES:-16}" \
    --max_eval_samples "${MAX_EVAL_SAMPLES:-4}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.01 \
    --gradient_checkpointing \
    --optim "adamw_torch" \
    --seed 42 \
    --bf16 \
    --report_to "none" \
    --save_strategy "no" \
    --eval_strategy "steps" \
    --eval_steps 4 \
    --do_train \
    --do_eval \
    --weight_decay 0.00 \
    --max_grad_norm 1.0 \
    --no_remove_unused_columns \
    --dataloader_num_workers 0 \
    --overwrite_output_dir \
    --use_duration_prediction \
    --num_duration_classes 32 \
    --duration_loss_weight 0.5

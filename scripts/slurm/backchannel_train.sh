#!/bin/bash
#SBATCH --job-name=bc_train
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#
# Train the joint text + audio backchannel classifier.
#
#   export TRAIN_FILE=/path/to/train.jsonl
#   export VAL_FILE=/path/to/val.jsonl
#   export OUTPUT_DIR=/path/to/checkpoints/joint_merged
#   sbatch --output=/path/to/logs/bc_train_%j.out scripts/slurm/backchannel_train.sh [extra train args]
#
# Optional: CONFIG (default configs/backchannel/joint_merged.yaml), ENV_SETUP,
# REPO_ROOT. Anything after the script name is passed to the trainer and
# overrides the config, e.g. "--balance_nobc --class_weight_method uniform".
set -euo pipefail

: "${TRAIN_FILE:?set TRAIN_FILE}"
: "${VAL_FILE:?set VAL_FILE}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ -n "${ENV_SETUP:-}" ]; then source "${ENV_SETUP}"; fi
cd "${REPO_ROOT}"

python -m neural_scheduler.backchannel.train \
    --config "${CONFIG:-configs/backchannel/joint_merged.yaml}" \
    --train_file "${TRAIN_FILE}" \
    --val_file "${VAL_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"

echo "Best model: ${OUTPUT_DIR}/best_model"

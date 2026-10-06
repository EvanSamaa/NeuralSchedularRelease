#!/bin/bash
#SBATCH --job-name=bc_eval
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#
# Evaluate a backchannel classifier checkpoint (any architecture).
#
#   export CHECKPOINT=/path/to/checkpoints/joint_merged/best_model
#   export TEST_FILE=/path/to/test.jsonl
#   sbatch --output=/path/to/logs/bc_eval_%j.out scripts/slurm/backchannel_eval.sh
#
# Optional: OUTPUT_DIR (default <checkpoint>/../eval), BATCH_SIZE, NUM_WORKERS,
# ABLATION_SAMPLES (if set, also runs the input ablations on that many
# examples), ENV_SETUP, REPO_ROOT.
set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT}"
: "${TEST_FILE:?set TEST_FILE}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${CHECKPOINT}")/eval}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ -n "${ENV_SETUP:-}" ]; then source "${ENV_SETUP}"; fi
cd "${REPO_ROOT}"

python -m neural_scheduler.backchannel.eval \
    --checkpoint "${CHECKPOINT}" \
    --val_file "${TEST_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --num_workers "${NUM_WORKERS:-4}"

if [ -n "${ABLATION_SAMPLES:-}" ]; then
    python -m neural_scheduler.backchannel.ablation \
        --checkpoint "${CHECKPOINT}" \
        --val_file "${TEST_FILE}" \
        --output_json "${OUTPUT_DIR}/ablation_results.json" \
        --num_samples "${ABLATION_SAMPLES}" \
        --batch_size 16
fi

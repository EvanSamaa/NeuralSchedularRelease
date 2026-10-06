#!/bin/bash
#SBATCH --job-name=bc_label
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-99
#
# Label backchannels in Seamless Interaction with Qwen3-Omni, one shard per array task.
#
#   export DATASET_ROOT=/path/to/seamless-interaction
#   export OUTPUT_DIR=/path/to/backchannel_v2
#   export QWEN_MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct
#   sbatch --output=/path/to/logs/bc_label_%A_%a.out scripts/slurm/backchannel_label.sh
#
# Optional: ENV_SETUP (script sourced to activate the environment),
# REPO_ROOT (default: submit directory), and the shard counts below. The array
# range must equal the sum of the four shard counts. Shard layout with the
# defaults: 0-23 improvised/train, 24-95 naturalistic/train, 96-97
# improvised/test, 98-99 naturalistic/test.
set -euo pipefail

: "${DATASET_ROOT:?set DATASET_ROOT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${QWEN_MODEL:?set QWEN_MODEL}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ -n "${ENV_SETUP:-}" ]; then source "${ENV_SETUP}"; fi
cd "${REPO_ROOT}"

N_TRAIN_IMP="${N_TRAIN_IMP:-24}"
N_TRAIN_NAT="${N_TRAIN_NAT:-72}"
N_TEST_IMP="${N_TEST_IMP:-2}"
N_TEST_NAT="${N_TEST_NAT:-2}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?run as an array job}"
OFFSET_TRAIN_NAT=$((N_TRAIN_IMP))
OFFSET_TEST_IMP=$((OFFSET_TRAIN_NAT + N_TRAIN_NAT))
OFFSET_TEST_NAT=$((OFFSET_TEST_IMP + N_TEST_IMP))

if [ "${TASK_ID}" -lt "${OFFSET_TRAIN_NAT}" ]; then
    SPLIT=train; DATASET_TYPE=improvised; SHARD_ID=$((TASK_ID)); NUM_SHARDS=${N_TRAIN_IMP}
elif [ "${TASK_ID}" -lt "${OFFSET_TEST_IMP}" ]; then
    SPLIT=train; DATASET_TYPE=naturalistic; SHARD_ID=$((TASK_ID - OFFSET_TRAIN_NAT)); NUM_SHARDS=${N_TRAIN_NAT}
elif [ "${TASK_ID}" -lt "${OFFSET_TEST_NAT}" ]; then
    SPLIT=test; DATASET_TYPE=improvised; SHARD_ID=$((TASK_ID - OFFSET_TEST_IMP)); NUM_SHARDS=${N_TEST_IMP}
else
    SPLIT=test; DATASET_TYPE=naturalistic; SHARD_ID=$((TASK_ID - OFFSET_TEST_NAT)); NUM_SHARDS=${N_TEST_NAT}
fi

echo "task ${TASK_ID}: ${SPLIT}/${DATASET_TYPE} shard ${SHARD_ID}/${NUM_SHARDS} on $(hostname)"

python -m neural_scheduler.backchannel.labeling.extract \
    --dataset_root "${DATASET_ROOT}" \
    --split "${SPLIT}" \
    --dataset_type "${DATASET_TYPE}" \
    --output_dir "${OUTPUT_DIR}" \
    --qwen_model_path "${QWEN_MODEL}" \
    --shard_id "${SHARD_ID}" \
    --num_shards "${NUM_SHARDS}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --seed 42 \
    --resume

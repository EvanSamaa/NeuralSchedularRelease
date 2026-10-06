#!/bin/bash
# Tokenise one Seamless Interaction shard per array task, for one (data type, split).
# Run preprocess_seamless_manifests.sh for the same DATA_TYPE / SPLIT / OUTPUT_DIR first.
#
# Submit from a directory on project or scratch storage (logs are written to ./logs).
# The array must be 0..NUM_SHARDS-1 of the manifests:
#
#   mkdir -p logs
#   export OUTPUT_DIR=/path/to/seamless_shards
#   DATA_TYPE=improvised   SPLIT=train sbatch --array=0-24 scripts/slurm/preprocess_seamless.sh
#   DATA_TYPE=naturalistic SPLIT=train sbatch --array=0-24 scripts/slurm/preprocess_seamless.sh
#   DATA_TYPE=improvised   SPLIT=test  sbatch --array=0-1  scripts/slurm/preprocess_seamless.sh
#   DATA_TYPE=naturalistic SPLIT=test  sbatch --array=0-1  scripts/slurm/preprocess_seamless.sh
#
# Environment:
#   REPO_DIR        checkout of this repository (default: the submit directory)
#   ENV_SETUP       optional script to source (module loads, virtualenv activation)
#   GLM4VOICE_REPO  checkout of https://github.com/THUDM/GLM-4-Voice
#   EXTRA_ARGS      extra flags for neural_scheduler.data.preprocess (e.g. "--resume")
#
#SBATCH --job-name=ns_prep_seamless
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${DATA_TYPE:?set DATA_TYPE to improvised or naturalistic}"
: "${SPLIT:?set SPLIT to train, dev or test}"
: "${GLM4VOICE_REPO:?set GLM4VOICE_REPO to a GLM-4-Voice checkout}"
export GLM4VOICE_REPO
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [[ -n "${ENV_SETUP:-}" ]]; then
    source "${ENV_SETUP}"
fi

cd "${REPO_DIR}"
# shellcheck disable=SC2086
python -m neural_scheduler.data.preprocess \
    --dataset seamless \
    --data_type "${DATA_TYPE}" \
    --split "${SPLIT}" \
    --output_dir "${OUTPUT_DIR}" \
    --shard_id "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA_ARGS:-}

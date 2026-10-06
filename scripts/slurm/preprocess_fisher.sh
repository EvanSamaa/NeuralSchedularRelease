#!/bin/bash
# Tokenise one Fisher shard per array task.
#
# Submit from a directory on project or scratch storage (logs are written to ./logs):
#
#   mkdir -p logs
#   NUM_SHARDS=21 OUTPUT_DIR=/path/to/fisher_shards \
#   LMDB_PATH=/path/to/fisher_audio.lmdb KEYS_PATH=/path/to/fisher_audio_keys.txt \
#   sbatch --array=0-20 scripts/slurm/preprocess_fisher.sh
#
# The array must be 0..NUM_SHARDS-1. Use AUDIO_DIR=/path/to/stereo/files instead
# of LMDB_PATH/KEYS_PATH to read audio files directly.
#
# Environment:
#   REPO_DIR        checkout of this repository (default: the submit directory)
#   ENV_SETUP       optional script to source (module loads, virtualenv activation)
#   GLM4VOICE_REPO  checkout of https://github.com/THUDM/GLM-4-Voice
#   EXTRA_ARGS      extra flags for neural_scheduler.data.preprocess (e.g. "--resume")
#
#SBATCH --job-name=ns_prep_fisher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

: "${NUM_SHARDS:?set NUM_SHARDS}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${GLM4VOICE_REPO:?set GLM4VOICE_REPO to a GLM-4-Voice checkout}"
export GLM4VOICE_REPO
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [[ -n "${ENV_SETUP:-}" ]]; then
    source "${ENV_SETUP}"
fi

if [[ -n "${AUDIO_DIR:-}" ]]; then
    SOURCE_ARGS=(--audio_dir "${AUDIO_DIR}")
else
    : "${LMDB_PATH:?set LMDB_PATH and KEYS_PATH, or AUDIO_DIR}"
    : "${KEYS_PATH:?set KEYS_PATH}"
    SOURCE_ARGS=(--lmdb_path "${LMDB_PATH}" --keys_path "${KEYS_PATH}")
fi

cd "${REPO_DIR}"
# shellcheck disable=SC2086
python -m neural_scheduler.data.preprocess \
    --dataset fisher \
    "${SOURCE_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_id "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA_ARGS:-}

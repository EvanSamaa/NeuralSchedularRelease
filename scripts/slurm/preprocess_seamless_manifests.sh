#!/bin/bash
# Write the Seamless Interaction shard manifests for one (data type, split). CPU only.
#
# Submit from a directory on project or scratch storage (logs are written to ./logs).
# The shard counts used for the released data were:
#
#   mkdir -p logs
#   export DATA_ROOT=/path/to/seamless-interaction OUTPUT_DIR=/path/to/seamless_shards
#   DATA_TYPE=improvised   SPLIT=train NUM_SHARDS=25 sbatch scripts/slurm/preprocess_seamless_manifests.sh
#   DATA_TYPE=naturalistic SPLIT=train NUM_SHARDS=25 sbatch scripts/slurm/preprocess_seamless_manifests.sh
#   DATA_TYPE=improvised   SPLIT=test  NUM_SHARDS=2  sbatch scripts/slurm/preprocess_seamless_manifests.sh
#   DATA_TYPE=naturalistic SPLIT=test  NUM_SHARDS=2  sbatch scripts/slurm/preprocess_seamless_manifests.sh
#
# Environment:
#   REPO_DIR    checkout of this repository (default: the submit directory)
#   ENV_SETUP   optional script to source (module loads, virtualenv activation)
#
#SBATCH --job-name=ns_seamless_manifests
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${DATA_TYPE:?set DATA_TYPE to improvised or naturalistic}"
: "${SPLIT:?set SPLIT to train, dev or test}"
: "${NUM_SHARDS:?set NUM_SHARDS}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [[ -n "${ENV_SETUP:-}" ]]; then
    source "${ENV_SETUP}"
fi

cd "${REPO_DIR}"
python -m neural_scheduler.data.preprocess \
    --dataset seamless \
    --create_manifests \
    --data_root "${DATA_ROOT}" \
    --data_type "${DATA_TYPE}" \
    --split "${SPLIT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_shards "${NUM_SHARDS}"

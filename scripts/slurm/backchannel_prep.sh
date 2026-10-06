#!/bin/bash
#SBATCH --job-name=bc_prep
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#
# Build classifier examples from labelled interactions (CPU only).
#
#   export INPUT_DIR=/path/to/backchannel_v2/train_naturalistic
#   export OUTPUT_DIR=/path/to/bc_classification_data/train_naturalistic_with_audio
#   sbatch --output=/path/to/logs/bc_prep_%j.out scripts/slurm/backchannel_prep.sh
#
# Optional: LABEL_SCHEME (semantic | duration), CONTEXT_WINDOW, MIN_SPEECH_DUR,
# NEG_RATIO, SEED, ENV_SETUP, REPO_ROOT. Use a different OUTPUT_DIR per label scheme.
set -euo pipefail

: "${INPUT_DIR:?set INPUT_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ -n "${ENV_SETUP:-}" ]; then source "${ENV_SETUP}"; fi
cd "${REPO_ROOT}"

python -m neural_scheduler.backchannel.data.prep \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --label_scheme "${LABEL_SCHEME:-semantic}" \
    --context_window "${CONTEXT_WINDOW:-10.0}" \
    --min_speech_duration "${MIN_SPEECH_DUR:-0.5}" \
    --neg_ratio "${NEG_RATIO:-4.0}" \
    --seed "${SEED:-42}"

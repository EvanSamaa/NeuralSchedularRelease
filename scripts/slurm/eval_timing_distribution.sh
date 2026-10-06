#!/bin/bash
# Gap / overlap duration distributions of sampled dialogues vs. ground truth.
#
#   export CHECKPOINT=/path/to/checkpoint-15200
#   export DATA_JSONL=/path/to/val.jsonl
#   export OUT_DIR=/scratch/$USER/neural_scheduler/eval
#   sbatch --output="$OUT_DIR/logs/%x-%j.out" scripts/slurm/eval_timing_distribution.sh
#
# Submit from the repository root on a file system the cluster allows jobs to run from.
# Optional: ENV_SETUP (script sourced to activate the Python environment), NAME,
# NUM_SAMPLES (100), SEED (1986), EXTRA_ARGS.
#SBATCH --job-name=ns_eval_timing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT to the scheduler checkpoint directory}"
: "${DATA_JSONL:?set DATA_JSONL to the tokenized validation JSONL}"
: "${OUT_DIR:?set OUT_DIR to the output directory}"

if [ -n "${ENV_SETUP:-}" ]; then
    source "$ENV_SETUP"
fi
export MPLBACKEND=Agg
mkdir -p "$OUT_DIR"

python -m neural_scheduler.evaluation.timing_distribution \
    --checkpoint "$CHECKPOINT" \
    --data_jsonl "$DATA_JSONL" \
    --out_dir "$OUT_DIR" \
    --name "${NAME:-dialogue_timing_comparison}" \
    --num_samples "${NUM_SAMPLES:-100}" \
    --seed "${SEED:-1986}" \
    ${EXTRA_ARGS:-}

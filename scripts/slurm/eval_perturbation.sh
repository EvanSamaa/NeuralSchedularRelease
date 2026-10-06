#!/bin/bash
# Turn-order and silence-duration perturbation perplexity.
#
#   export CHECKPOINT=/path/to/checkpoint-15200
#   export DATA_JSONL=/path/to/val.jsonl          # or the synthetic question/answer set for turn order
#   export OUT_DIR=/scratch/$USER/neural_scheduler/eval
#   sbatch --output="$OUT_DIR/logs/%x-%j.out" scripts/slurm/eval_perturbation.sh
#
# Submit from the repository root on a file system the cluster allows jobs to run from.
# Optional: ENV_SETUP (script sourced to activate the Python environment),
# EXPERIMENTS ("turn_order silence"), NUM_SAMPLES (200), SEED (1986),
# TURN_ORDER_ARGS, SILENCE_ARGS.
#SBATCH --job-name=ns_eval_perturbation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT to the scheduler checkpoint directory}"
: "${DATA_JSONL:?set DATA_JSONL to a tokenized JSONL}"
: "${OUT_DIR:?set OUT_DIR to the output directory}"

if [ -n "${ENV_SETUP:-}" ]; then
    source "$ENV_SETUP"
fi
export MPLBACKEND=Agg

for experiment in ${EXPERIMENTS:-turn_order silence}; do
    case "$experiment" in
        turn_order)
            python -m neural_scheduler.evaluation.turn_order \
                --checkpoint "$CHECKPOINT" --data_jsonl "$DATA_JSONL" \
                --out_dir "$OUT_DIR/turn_order" \
                --num_samples "${NUM_SAMPLES:-200}" --seed "${SEED:-1986}" \
                ${TURN_ORDER_ARGS:-}
            ;;
        silence)
            python -m neural_scheduler.evaluation.silence_perturbation \
                --checkpoint "$CHECKPOINT" --data_jsonl "$DATA_JSONL" \
                --out_dir "$OUT_DIR/silence_perturbation" \
                --num_samples "${NUM_SAMPLES:-200}" --seed "${SEED:-1986}" \
                ${SILENCE_ARGS:-}
            ;;
        *)
            echo "unknown experiment: $experiment" >&2
            exit 1
            ;;
    esac
done

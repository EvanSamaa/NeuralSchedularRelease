#!/bin/bash
# Multi-node training of the scheduler LM (the released checkpoint used 4 nodes x 4 GPUs).
#
# Submit from a directory on project or scratch storage, with the variables
# train_worker.sh needs in the environment, for example:
#
#   export TRAIN_FILE=/data/seamless_train.jsonl VALIDATION_FILE=/data/seamless_val.jsonl
#   export OUTPUT_DIR=/scratch/$USER/scheduler_run ENV_SETUP=/path/to/env.sh
#   sbatch --nodes=4 --output=/scratch/$USER/logs/train-%j.out scripts/slurm/train_multinode.sh
#
# Add --account / --partition as your cluster requires. A run that hits the
# time limit continues from its last checkpoint when the same command is
# submitted again.
#SBATCH --job-name=scheduler-train
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --mem=448G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=scheduler-train-%j.out
set -euo pipefail

# Slurm runs a copy of this script from its spool directory; the submit directory is the reference.
export REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29500}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

echo "Master node: $MASTER_ADDR"
echo "All nodes: $SLURM_JOB_NODELIST"

# One launcher per node; --overlap lets the steps share the job's allocation.
srun --overlap bash "$REPO_ROOT/scripts/slurm/train_worker.sh"

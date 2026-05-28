#!/bin/bash
#SBATCH --job-name=tree-matrix
#SBATCH --array=1-500%50
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tree_matrix_%A_%a.out
#SBATCH --error=logs/tree_matrix_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Initialize conda if available, then activate the project env.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    CONDA_ENV_NAME="qcds2026"
    conda activate "${CONDA_ENV_NAME}"
    echo "Activated conda env: ${CONDA_ENV_NAME}"
fi

SEED="${SLURM_ARRAY_TASK_ID}"

echo "Running seed ${SEED} on host $(hostname)"

# Pass optional extra CLI args to run_tree_matrix.py through sbatch.
python new_experiment/run_tree_matrix.py --seeds "${SEED}" "$@"

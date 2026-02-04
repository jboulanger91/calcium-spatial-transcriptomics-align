#!/bin/bash
#SBATCH --job-name=ants_reg
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu
# If your cluster uses account/QOS, add:
# SBATCH --account=YOUR_ACCOUNT
# SBATCH --qos=YOUR_QOS

set -euo pipefail

mkdir -p logs

# ---- ANTs install ----
export ANTS_HOME=/ANTs_install
export PATH="${ANTS_HOME}/bin:${PATH}"

# Optional: ITK threading
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK}

# ---- (Optional) activate python env ----
# source /path/to/venv/bin/activate
# or: module load python/3.10

python3 ANTs_register_slurm.py \
  --fixed "/path/to/fixed.tif" \
  --moving "/path/to/moving.tif" \
  --fixed-spacing-um 0.621 0.621 1.0 \
  --moving-spacing-um 0.396 0.396 2.0 \
  --exp-id exp_001 \
  --fish 2 \
  --out-dir "/path/to/ANTs_output" \
  --keep-nii
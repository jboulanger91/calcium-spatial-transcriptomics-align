#!/bin/bash
#
# BigStream overlay (rigid+affine) on MCMeSU (MeSU cluster)
#

#SBATCH --job-name=bs_overlay
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=jonathan.boulanger@inserm.fr
#SBATCH --mail-type=ALL
#SBATCH --partition=std
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00-12:00:00

set -euo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

# ----------------------------
# Threading (shared-memory)
# ----------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ----------------------------
# Job info
# ----------------------------
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Job name:      ${SLURM_JOB_NAME}"
echo "Node:          ${HOSTNAME}"
echo "CPUs/task:     ${SLURM_CPUS_PER_TASK}"
echo "TMPDIR:        ${TMPDIR}"
echo "Submit dir:    ${SLURM_SUBMIT_DIR}"
echo "SCRATCH:       ${SCRATCH:-}" 
echo "STORE:         ${STORE:-}" 

# ----------------------------
# (Optional) Environment setup
# ----------------------------
# module load python/3.10
# source "${SLURM_SUBMIT_DIR}/venv/bin/activate"

python3 -c "import sys; print('[debug] python:', sys.executable); print('[debug] version:', sys.version.split()[0])"

# ----------------------------
# Define inputs / outputs
# ----------------------------
DATADIR="${SLURM_SUBMIT_DIR}/data"
OUTDIR="${SLURM_SUBMIT_DIR}/BigStream_output"

FIXED="${DATADIR}/exp_001_fish2_s07_pre_GCaMP_cropped.tif"
MOVING="${DATADIR}/2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz_CARE.tif"

mkdir -p "${OUTDIR}"

# Safety checks
ls -l "${FIXED}"  || { echo "[ERROR] Fixed TIFF not found: ${FIXED}"; exit 1; }
ls -l "${MOVING}" || { echo "[ERROR] Moving TIFF not found: ${MOVING}"; exit 1; }

# ----------------------------
# Stage data to local disk
# ----------------------------
WORKDIR="${TMPDIR}/bs_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}"

echo "Staging data to ${WORKDIR}"
cp "${FIXED}"  "${WORKDIR}/fixed.tif"
cp "${MOVING}" "${WORKDIR}/moving.tif"

# ----------------------------
# Run BigStream overlay export
# ----------------------------
export DATADIR
export OUTDIR
export WORKDIR

echo "Starting BigStream overlay pipeline"
python3 -u "${SLURM_SUBMIT_DIR}/BigStream_register_cluster.py" \
  --fixed "${WORKDIR}/fixed.tif" \
  --moving "${WORKDIR}/moving.tif" \
  --out-dir "${OUTDIR}" \
  --run-id "exp_001_fish2"

echo "BigStream overlay finished"
exit 0
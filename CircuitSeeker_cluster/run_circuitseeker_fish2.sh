#!/bin/bash
#
# CircuitSeeker multimodal registration on MCMeSU (MeSU cluster)
#

#SBATCH --job-name=cs_reg
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

# Optional: keep numpy/BLAS from oversubscribing threads
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
echo "SCRATCH:       ${SCRATCH}"
echo "STORE:         ${STORE}"

# ----------------------------
# (Optional) Environment setup
# ----------------------------
# If you use a venv/conda, uncomment and adjust:
# source "${SLURM_SUBMIT_DIR}/venv/bin/activate"
# or:
# module load python/3.10

# Quick sanity: show python + key deps if you want
python3 -c "import sys; print('[debug] python:', sys.executable); print('[debug] version:', sys.version.split()[0])"

# ----------------------------
# Define inputs / outputs
# ----------------------------
# EDIT these to match where you put the TIFFs on MCMeSU (SCRATCH/STORE/submit dir).
DATADIR="${SLURM_SUBMIT_DIR}/data"
OUTDIR="${SLURM_SUBMIT_DIR}/CircuitSeeker_output"

FIXED="${DATADIR}/exp_001_fish2_s07_pre_GCaMP_cropped.tif"
MOVING="${DATADIR}/2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz_enh.tif"

EXP_ID="exp_001"
FISH="2"

# Spacing (microns), order: X Y Z
FIXED_SPACING=(0.621 0.621 1.0)
MOVING_SPACING=(0.396 0.396 2.0)

mkdir -p "${OUTDIR}"

# Safety checks
ls -l "${FIXED}"  || { echo "[ERROR] Fixed TIFF not found: ${FIXED}"; exit 1; }
ls -l "${MOVING}" || { echo "[ERROR] Moving TIFF not found: ${MOVING}"; exit 1; }

# ----------------------------
# Stage data to local disk
# ----------------------------
WORKDIR="${TMPDIR}/cs_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}"

echo "Staging data to ${WORKDIR}"
cp "${FIXED}"  "${WORKDIR}/fixed.tif"
cp "${MOVING}" "${WORKDIR}/moving.tif"

# ----------------------------
# Run CircuitSeeker registration
# ----------------------------
echo "Starting CircuitSeeker registration"

python3 "${SLURM_SUBMIT_DIR}/circuitseeker_multimodal_registration_slurm.py" \
  --fixed "${WORKDIR}/fixed.tif" \
  --moving "${WORKDIR}/moving.tif" \
  --fixed-spacing-um  "${FIXED_SPACING[@]}" \
  --moving-spacing-um "${MOVING_SPACING[@]}" \
  --exp-id "${EXP_ID}" \
  --fish "${FISH}" \
  --out-dir "${OUTDIR}" \
  --pad-um 20 \
  --save-padded-tiffs \
  --mask-sigma 2 \
  --mask-dilate-iter 32 \
  --mask-close-shape 5 5 5 \
  --alignment-spacing 2 \
  --shrink-factors 2 \
  --smooth-sigmas 8 \
  --iterations 2000 \
  --control-point-spacing 8 \
  --control-point-levels 1 2 4 8 16 32 64 \
  --verbose

echo "CircuitSeeker registration finished"
exit 0
#!/bin/bash

#SBATCH --job-name=ants_reg_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=jonathan.boulanger@inserm.fr
#SBATCH --mail-type=ALL
#SBATCH --partition=std
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00-12:00:00

set -euo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

# ----------------------------
# Threading (shared-memory)
# ----------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK}

# ----------------------------
# ANTs installation (project-local)
# ----------------------------
export ANTS_HOME="${SLURM_SUBMIT_DIR}/ANTs_install/install"
export PATH="${ANTS_HOME}/bin:${PATH}"

echo "[debug] ANTS_HOME=${ANTS_HOME}"
ls -l "${ANTS_HOME}/bin/antsRegistration" || exit 1
which antsRegistration || exit 1

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
# Define inputs / outputs
# ----------------------------
# Put your data somewhere you can read from: SCRATCH or STORE.
# EDIT THESE to match where you actually copied the TIFFs on MCMeSU.
DATADIR="${SLURM_SUBMIT_DIR}/data"    
OUTDIR="${SLURM_SUBMIT_DIR}/output"   

FIXED="${DATADIR}/exp_001_fish2_s05-s09_montaged_MattesMI_GCaMP_ch1.tif"
MOVING="${DATADIR}/2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz.tif"

EXP_ID="exp_001"
FISH="2"

mkdir -p "${OUTDIR}"

# Safety checks before copying
ls -l "${FIXED}"  || { echo "[ERROR] Fixed TIFF not found: ${FIXED}"; exit 1; }
ls -l "${MOVING}" || { echo "[ERROR] Moving TIFF not found: ${MOVING}"; exit 1; }

# ----------------------------
# Stage data to local disk
# ----------------------------
WORKDIR="${TMPDIR}/ants_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}"

echo "Staging data to ${WORKDIR}"
cp "${FIXED}"  "${WORKDIR}/fixed.tif"
cp "${MOVING}" "${WORKDIR}/moving.tif"

# ----------------------------
# Run registration
# ----------------------------
echo "Starting ANTs registration"

python3 ANTs_register_slurm_wSyn.py \
  --fixed "${WORKDIR}/fixed.tif" \
  --moving "${WORKDIR}/moving.tif" \
  --fixed-spacing-um 0.621 0.621 1.0 \
  --moving-spacing-um 0.396 0.396 2.0 \
  --exp-id "${EXP_ID}" \
  --fish "${FISH}" \
  --out-dir "${OUTDIR}" \
  --keep-nii

echo "ANTs registration finished"
exit 0
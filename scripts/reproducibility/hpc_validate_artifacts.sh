#!/bin/bash
#SBATCH --account=wfr5091_cr_default
#SBATCH --partition=standard
#SBATCH --job-name=mfpa-artifacts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=mfpa_artifacts_%j.log

set -euo pipefail

: "${MFPA_RELEASE_ROOT:?Set MFPA_RELEASE_ROOT to the extracted release directory}"
: "${MFPA_PYTHON:?Set MFPA_PYTHON to the locked environment Python executable}"

MFPA_CODE_ROOT="${MFPA_RELEASE_ROOT}/code"
MFPA_ARTIFACT_ROOT="${MFPA_RELEASE_ROOT}/artifacts"
MFPA_SMOKE_OUTPUT="${MFPA_SMOKE_OUTPUT:-${SLURM_SUBMIT_DIR}/artifact_smoke_${SLURM_JOB_ID}}"
MFPA_CHECKPOINT_DATASET="${MFPA_CHECKPOINT_DATASET:-all}"
MFPA_SKIP_FIGURES="${MFPA_SKIP_FIGURES:-0}"
MFPA_PYCACHE_ROOT="${MFPA_SMOKE_OUTPUT}/pycache"

mkdir -p "${MFPA_PYCACHE_ROOT}"

export PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="${MFPA_PYCACHE_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Release: ${MFPA_RELEASE_ROOT}"
echo "Python: ${MFPA_PYTHON}"

"${MFPA_PYTHON}" "${MFPA_CODE_ROOT}/scripts/reproducibility/export_model_checkpoints.py" \
  --dataset "${MFPA_CHECKPOINT_DATASET}" \
  --code-root "${MFPA_CODE_ROOT}" \
  --artifact-root "${MFPA_ARTIFACT_ROOT}"

"${MFPA_PYTHON}" "${MFPA_CODE_ROOT}/scripts/reproducibility/validate_quick_artifacts.py" \
  --code-root "${MFPA_CODE_ROOT}" \
  --artifact-root "${MFPA_ARTIFACT_ROOT}"

if [[ "${MFPA_SKIP_FIGURES}" != "1" ]]; then
  "${MFPA_PYTHON}" "${MFPA_CODE_ROOT}/scripts/reproducibility/make_figures_quick.py" \
    --code-root "${MFPA_CODE_ROOT}" \
    --artifact-root "${MFPA_ARTIFACT_ROOT}" \
    --output "${MFPA_SMOKE_OUTPUT}/figures"
fi

"${MFPA_PYTHON}" -m unittest discover -s "${MFPA_CODE_ROOT}/tests" -v
"${MFPA_PYTHON}" -m compileall -q \
  "${MFPA_CODE_ROOT}/scripts" \
  "${MFPA_CODE_ROOT}/screening/scripts"

echo "Smoke outputs: ${MFPA_SMOKE_OUTPUT}"
echo "Finished: $(date --iso-8601=seconds)"

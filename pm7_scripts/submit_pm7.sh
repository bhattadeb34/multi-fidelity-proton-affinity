#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM array-job template for parallel PM7 (MOPAC) proton-affinity
# calculations. Splits an input CSV of SMILES into N chunks and runs one
# array task per chunk.
#
# Usage:
#     sbatch submit_pm7.sh /absolute/path/to/input.csv [N_CHUNKS]
#
# Required edits before first submission:
#   1. Replace every "<...>" placeholder in the SBATCH block below.
#   2. Point CONDA_ACTIVATE / CONDA_ENV at your installation.
#
# Requirements on the compute node:
#   - mopac        (on PATH, v2016+ recommended)
#   - python env   (see dependencies.md / requirements.txt)
# -----------------------------------------------------------------------------

#SBATCH --account=<YOUR_SLURM_ACCOUNT>       # e.g. abc123_default
#SBATCH --partition=<YOUR_CPU_PARTITION>     # e.g. standard / cpu
#SBATCH --job-name=pm7_array_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=168:00:00
#SBATCH --array=1-10                          # <-- must equal N_CHUNKS
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-user=<YOUR_EMAIL@EXAMPLE.COM>
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# -----------------------------------------------------------------------------
# 1. Argument parsing
# -----------------------------------------------------------------------------
if [ -z "${1:-}" ]; then
    echo "ERROR: No input CSV provided."
    echo "USAGE: sbatch $0 /absolute/path/to/input.csv [N_CHUNKS]"
    exit 1
fi

INPUT_CSV=$(realpath "$1")
TOTAL_CHUNKS="${2:-10}"
SCRIPT_DIR="$SLURM_SUBMIT_DIR"

if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: Input CSV not found: $INPUT_CSV"
    exit 1
fi
for f in run_pm7_parallel.py mopac_calculator.py; do
    if [ ! -f "${SCRIPT_DIR}/${f}" ]; then
        echo "ERROR: ${f} not found in ${SCRIPT_DIR}"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# 2. Per-chunk output directories
# -----------------------------------------------------------------------------
MASTER_PROJECT_DIR="run_$(basename "$INPUT_CSV" .csv)_job${SLURM_ARRAY_JOB_ID}"
CHUNK_PROJECT_NAME="results_chunk_${SLURM_ARRAY_TASK_ID}_of_${TOTAL_CHUNKS}"

echo "========================================================"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID of $TOTAL_CHUNKS"
echo "Input CSV:     $INPUT_CSV"
echo "Master dir:    $MASTER_PROJECT_DIR"
echo "Chunk dir:     $CHUNK_PROJECT_NAME"
echo "========================================================"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$MASTER_PROJECT_DIR"
cd "$MASTER_PROJECT_DIR"

# -----------------------------------------------------------------------------
# 3. Environment (EDIT THESE)
# -----------------------------------------------------------------------------
CONDA_ACTIVATE="${CONDA_ACTIVATE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pm7_calc}"

# shellcheck disable=SC1090
source "$CONDA_ACTIVATE"
conda activate "$CONDA_ENV"

# Force single-threaded MOPAC workers - parallelism comes from the
# Python ProcessPool, NOT from MOPAC itself. Leaving BLAS threads > 1 will
# over-subscribe and tank performance.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Python:  $(which python)"
echo "Version: $(python --version)"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"

# -----------------------------------------------------------------------------
# 4. Run
# -----------------------------------------------------------------------------
mkdir -p "$CHUNK_PROJECT_NAME"
LOG_FILE="$CHUNK_PROJECT_NAME/python_run.log"

echo "--------------------------------------------------------"
echo "Python log -> $LOG_FILE"
echo "--------------------------------------------------------"

python "${SCRIPT_DIR}/run_pm7_parallel.py" "$INPUT_CSV" \
    --project_name   "$CHUNK_PROJECT_NAME" \
    --n_processes    "$SLURM_CPUS_PER_TASK" \
    --total_chunks   "$TOTAL_CHUNKS" \
    --current_chunk  "$SLURM_ARRAY_TASK_ID" \
    --force \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "========================================================"
echo "Chunk ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================"
tail -20 "$LOG_FILE" || true
ls -lh "$CHUNK_PROJECT_NAME/" || true

exit $EXIT_CODE

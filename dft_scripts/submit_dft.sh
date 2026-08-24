#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM submission template for B3LYP/def2-TZVP proton-affinity calculations
# using PySCF + gpu4pyscf on a single A100 GPU.
#
# Required edits before sbatch:
#   1. Replace every "<...>" placeholder in the SBATCH block below.
#   2. Point CONDA_ACTIVATE / CONDA_ENV at your installation (see notes).
#   3. Set INPUT_CSV / OUTPUT_DIR for your molecules.
#
# Usage:  sbatch submit_dft.sh
#
# The CSV must have columns: mol_idx,smiles
# Example:
#     mol_idx,smiles
#     mol_001,CCO
#     mol_002,c1ccncc1
# -----------------------------------------------------------------------------

#SBATCH --account=<YOUR_SLURM_ACCOUNT>       # e.g. abc123_default
#SBATCH --partition=<YOUR_GPU_PARTITION>     # e.g. standard / gpu / a100
#SBATCH --job-name=dft_pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1                    # 1x A100 (or edit for your hardware)
#SBATCH --time=48:00:00
#SBATCH --output=logs/dft_pa_%j.out
#SBATCH --error=logs/dft_pa_%j.err

set -euo pipefail

echo "DFT Proton-Affinity Run - B3LYP/def2-TZVP"
echo "Node:  $(hostname)"
echo "Start: $(date)"

# -----------------------------------------------------------------------------
# 1. Environment
# -----------------------------------------------------------------------------
# Option A - conda / mamba / micromamba. Replace the two variables below with
# whatever activates a Python env that has: pyscf, gpu4pyscf, rdkit, pandas,
# numpy, geometric.
CONDA_ACTIVATE="${CONDA_ACTIVATE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-gpu_dft}"

# shellcheck disable=SC1090
source "$CONDA_ACTIVATE"
conda activate "$CONDA_ENV"

# CUDA module (adjust for your cluster; delete if CUDA is on PATH already)
# module load cuda/12.6.2

# Thread tuning
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYSCF_MAX_MEMORY=40000    # MB per PySCF worker

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# -----------------------------------------------------------------------------
# 2. Inputs / outputs (EDIT THESE)
# -----------------------------------------------------------------------------
INPUT_CSV="${INPUT_CSV:-my_molecules.csv}"          # <-- your CSV
RESULTS_DIR="${RESULTS_DIR:-dft_results}"           # per-molecule JSON files
FILES_DIR="${FILES_DIR:-dft_files}"                 # optimized geometries / moldens

export RESULTS_DIR

mkdir -p logs "$RESULTS_DIR" "$FILES_DIR"

# -----------------------------------------------------------------------------
# 3. Run
# -----------------------------------------------------------------------------
python run_dft_pa.py \
    --csv "$INPUT_CSV" \
    --save-files \
    --files-dir "$FILES_DIR"

EXIT_CODE=$?
echo "Finished:  $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE

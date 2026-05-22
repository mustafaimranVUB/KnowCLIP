#!/bin/bash
#SBATCH --job-name=knoclip_preprocess
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tsalaar.2003@gmail.com
# ============================================================
# DICOM Preprocessing (CPU-only)
#
# NOTE: This script is LEGACY / UNUSED in the current codebase.
# The module src.data.dicom_loader does NOT exist in this checkout.
# The project uses pre-processed MIMIC-CXR-JPG images directly.
# This file is kept for reference; it is safe to delete it.
#
# Submit: sbatch scripts/hydra/preprocess_dicom.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SLURM copies the batch script to its spool dir; BASH_SOURCE[0] then points
# there instead of the repo.  Fall back to SLURM_SUBMIT_DIR when needed.
if [[ ! -f "${SCRIPT_DIR}/common_paths.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/hydra"
fi
# shellcheck source=./common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"

echo "=== DICOM Preprocessing ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Date:   $(date)"

# ── Mamba environment ────────────────────────────────────────────────────────
module purge
module load Mamba

MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"
mamba activate "${THESIS_ENV_PATH}"

PYTHON_BIN="${THESIS_ENV_PATH}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: python not found at ${PYTHON_BIN}" >&2
    exit 1
fi

cd "${THESIS_REPO_DIR}"

# ── Caches → scratch ──────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-${THESIS_SCRATCH_ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export TORCH_HOME="${TORCH_HOME:-${THESIS_SCRATCH_ROOT}/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${THESIS_SCRATCH_ROOT}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${THESIS_SCRATCH_ROOT}/pip_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${THESIS_SCRATCH_ROOT}/matplotlib}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

# ── Path resolution (via common_paths.sh) ────────────────────────────────────
INPUT_DIR="${THESIS_DICOM_SOURCE_ROOT:-${THESIS_SCRATCH_ROOT}/mimic-cxr}"
OUTPUT_DIR="${THESIS_DICOM_OUTPUT_ROOT}"
SPLIT_CSV="${INPUT_DIR}/mimic-cxr-2.0.0-split.csv"

echo "Input  : ${INPUT_DIR}/p10"
echo "Output : ${OUTPUT_DIR}"

# ── Preflight check ───────────────────────────────────────────────────────────
"${PYTHON_BIN}" - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("src.data.dicom_loader") is None:
    print("ERROR: Module src.data.dicom_loader not found.")
    print("This script is LEGACY. The project uses JPG images directly.")
    print("Delete preprocess_dicom.sh if you no longer need DICOM conversion.")
    sys.exit(1)
PY

# ── Run ────────────────────────────────────────────────────────────────────────
"${PYTHON_BIN}" -m src.data.dicom_loader \
    --input-dir "${INPUT_DIR}/p10" \
    --output-dir "${OUTPUT_DIR}" \
    --split-csv "${SPLIT_CSV}" \
    --image-size 224 \
    --num-workers "${SLURM_CPUS_PER_TASK}"

echo ""
echo "Preprocessing completed at $(date)"


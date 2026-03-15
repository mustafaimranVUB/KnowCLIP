#!/bin/bash
#SBATCH --job-name=knoclip_preprocess
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# ============================================================
# DICOM Preprocessing (CPU-only)
# Submit: sbatch scripts/hydra/preprocess_dicom.sh
# ============================================================
set -euo pipefail

echo "=== DICOM Preprocessing ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Date:   $(date)"

# Load modules
PROJECT_DIR="${VSC_DATA}/thesis"
ENV_PATH="${PROJECT_DIR}/envs/knowclip"

module purge
module load Mamba

MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"
mamba activate "${ENV_PATH}"

PYTHON_BIN="${ENV_PATH}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: expected python not found at ${PYTHON_BIN}"
    exit 1
fi

# Route caches to scratch to avoid $VSC_HOME growth
export HF_HOME="${VSC_SCRATCH}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TORCH_HOME="${VSC_SCRATCH}/torch_cache"
export XDG_CACHE_HOME="${VSC_SCRATCH}/.cache"
export PIP_CACHE_DIR="${VSC_SCRATCH}/pip_cache"
export MPLCONFIGDIR="${VSC_SCRATCH}/matplotlib"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

# Preflight check
"${PYTHON_BIN}" - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("src.data.dicom_loader") is None:
    print("ERROR: Module src.data.dicom_loader not found in this checkout.")
    print("Either add src/data/dicom_loader.py or skip preprocess_dicom.sh for now.")
    sys.exit(1)
PY

# Preprocess DICOM files → .pt tensors
"${PYTHON_BIN}" -m src.data.dicom_loader \
    --input-dir "$VSC_SCRATCH/mimic-cxr/p10" \
    --output-dir "$VSC_SCRATCH/mimic-cxr-processed" \
    --split-csv "$VSC_SCRATCH/mimic-cxr/mimic-cxr-2.0.0-split.csv" \
    --image-size 224 \
    --num-workers 32

echo ""
echo "Preprocessing completed at $(date)"

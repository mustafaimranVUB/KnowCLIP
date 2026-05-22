#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tsalaar.2003@gmail.com
# ============================================================
# Hydra Environment Setup (SLURM Batch Job)
# Run ONCE to set up the persistent environment.
# Usage: sbatch scripts/hydra/setup_env_sbatch.sh
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

echo "=== KnoCLIP-XAI Environment Setup ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "User   : $USER"
echo "VSC_HOME:    $VSC_HOME"
echo "VSC_DATA:    $VSC_DATA"
echo "VSC_SCRATCH: $VSC_SCRATCH"

PROJECT_DIR="${THESIS_REPO_DIR}"
ENV_PATH="${THESIS_ENV_PATH}"

# Step 1: Load required modules
echo ""
echo "[1/5] Loading modules …"
module purge
module load Mamba

# Step 2: Create persistent conda environment
# 1. Initialize Mamba/Conda for the current shell
# This ensures the 'mamba' and 'conda' commands work within the script
MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"

# 2. Check if the environment directory exists
if [ ! -d "${ENV_PATH}" ]; then
    echo "Environment NOT found at ${ENV_PATH}. Creating now..."
    mamba create -p "${ENV_PATH}" python=3.10 -y
else
    echo "Environment already exists at ${ENV_PATH}. Skipping creation."
fi

# 3. Always activate the environment (whether just created or already existed)
echo "Activating environment..."
mamba activate "${ENV_PATH}"

PYTHON_BIN="${ENV_PATH}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: expected python not found at ${PYTHON_BIN}"
    exit 1
fi

echo "Using Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

# Route all large caches to the new scratch root.
export HF_HOME="${HF_HOME:-${THESIS_SCRATCH_ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export TORCH_HOME="${TORCH_HOME:-${THESIS_SCRATCH_ROOT}/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${THESIS_SCRATCH_ROOT}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${THESIS_SCRATCH_ROOT}/pip_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${THESIS_SCRATCH_ROOT}/matplotlib}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

# Step 3: Activate and install
echo ""
echo "[3/5] Installing dependencies …"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1.
# On this cluster index, the latest available cu121 build is 2.5.1.
"${PYTHON_BIN}" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install torch-geometric and requirements
"${PYTHON_BIN}" -m pip install torch-geometric==2.6.0
"${PYTHON_BIN}" -m pip install -r "${PROJECT_DIR}/requirements.txt"

# OpenCLIP support for non-standard CLIP checkpoints (e.g., BioMedCLIP variants)
"${PYTHON_BIN}" -m pip install open-clip-torch

# Step 4: DICOM handling extras
echo ""
echo "[4/5] Installing DICOM processing extras …"
"${PYTHON_BIN}" -m pip install pydicom python-gdcm Pillow scikit-image

# Step 4b: scispaCy model (installed to $VSC_SCRATCH to avoid quota issues)
echo ""
echo "[4b/5] Installing en_core_sci_lg (scispaCy) to ${THESIS_SCISPACY_CACHE_WRITE} …"
SCISPACY_CACHE="${THESIS_SCISPACY_CACHE_WRITE}"
mkdir -p "${SCISPACY_CACHE}"

# Install scispacy package itself
"${PYTHON_BIN}" -m pip install scispacy==0.5.5

# Install the large en_core_sci_lg model into the scratch cache directory
"${PYTHON_BIN}" -m pip install \
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz" \
    --target "${SCISPACY_CACHE}"

# Export SCISPACY_CACHE so the pipeline can find it
export SCISPACY_CACHE="${SCISPACY_CACHE}"
if ! grep -q "THESIS_SCRATCH_ROOT=/scratch/brussel/vo/000/bvo00010/vsc11249" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<'EOF'
export THESIS_DATA_ROOT=/data/brussel/vo/000/bvo00010/vsc11249
export THESIS_SCRATCH_ROOT=/scratch/brussel/vo/000/bvo00010/vsc11249
export SCISPACY_CACHE=${THESIS_SCRATCH_ROOT}/scispacy_cache
export HF_HOME=${THESIS_SCRATCH_ROOT}/hf_cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}
export TORCH_HOME=${THESIS_SCRATCH_ROOT}/torch_cache
export XDG_CACHE_HOME=${THESIS_SCRATCH_ROOT}/.cache
export PIP_CACHE_DIR=${THESIS_SCRATCH_ROOT}/pip_cache
export MPLCONFIGDIR=${THESIS_SCRATCH_ROOT}/matplotlib
EOF
    echo "Added Hydra storage exports to ~/.bashrc"
fi

# Step 5: Verify installation
echo ""
echo "[5/5] Verifying installation …"
"${PYTHON_BIN}" -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import torch_geometric; print(f'PyG: {torch_geometric.__version__}')
import transformers; print(f'Transformers: {transformers.__version__}')
"

# Create output directories
echo ""
echo "Creating output directories …"
mkdir -p "${THESIS_WRITE_OUTPUT_ROOT}/KG"
mkdir -p "${THESIS_WRITE_OUTPUT_ROOT}/checkpoints"
mkdir -p "${THESIS_WRITE_OUTPUT_ROOT}/logs"

echo ""
echo "=== Setup complete ==="
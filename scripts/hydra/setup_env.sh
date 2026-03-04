#!/bin/bash
# ============================================================
# Hydra Environment Setup
# Run ONCE after first login to set up the persistent environment.
# Usage: bash scripts/hydra/setup_env.sh
# ============================================================
set -euo pipefail

echo "=== KnoCLIP-XAI Environment Setup ==="
echo "Date: $(date)"
echo "User: $USER"
echo "VSC_HOME:    $VSC_HOME"
echo "VSC_DATA:    $VSC_DATA"
echo "VSC_SCRATCH: $VSC_SCRATCH"

PROJECT_DIR="${VSC_DATA}/thesis"
ENV_PATH="${PROJECT_DIR}/envs/knowclip"

# Step 1: Load required modules
echo ""
echo "[1/5] Loading modules …"
module purge
module load Mamba


#Step 2: Create persistent conda environment
if [ -d "${ENV_PATH}" ]; then
    echo "Removing existing environment at ${ENV_PATH}..."
    rm -rf "${ENV_PATH}"
fi

echo "Creating conda environment with Python 3.10..."
mamba create -p "${ENV_PATH}" python=3.10 -y

MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"
mamba activate "${ENV_PATH}"


# Step 3: Activate and install
echo ""
echo "[3/5] Installing dependencies …"

pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 (Combined into one line to avoid backslash errors)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install torch-geometric and requirements
pip install torch-geometric==2.6.0
pip install -r "$VSC_DATA/thesis/requirements.txt"

# Step 4: DICOM handling extras
echo ""
echo "[4/5] Installing DICOM processing extras …"
pip install pydicom python-gdcm Pillow scikit-image

# Step 4b: scispaCy model (installed to $VSC_SCRATCH to avoid quota issues)
# The model is large (~500 MB); store it on scratch.
echo ""
echo "[4b/5] Installing en_core_sci_lg (scispaCy) to \$VSC_SCRATCH/scispacy_cache …"
SCISPACY_CACHE="${VSC_SCRATCH}/scispacy_cache"
mkdir -p "${SCISPACY_CACHE}"
# Install scispacy package itself (already in requirements.txt but ensure it's there)
pip install scispacy==0.5.5
# Install the large en_core_sci_lg model into the scratch cache directory
pip install \
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_lg-0.5.5.tar.gz" \
    --target "${SCISPACY_CACHE}"
# Export SCISPACY_CACHE so the pipeline can find it (also add to ~/.bashrc for persistence)
export SCISPACY_CACHE="${SCISPACY_CACHE}"
if ! grep -q "SCISPACY_CACHE" ~/.bashrc 2>/dev/null; then
    echo "export SCISPACY_CACHE=\${VSC_SCRATCH}/scispacy_cache" >> ~/.bashrc
    echo "Added SCISPACY_CACHE to ~/.bashrc"
fi

# Step 5: Verify installation
echo ""
echo "[5/5] Verifying installation …"
python -c "
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

mkdir -p "$VSC_DATA/thesis/outputs/KG"
mkdir -p "$VSC_DATA/thesis/outputs/checkpoints"
mkdir -p "$VSC_DATA/thesis/outputs/logs"

echo ""
echo "=== Setup complete ==="


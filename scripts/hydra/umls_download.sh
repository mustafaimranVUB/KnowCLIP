#!/bin/bash
set -euo pipefail

module purge
module load Mamba

PROJECT_DIR="${VSC_DATA}/thesis"
ENV_PATH="${PROJECT_DIR}/envs/knowclip"

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

# Install gdown and any core requirements
echo "Installing dependencies..."
"${PYTHON_BIN}" -m pip install gdown
# Optional: pip install -r requirements.txt (if you have one)

# ── 3. Download UMLS Files ─────────────────────────────────────────────────
# Define your Google Drive IDs here
declare -A FILES=(
    ["MRCONSO.RRF"]="1cZI9UDzW-vt0IS9nHifRP5wzE-MSx-IQ"
    ["Needed.tar.xz"]="1SqRjIxC-_lrksMqGdAWqFBw3SIjcSDBm"
)

OUTDIR="$VSC_SCRATCH/ontology/umls-2025AB-metathesaurus-full/2025AB/META"
mkdir -p "$OUTDIR"

for filename in "${!FILES[@]}"; do
    if [ ! -f "$OUTDIR/$filename" ]; then
        echo "Downloading $filename..."
        gdown "${FILES[$filename]}" -O "$OUTDIR/$filename" --remaining-ok
    else
        echo "$filename already exists, skipping download."
    fi
done




echo "=== Job Finished: $(date) ==="
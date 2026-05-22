#!/bin/bash
#SBATCH --job-name=umls_download
#SBATCH --partition=zen4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tsalaar.2003@gmail.com
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SLURM copies the batch script to its spool dir; BASH_SOURCE[0] then points
# there instead of the repo.  Fall back to SLURM_SUBMIT_DIR when needed.
if [[ ! -f "${SCRIPT_DIR}/common_paths.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/hydra"
fi
# shellcheck source=./common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"

module purge
module load Mamba

PROJECT_DIR="${THESIS_REPO_DIR}"
ENV_PATH="${THESIS_ENV_PATH}"

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
export HF_HOME="${HF_HOME:-${THESIS_SCRATCH_ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export TORCH_HOME="${TORCH_HOME:-${THESIS_SCRATCH_ROOT}/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${THESIS_SCRATCH_ROOT}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${THESIS_SCRATCH_ROOT}/pip_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${THESIS_SCRATCH_ROOT}/matplotlib}"
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

OUTDIR="${UMLS_META_ROOT:-${THESIS_UMLS_META_ROOT_WRITE}}"
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
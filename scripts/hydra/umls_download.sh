#!/bin/bash
set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
# Activate the environment
ENV_DIR="$VSC_DATA/thesis/envs/knoclip"
source "$ENV_DIR/bin/activate"

# Install gdown and any core requirements
echo "Installing dependencies..."
pip install gdown
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
#!/bin/bash
#SBATCH --job-name=knoclip_preprocess
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
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
module load Python/3.11.5-GCCcore-13.2.0

# Activate environment
source "$VSC_DATA/envs/knoclip/bin/activate"

# Working directory
cd "$VSC_HOME/knoclip-xai"

# Preprocess DICOM files → .pt tensors
python -m src.data.dicom_loader \
    --input-dir "$VSC_SCRATCH/mimic-cxr/p10" \
    --output-dir "$VSC_SCRATCH/mimic-cxr-processed" \
    --split-csv "$VSC_SCRATCH/mimic-cxr/mimic-cxr-2.0.0-split.csv" \
    --image-size 224 \
    --num-workers 32

echo ""
echo "Preprocessing completed at $(date)"

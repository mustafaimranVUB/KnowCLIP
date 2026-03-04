#!/bin/bash
#SBATCH --job-name=knoclip_train_ns
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_ns_%j.out
#SBATCH --error=logs/train_ns_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@vub.be
# ============================================================
# Phase II — Neuro-Symbolic Training (KG + report generation)
# Submit: sbatch scripts/hydra/train_neurosymbolic.sh
# ============================================================
set -euo pipefail

echo "=== Phase II: Neuro-Symbolic Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "GPUs:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Load modules
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate environment
source "$VSC_DATA/envs/knoclip/bin/activate"

# Working directory
cd "$VSC_HOME/knoclip-xai"

# Set environment
export MIMIC_ROOT="$VSC_SCRATCH/mimic-cxr"
export MIMIC_REPORTS="$VSC_SCRATCH/mimic-cxr-reports/files"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8

# Run training
python main.py train \
    --config configs/phase2_neurosymbolic.yaml \
    --seed 42

echo ""
echo "Neuro-symbolic training completed at $(date)"

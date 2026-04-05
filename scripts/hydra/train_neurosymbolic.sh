#!/bin/bash
#SBATCH --job-name=knoclip_train_ns
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/train_ns_%j.out
#SBATCH --error=logs/train_ns_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tsalaar.2003@gmail.com
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
# Load modules
module purge
module load Mamba

MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"
mamba activate "${VSC_DATA}/thesis/envs/knowclip"

# Working directory
cd "$VSC_DATA/thesis"

mkdir -p outputs/logs

# Set environment
export MIMIC_ROOT="$VSC_SCRATCH/mimic-cxr"
export MIMIC_REPORTS="$VSC_SCRATCH/mimic-cxr-reports/files"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
# Enable TF32 on A100 for faster matmul/convolutions
export NVIDIA_TF32_OVERRIDE=1
export HF_HOME="$VSC_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TORCH_HOME="$VSC_SCRATCH/torch_cache"
export XDG_CACHE_HOME="$VSC_SCRATCH/.cache"
export PIP_CACHE_DIR="$VSC_SCRATCH/pip_cache"
export MPLCONFIGDIR="$VSC_SCRATCH/matplotlib"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

# Run training
CONFIG_PATH="${CONFIG_PATH:-configs/phase2_neurosymbolic.yaml}"
SEED="${SEED:-42}"

echo "Config : ${CONFIG_PATH}"
echo "Seed   : ${SEED}"

python -m main \
    --config "${CONFIG_PATH}" \
    --seed "${SEED}" \
    train

echo ""
echo "Neuro-symbolic training completed at $(date)"

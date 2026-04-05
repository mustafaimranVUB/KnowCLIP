#!/bin/bash
#SBATCH --job-name=phase1_kg
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# ============================================================
# Phase I — Knowledge Graph Construction (full dataset, GPU)
# Submit from thesis root: sbatch scripts/hydra/run_phase1.sh
#
# Optional: resume after interruption by adding --skip-extraction
# and/or --skip-grounding --skip-embedding if those .pt files exist.
# ============================================================
set -euo pipefail

echo "=== Phase I: KG Construction ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Start   : $(date)"

# ── Modules ──────────────────────────────────────────────────────────────────
# module purge
# module load Python/3.11.3-GCCcore-12.3.0
# module load CUDA/12.1.1
# module load cuDNN/8.9.2.26-CUDA-12.1.1

# # ── Environment ──────────────────────────────────────────────────────────────
# source "$VSC_DATA/thesis/envs/knowclip/bin/activate"

PROJECT_DIR="${VSC_DATA}/thesis"
ENV_PATH="${PROJECT_DIR}/envs/knowclip"

module purge
module load Mamba

MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"
mamba activate "${ENV_PATH}"

# Redirect HF / torch cache to scratch (quota-safe)
export HF_HOME="$VSC_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TORCH_HOME="$VSC_SCRATCH/torch_cache"
export XDG_CACHE_HOME="$VSC_SCRATCH/.cache"
export PIP_CACHE_DIR="$VSC_SCRATCH/pip_cache"
export MPLCONFIGDIR="$VSC_SCRATCH/matplotlib"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

# ── Working directory ─────────────────────────────────────────────────────────
cd "$VSC_DATA/thesis"
mkdir -p outputs/KG_outputs/logs

echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python  : $(which python)"
echo ""

# ── Run Phase I ───────────────────────────────────────────────────────────────
# --batch-size 64  : safe for A100 with modern-radgraph-xl + SapBERT
# Remove --max-studies to process the full dataset.
# Add --skip-extraction / --skip-grounding / --skip-embedding to resume
# from a checkpoint if the job is restarted.
srun python -m main \
    --config configs/phase1_kg_gpu.yaml \
    --device cuda \
    phase1 \
    --batch-size 64 \
    2>&1 | tee outputs/KG_outputs/logs/phase1_${SLURM_JOB_ID}.log

echo ""
echo "=== Phase I finished: $(date) ==="

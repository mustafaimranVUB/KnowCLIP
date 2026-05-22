#!/bin/bash
#SBATCH --job-name=phase1_kg
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tsalaar.2003@gmail.com
# ============================================================
# Phase I — Knowledge Graph Construction (full dataset, GPU)
# Submit from thesis root: sbatch scripts/hydra/run_phase1.sh
#
# Optional: resume after interruption by adding --skip-extraction
# and/or --skip-grounding --skip-embedding if those .pt files exist.
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

echo "=== Phase I: KG Construction ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Start   : $(date)"

# ── Modules ──────────────────────────────────────────────────────────────────
# module purge
# module load Python/3.11.3-GCCcore-12.3.0
# module load CUDA/12.1.1
# module load cuDNN/8.9.2.26-CUDA-12.1.1

PROJECT_DIR="${THESIS_REPO_DIR}"
ENV_PATH="${THESIS_ENV_PATH}"

module purge
module load Mamba

MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"
mamba activate "${ENV_PATH}"
PYTHON_BIN="${ENV_PATH}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: python not found at ${PYTHON_BIN}" >&2
    exit 1
fi

export PHASE1_SOURCE_ROOT="${PHASE1_SOURCE_ROOT:-${THESIS_PHASE1_REPORT_SOURCE_ROOT}}"
export PHASE1_REPORTS_ROOT="${PHASE1_REPORTS_ROOT:-${THESIS_PHASE1_REPORTS_ROOT}}"
export KG_OUTPUT_DIR="${KG_OUTPUT_DIR:-${THESIS_KG_WRITE_DIR}}"
CONFIG_PATH="${CONFIG_PATH:-configs/phase1_kg_gpu.yaml}"
export LOG_DIR="${LOG_DIR:-${THESIS_LOG_WRITE_DIR}}"
export UMLS_MRCONSO_PATH="${UMLS_MRCONSO_PATH:-${THESIS_UMLS_META_ROOT_READ}/MRCONSO.RRF}"
export UMLS_MRSTY_PATH="${UMLS_MRSTY_PATH:-${THESIS_UMLS_META_ROOT_READ}/MRSTY.RRF}"
export UMLS_MRREL_PATH="${UMLS_MRREL_PATH:-${THESIS_UMLS_META_ROOT_READ}/MRREL.RRF}"
export SCISPACY_CACHE="${SCISPACY_CACHE:-${THESIS_SCISPACY_CACHE_READ}}"

export HF_HOME="${HF_HOME:-${THESIS_SCRATCH_ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TORCH_HOME="${TORCH_HOME:-${THESIS_SCRATCH_ROOT}/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${THESIS_SCRATCH_ROOT}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${THESIS_SCRATCH_ROOT}/pip_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${THESIS_SCRATCH_ROOT}/matplotlib}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

# ── Working directory ─────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"
mkdir -p "${KG_OUTPUT_DIR}" "${LOG_DIR}" "${THESIS_WRITE_OUTPUT_ROOT}/KG_outputs/logs"

echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python  : $(which python)"
echo "Reports : ${PHASE1_REPORTS_ROOT}"
echo "KG out  : ${KG_OUTPUT_DIR}"
echo "Config  : ${CONFIG_PATH}"
echo "UMLS    : ${THESIS_UMLS_META_ROOT_READ}"
echo ""

# ── Run Phase I ───────────────────────────────────────────────────────────────
# --batch-size 64  : safe for A100 with modern-radgraph-xl + SapBERT
# Remove --max-studies to process the full dataset.
# Add --skip-extraction / --skip-grounding / --skip-embedding to resume
# from a checkpoint if the job is restarted.
srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m main \
    --config "${CONFIG_PATH}" \
    --device cuda \
    phase1 \
    --batch-size 64 \
    2>&1 | tee "${THESIS_WRITE_OUTPUT_ROOT}/KG_outputs/logs/phase1_${SLURM_JOB_ID}.log"

echo ""
echo "=== Phase I finished: $(date) ==="

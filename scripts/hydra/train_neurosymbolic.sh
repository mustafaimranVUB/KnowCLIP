#!/bin/bash
#SBATCH --job-name=knoclip_train_ns
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=logs/train_ns_%j.out
#SBATCH --error=logs/train_ns_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tsalaar.2003@gmail.com
# ============================================================
# Phase II — Neuro-Symbolic Training (KG + report generation)
# Submit: sbatch scripts/hydra/train_neurosymbolic.sh
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

echo "=== Phase II: Neuro-Symbolic Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "GPUs:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

module purge
module load Mamba

MAMBA_ROOT=$(dirname $(dirname $(which mamba)))
source "${MAMBA_ROOT}/etc/profile.d/conda.sh"
source "${MAMBA_ROOT}/etc/profile.d/mamba.sh"
mamba activate "${THESIS_ENV_PATH}"
PYTHON_BIN="${THESIS_ENV_PATH}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: python not found at ${PYTHON_BIN}" >&2
    exit 1
fi
cd "${THESIS_REPO_DIR}"

mkdir -p "${THESIS_LOG_WRITE_DIR}"

# Run configuration
CONFIG_PATH="${CONFIG_PATH:-configs/hydra_phase2_neurosymbolic_gpt2_jpg.yaml}"
SEED="${SEED:-42}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Missing config: ${CONFIG_PATH}" >&2
    exit 1
fi

config_base="$(basename "${CONFIG_PATH}" .yaml)"
if [[ "${config_base}" == "hydra_phase2_neurosymbolic_gpt2_jpg" ]]; then
    default_run_name="neurosymbolic_gpt2_hydra_jpg"
else
    default_run_name="${config_base}"
fi
RUN_NAME="${RUN_NAME:-${default_run_name}}"
TRAIN_NO_EVAL="${TRAIN_NO_EVAL:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"

# Set environment
DATASET_ROOT="${MIMIC_DATASET_ROOT:-${THESIS_DATASET_READ_ROOT}}"
export MIMIC_ROOT="${MIMIC_ROOT:-${DATASET_ROOT}}"
export MIMIC_REPORTS="${MIMIC_REPORTS:-${DATASET_ROOT}/mimic-cxr-reports/files}"
export MIMIC_SPLIT_CSV="${MIMIC_SPLIT_CSV:-${DATASET_ROOT}/mimic-cxr-2.0.0-split.csv.gz}"
export MIMIC_CHEXPERT_CSV="${MIMIC_CHEXPERT_CSV:-${DATASET_ROOT}/mimic-cxr-2.0.0-chexpert.csv.gz}"
export KG_ARTIFACTS_DIR="${KG_ARTIFACTS_DIR:-${THESIS_KG_READ_DIR}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${THESIS_WRITE_OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
export LOG_DIR="${LOG_DIR:-${THESIS_LOG_WRITE_DIR}}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
# Enable TF32 on A100 for faster matmul/convolutions
export NVIDIA_TF32_OVERRIDE=1
export HF_HOME="${HF_HOME:-${THESIS_SCRATCH_ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TORCH_HOME="${TORCH_HOME:-${THESIS_SCRATCH_ROOT}/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${THESIS_SCRATCH_ROOT}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${THESIS_SCRATCH_ROOT}/pip_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${THESIS_SCRATCH_ROOT}/matplotlib}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"

resume_explicit=0
RESUME_FROM="${RESUME_FROM:-}"
if [[ -n "${RESUME_FROM}" ]]; then
    resume_explicit=1
fi

is_resume_compatible() {
    local ckpt_path="$1"
    local cfg_path="$2"
    "${PYTHON_BIN}" - "$ckpt_path" "$cfg_path" <<'PY'
import sys
from pathlib import Path
import torch
from src.core.config import load_config

ckpt_path = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])

def get_path(dct, path, default=None):
    cur = dct
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

try:
    payload = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
except Exception as exc:
    print(f"ERROR: cannot read checkpoint '{ckpt_path}': {exc}")
    sys.exit(2)

saved_cfg = payload.get('config', {}) if isinstance(payload, dict) else {}
saved_model = saved_cfg.get('model', {}) if isinstance(saved_cfg, dict) else {}
target_model = load_config(cfg_path).to_dict().get('model', {})

keys = [
    'use_kg',
    'enable_classification',
    'enable_report_generation',
    'visual_encoder.backbone_type',
    'visual_encoder.output_dim',
    'report_generation.decoder_type',
    'report_generation.num_decoder_layers',
    'report_generation.decoder_dim',
    'report_generation.num_decoder_heads',
    'knowledge_encoder.num_gat_layers',
    'knowledge_encoder.hidden_channels',
    'knowledge_encoder.num_attention_heads',
    'knowledge_encoder.output_dim',
    'fusion_module.num_fusion_layers',
    'fusion_module.num_heads',
    'fusion_module.bottleneck_dim',
    'classification_head.num_classes',
    'classification_head.hidden_dim',
]

mismatches = []
for key in keys:
    if get_path(saved_model, key) != get_path(target_model, key):
        mismatches.append(key)

if mismatches:
    print('INCOMPATIBLE model signature fields: ' + ', '.join(mismatches))
    sys.exit(1)

sys.exit(0)
PY
}

if [[ "${AUTO_RESUME}" == "1" && -z "${RESUME_FROM}" ]]; then
    latest_checkpoint=$(ls -1 "${CHECKPOINT_DIR}"/checkpoint_epoch*.pt 2>/dev/null | sort | tail -n 1 || true)
    if [[ -n "${latest_checkpoint}" ]]; then
        RESUME_FROM="${latest_checkpoint}"
    elif [[ -f "${CHECKPOINT_DIR}/best_model.pt" ]]; then
        RESUME_FROM="${CHECKPOINT_DIR}/best_model.pt"
    fi
fi

if [[ -n "${RESUME_FROM}" ]]; then
    if ! is_resume_compatible "${RESUME_FROM}" "${CONFIG_PATH}"; then
        if [[ "${resume_explicit}" -eq 1 ]]; then
            echo "ERROR: Explicit RESUME_FROM is incompatible with ${CONFIG_PATH}: ${RESUME_FROM}" >&2
            exit 1
        fi
        echo "WARN: Skipping incompatible auto-resume checkpoint for ${CONFIG_PATH}: ${RESUME_FROM}" >&2
        RESUME_FROM=""
    fi
fi

echo "Config : ${CONFIG_PATH}"
echo "Run    : ${RUN_NAME}"
echo "Seed   : ${SEED}"
echo "No eval: ${TRAIN_NO_EVAL}"
echo "Auto rs: ${AUTO_RESUME}"
echo "Data   : ${DATASET_ROOT}"
echo "KG dir : ${KG_ARTIFACTS_DIR}"
echo "CKPT   : ${CHECKPOINT_DIR}"
if [[ -n "${RESUME_FROM}" ]]; then
    echo "Resume : ${RESUME_FROM}"
fi

cmd=("${PYTHON_BIN}" -m main \
    --config "${CONFIG_PATH}" \
    --seed "${SEED}" \
    train)

if [[ -n "${RESUME_FROM}" ]]; then
    cmd+=(--resume "${RESUME_FROM}")
fi

if [[ "${TRAIN_NO_EVAL}" == "1" ]]; then
    cmd+=(--no-eval)
fi

srun --nodes=1 --ntasks=1 --cpu-bind=none "${cmd[@]}"

echo ""
echo "Neuro-symbolic training completed at $(date)"

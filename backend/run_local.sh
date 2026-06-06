#!/bin/bash
# ============================================================
# run_local.sh — Run the KnoCLIP-XAI inference API on a GPU machine WITHOUT Slurm.
#
# For the client's own NVIDIA GPU box (or any GPU VM). No conda, no mamba, no Slurm.
# Creates a local .venv, installs the CUDA torch stack + serve deps, then starts the API
# on http://localhost:8000. (On Hydra, use sbatch scripts/hydra/serve_app.sh instead.)
#
# Usage (from the repo root):
#   MODEL_CHECKPOINT=/abs/path/best_model.pt \
#   KG_ARTIFACTS_DIR=/abs/path/outputs/KG \
#     bash backend/run_local.sh
#
# Required:
#   MODEL_CHECKPOINT   path to best_model.pt
# Optional (sensible defaults):
#   MODEL_CONFIG       default: configs/hydra_phase2_neurosymbolic_gpt2_jpg_impression_clinical_v1.yaml
#   KG_ARTIFACTS_DIR   only needed for the KG Explorer card; predict works without it
#   DEVICE             default: cuda:0
#   PORT               default: 8000
#   HF_HOME            point at a pre-warmed HF cache for offline use
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Defaults ────────────────────────────────────────────────────────────────
export MODEL_CONFIG="${MODEL_CONFIG:-configs/hydra_phase2_neurosymbolic_gpt2_jpg_impression_clinical_v1.yaml}"
export DEVICE="${DEVICE:-cuda:0}"
export PORT="${PORT:-8000}"

if [[ -z "${MODEL_CHECKPOINT:-}" ]]; then
    echo "ERROR: MODEL_CHECKPOINT is required." >&2
    echo "  e.g. MODEL_CHECKPOINT=/abs/path/best_model.pt bash backend/run_local.sh" >&2
    exit 1
fi

# Resolve relative paths against the repo root.
[[ "${MODEL_CONFIG}" != /* ]] && export MODEL_CONFIG="${REPO_ROOT}/${MODEL_CONFIG}"
[[ "${MODEL_CHECKPOINT}" != /* ]] && export MODEL_CHECKPOINT="${REPO_ROOT}/${MODEL_CHECKPOINT}"

if [[ ! -f "${MODEL_CONFIG}" ]]; then
    echo "ERROR: MODEL_CONFIG not found: ${MODEL_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_CHECKPOINT}" ]]; then
    echo "ERROR: MODEL_CHECKPOINT not found: ${MODEL_CHECKPOINT}" >&2
    exit 1
fi

# ── Virtual environment ─────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
PY="${VENV_DIR}/bin/python"

# ── Dependencies (idempotent; skip when torch already importable) ───────────
if ! "${PY}" -c "import torch, fastapi, transformers" >/dev/null 2>&1; then
    echo "Installing CUDA torch + serve dependencies (first run only)..."
    "${PY}" -m pip install -q --upgrade pip
    # Default PyPI torch wheels are CUDA-enabled.
    "${PY}" -m pip install -q torch==2.5.1 torchvision==0.20.1
    "${PY}" -m pip install -q -r "${REPO_ROOT}/backend/requirements-serve.txt"
fi

if ! "${PY}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "WARNING: torch.cuda.is_available() is False — no GPU visible. Inference will be very slow." >&2
    echo "         Check the NVIDIA driver, or set DEVICE=cpu to force CPU." >&2
fi

echo ""
echo "MODEL_CONFIG     : ${MODEL_CONFIG}"
echo "MODEL_CHECKPOINT : ${MODEL_CHECKPOINT}"
echo "KG_ARTIFACTS_DIR : ${KG_ARTIFACTS_DIR:-<not set — KG Explorer disabled>}"
echo "DEVICE           : ${DEVICE}"
echo "PORT             : ${PORT}"
echo ""
echo "Open frontend/index.html in your browser; API URL is http://localhost:${PORT}"
echo ""

# start.sh execs uvicorn with the env we just set.
exec bash "${REPO_ROOT}/backend/start.sh"

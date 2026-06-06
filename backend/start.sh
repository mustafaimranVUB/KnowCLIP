#!/bin/bash
# Start the KnoCLIP-XAI inference API
# Usage: bash backend/start.sh
#
# Required env vars:
#   MODEL_CONFIG     - path to YAML config (e.g. configs/hydra_phase2_neurosymbolic_gpt2_jpg.yaml)
#   MODEL_CHECKPOINT - path to best_model.pt
#
# Optional env vars:
#   KG_ARTIFACTS_DIR - path to KG artifacts dir (required for neuro-symbolic models)
#   HOST             - bind host (default: 0.0.0.0)
#   PORT             - bind port (default: 8000)
#   DEVICE           - cpu/cuda:0 (default: auto-detect)
#   LOG_LEVEL        - info/debug/warning (default: info)

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (parent directory of this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Validate required env vars
# ---------------------------------------------------------------------------
if [[ -z "${MODEL_CONFIG:-}" ]]; then
    echo "ERROR: MODEL_CONFIG env var is required but not set." >&2
    echo "  Example: export MODEL_CONFIG=configs/hydra_phase2_neurosymbolic_gpt2_jpg.yaml" >&2
    exit 1
fi

if [[ -z "${MODEL_CHECKPOINT:-}" ]]; then
    echo "ERROR: MODEL_CHECKPOINT env var is required but not set." >&2
    echo "  Example: export MODEL_CHECKPOINT=outputs/checkpoints/neurosymbolic_gpt2_hydra_jpg/best_model.pt" >&2
    exit 1
fi

# Resolve paths relative to repo root when they are not absolute
if [[ "${MODEL_CONFIG}" != /* ]]; then
    MODEL_CONFIG="${REPO_ROOT}/${MODEL_CONFIG}"
fi
if [[ "${MODEL_CHECKPOINT}" != /* ]]; then
    MODEL_CHECKPOINT="${REPO_ROOT}/${MODEL_CHECKPOINT}"
fi

if [[ ! -f "${MODEL_CONFIG}" ]]; then
    echo "ERROR: MODEL_CONFIG file not found: ${MODEL_CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_CHECKPOINT}" ]]; then
    echo "ERROR: MODEL_CHECKPOINT file not found: ${MODEL_CHECKPOINT}" >&2
    exit 1
fi

export MODEL_CONFIG
export MODEL_CHECKPOINT

# ---------------------------------------------------------------------------
# Optional: activate venv
# ---------------------------------------------------------------------------
VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
    echo "Activating virtual environment: ${VENV_ACTIVATE}"
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
fi

# ---------------------------------------------------------------------------
# Defaults for optional env vars
# ---------------------------------------------------------------------------
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# ---------------------------------------------------------------------------
# Change to repo root so that relative imports and paths resolve correctly
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"

echo "Starting KnoCLIP-XAI API..."
echo "  Config:     ${MODEL_CONFIG}"
echo "  Checkpoint: ${MODEL_CHECKPOINT}"
echo "  KG dir:     ${KG_ARTIFACTS_DIR:-<not set>}"
echo "  Device:     ${DEVICE:-auto}"
echo "  Host:       ${HOST}"
echo "  Port:       ${PORT}"
echo "  Log level:  ${LOG_LEVEL}"

exec uvicorn backend.app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level "${LOG_LEVEL}"

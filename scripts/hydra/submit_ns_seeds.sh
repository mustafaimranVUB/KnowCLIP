#!/bin/bash
# Submit the same neuro-symbolic configuration across multiple seeds.
# Usage: bash scripts/hydra/submit_ns_seeds.sh [config_path]
set -euo pipefail

CONFIG_PATH="${1:-configs/hydra_phase2_neurosymbolic_gpt2_jpg.yaml}"
SEEDS=(42 123 2026)

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "Submitting multi-seed neuro-symbolic runs"
echo "Config: ${CONFIG_PATH}"

for SEED in "${SEEDS[@]}"; do
  echo "Submitting seed ${SEED}"
  sbatch --export=ALL,CONFIG_PATH="${CONFIG_PATH}",SEED="${SEED}" scripts/hydra/train_neurosymbolic.sh
  sleep 1
done

echo "Done. Track jobs with: squeue -u $USER"

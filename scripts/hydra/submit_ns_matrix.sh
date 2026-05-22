#!/bin/bash
# Submit the currently maintained neuro-symbolic experiment matrix.
# Usage: bash scripts/hydra/submit_ns_matrix.sh
set -euo pipefail

require_config() {
  local config_path="$1"
  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config: ${config_path}" >&2
    exit 1
  fi
}

CONFIGS=(
  "configs/phase2_neurosymbolic_gpt2.yaml"
  "configs/hydra_phase2_neurosymbolic_gpt2_jpg.yaml"
)

echo "Submitting neuro-symbolic matrix (${#CONFIGS[@]} configs)"
for CFG in "${CONFIGS[@]}"; do
  require_config "${CFG}"
  echo "Submitting config: ${CFG}"
  sbatch --export=ALL,CONFIG_PATH="${CFG}",SEED=42 scripts/hydra/train_neurosymbolic.sh
  sleep 1
done

echo "Done. Track jobs with: squeue -u $USER"

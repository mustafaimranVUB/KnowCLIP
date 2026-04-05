#!/bin/bash
# Submit the neuro-symbolic experiment matrix (E1, E2, E3).
# Usage: bash scripts/hydra/submit_ns_matrix.sh
set -euo pipefail

CONFIGS=(
  "configs/phase2_neurosymbolic.yaml"
  "configs/phase2_neurosymbolic_e2.yaml"
  "configs/phase2_neurosymbolic_e3.yaml"
)

echo "Submitting neuro-symbolic matrix (${#CONFIGS[@]} configs)"
for CFG in "${CONFIGS[@]}"; do
  echo "Submitting config: ${CFG}"
  sbatch --export=ALL,CONFIG_PATH="${CFG}",SEED=42 scripts/hydra/train_neurosymbolic.sh
  sleep 1
done

echo "Done. Track jobs with: squeue -u $USER"

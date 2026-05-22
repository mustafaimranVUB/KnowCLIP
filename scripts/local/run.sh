#!/bin/bash
# ============================================================
# Local development runner
# Usage: bash scripts/local/run.sh [command] [args...]
# Examples:
#   bash scripts/local/run.sh validate
#   bash scripts/local/run.sh phase1 --config configs/phase1_kg_local.yaml
#   bash scripts/local/run.sh train --config configs/hydra_phase2_baseline_jpg.yaml
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

# Defaults for local dev
export MIMIC_ROOT="${MIMIC_ROOT:-data/mimic-cxr}"
export MIMIC_REPORTS="${MIMIC_REPORTS:-data/mimic-cxr-reports/files}"

echo "=== KnoCLIP-XAI (local) ==="
echo "Project: $PROJECT_DIR"
echo "Python:  $(python --version)"
echo "Command: $@"
echo ""

python main.py "$@"

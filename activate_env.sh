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
export TORCH_HOME="$VSC_SCRATCH/torch_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ── Working directory ─────────────────────────────────────────────────────────
cd "$VSC_DATA/thesis"
mkdir -p outputs/KG outputs/logs

echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python  : $(which python)"
echo ""

# ── Run Phase I ───────────────────────────────────────────────────────────────
# --batch-size 64  : safe for A100 with modern-radgraph-xl + SapBERT
# Remove --max-studies to process the full dataset.
# Add --skip-extraction / --skip-grounding / --skip-embedding to resume
# from a checkpoint if the job is restarted.
srun python -m  main.py \
    --config configs/phase1_kg_gpu.yaml \
    --device cuda \
    phase1 \
    --batch-size 64 \

echo ""
echo "=== Phase I finished: $(date) ==="

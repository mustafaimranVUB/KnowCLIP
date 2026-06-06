# syntax=docker/dockerfile:1
# ============================================================
# KnoCLIP-XAI inference API — GPU (CUDA) image.
#
# The most portable way to run the backend on any NVIDIA GPU machine (the client's box or
# a cloud GPU VM): one image, identical everywhere, driven by the host's NVIDIA driver via
# the NVIDIA Container Toolkit.
#
# Host prerequisites: NVIDIA driver + nvidia-container-toolkit installed
# (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
#
# Build (from the REPO ROOT, so the context includes src/, configs/, frontend/):
#   docker build -f backend/Dockerfile -t knoclip-api:gpu .
#
# Run (note --gpus all):
#   docker run --rm --gpus all -p 8000:8000 \
#     -e MODEL_CONFIG=configs/hydra_phase2_neurosymbolic_gpt2_jpg_impression_clinical_v1.yaml \
#     -e MODEL_CHECKPOINT=/models/best_model.pt \
#     -e KG_ARTIFACTS_DIR=/kg \
#     -e DEVICE=cuda:0 \
#     -v /abs/path/best_model.pt:/models/best_model.pt:ro \
#     -v /abs/path/outputs/KG:/kg:ro \
#     -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
#     knoclip-api:gpu
#
# The checkpoint and KG artifacts are NOT baked into the image — they are mounted at run
# time, so the same image works for any checkpoint.
# ============================================================

# Base image ships torch==2.5.1 + torchvision built against CUDA 12.1 (matches the pinned
# training stack), so no manual CUDA wheel juggling.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# System libs needed by Pillow / scikit-image image decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 \
    && apt-get clean

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# torch/torchvision already present from the base image; install only the serve-only deps.
COPY backend/requirements-serve.txt /app/backend/requirements-serve.txt
RUN pip install --no-cache-dir -r backend/requirements-serve.txt

# Copy the code needed to serve (build context is the repo root).
COPY src/ /app/src/
COPY backend/ /app/backend/
COPY kg_viewer/ /app/kg_viewer/
COPY configs/ /app/configs/
COPY main.py /app/main.py

EXPOSE 8000

ENV DEVICE=cuda:0 \
    HOST=0.0.0.0 \
    PORT=8000

# Start the API directly (no Slurm/conda); start.sh validates MODEL_CONFIG/MODEL_CHECKPOINT.
CMD ["bash", "backend/start.sh"]

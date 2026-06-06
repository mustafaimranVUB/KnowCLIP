# KnoCLIP-XAI — Inference API Backend

FastAPI server that wraps the `InferencePipeline` as a real-time REST API.  
The model is loaded **once** at startup and reused for every request — no per-request reload overhead.

---

## Prerequisites

- A trained checkpoint (`best_model.pt`)
- Python deps. On a GPU machine use the minimal serve set (no scispaCy/RadGraph):
  ```bash
  pip install torch==2.5.1 torchvision==0.20.1        # default PyPI wheels are CUDA-enabled
  pip install -r backend/requirements-serve.txt
  ```
  (Inside the full project env, the root `requirements.txt` already covers everything.)
- Phase I KG artifacts (`outputs/KG/`) are **optional** — only the KG Explorer card needs
  them; `/predict` works without them.

> For full deploy instructions (Docker/venv on a client GPU box, and Slurm on Hydra) see
> `documents/WEBAPP_RUNBOOK.md`. The easiest client path is `bash backend/run_local.sh`.

---

## Quick Start

```bash
# 1. Set required env vars
export MODEL_CONFIG=configs/hydra_phase2_neurosymbolic_gpt2_jpg_impression_clinical_v1.yaml
export MODEL_CHECKPOINT=/abs/path/to/outputs/checkpoints/ablation_genw025/best_model.pt
export KG_ARTIFACTS_DIR=/abs/path/to/outputs/KG   # optional: only for the KG Explorer card
export DEVICE=cuda:0                               # or leave unset to auto-detect

# 2. Start the server
bash backend/start.sh
```

The server binds to `0.0.0.0:8000` by default.  
Interactive Swagger UI: **http://localhost:8000/docs**

---

## Configuration

Copy `backend/config_example.env` to `.env` and edit as needed:

```bash
cp backend/config_example.env .env
# edit .env
set -a && source .env && set +a   # load into current shell
bash backend/start.sh
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_CONFIG` | **yes** | — | Path to YAML config file |
| `MODEL_CHECKPOINT` | **yes** | — | Path to `best_model.pt` |
| `KG_ARTIFACTS_DIR` | no | — | Phase I KG outputs; only enables the KG Explorer (predict works without it) |
| `HOST` | no | `0.0.0.0` | Bind host |
| `PORT` | no | `8000` | Bind port |
| `DEVICE` | no | auto | `cpu`, `cuda:0`, etc. |
| `LOG_LEVEL` | no | `info` | `debug` / `info` / `warning` |

---

## API Endpoints

### `GET /health`
Liveness probe. Returns `200` regardless of model state.

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda:0"
}
```

### `POST /predict`
Upload a chest X-ray and receive classification + report generation.

**Request** — `multipart/form-data`:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | **yes** | `.jpg` / `.jpeg` / `.png`, max 10 MB |
| `subject_id` | integer | no | MIMIC-CXR subject ID — KG Explorer lookup only; **does not affect inference** |
| `study_id` | integer | no | MIMIC-CXR study ID — KG Explorer lookup only; **does not affect inference** |
| `save_explainability` | boolean | no | Also run explainability export (slower) |

**Response** `200 OK`:
```json
{
  "classification": {
    "No Finding": 0.03,
    "Cardiomegaly": 0.91,
    "Pleural Effusion": 0.78,
    ...
  },
  "generated_report": "There is moderate cardiomegaly with bilateral pleural effusions.",
  "explainability": null,
  "image_filename": "chest_xray.jpg",
  "processing_time_ms": 312.5
}
```

**Error responses**:
- `422` — invalid file type or file too large
- `503` — model not loaded (check `/health`)
- `500` — inference error (see server log)

### `GET /classes`
Returns the 14 CheXpert class names.

### `GET /config`
Returns non-sensitive model configuration (backbone, decoder type, image size, etc.).

---

## Deployment Notes

### Local (development)
```bash
bash backend/start.sh
```

### Systemd service (production)
Create `/etc/systemd/system/knoclip-api.service`:
```ini
[Unit]
Description=KnoCLIP-XAI Inference API
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/repo
EnvironmentFile=/path/to/repo/.env
ExecStart=/path/to/repo/.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable knoclip-api
sudo systemctl start knoclip-api
```

### Docker (GPU, recommended for client machines)
A production-ready GPU image is provided — do **not** hand-roll one from the full
`requirements.txt` (it pulls in heavy Phase-I-only deps like scispaCy/RadGraph that the
server never imports). Host needs the NVIDIA driver + nvidia-container-toolkit. Build from
the repo root:
```bash
docker build -f backend/Dockerfile -t knoclip-api:gpu .
docker run --rm --gpus all -p 8000:8000 \
  -e MODEL_CONFIG=configs/hydra_phase2_neurosymbolic_gpt2_jpg_impression_clinical_v1.yaml \
  -e MODEL_CHECKPOINT=/models/best_model.pt -e DEVICE=cuda:0 \
  -v /abs/path/best_model.pt:/models/best_model.pt:ro \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  knoclip-api:gpu
```
See `documents/WEBAPP_RUNBOOK.md` (Option A) and `backend/docker-compose.yml` for details.

---

## Connecting the Frontend

Open `frontend/index.html` in a browser. Set the API URL to `http://localhost:8000` (or your server address). The frontend has no CORS restrictions from the server side — CORS is already enabled for all origins in `backend/app.py`.

# KnoCLIP-XAI — Web Frontend

A standalone, zero-installation web interface for the KnoCLIP-XAI radiology AI system. Works directly in any modern browser — no Node.js, no npm, no build step required.

## How to use

1. **Start the backend** on a GPU machine. Easiest on the client's GPU box: `bash backend/run_local.sh` (or Docker — see `documents/WEBAPP_RUNBOOK.md`). The API must be running before you can analyze images.

2. **Open `index.html`** in your browser:
   - Double-click the file in your file manager, or
   - Drag it into an open browser window, or
   - From a terminal: `open frontend/index.html` (macOS) / `xdg-open frontend/index.html` (Linux)

3. **Configure the API URL** (top panel):
   - The default `http://localhost:8000` is correct if the backend is running locally.
   - Click **Test Connection** to verify the backend is reachable and the model is loaded.

4. **Upload a chest X-ray**:
   - Drag and drop a `.jpg`, `.jpeg`, or `.png` image onto the upload zone, or click to browse.
   - **Subject ID / Study ID are optional and do not affect the prediction** — they only let the Knowledge-Graph Explorer load that MIMIC-CXR study's graph. Leave them blank for any other image.
   - Tick **Include explainability analysis** to generate GradCAM / LIME outputs (slower).

5. **Click Analyze** and wait for results:
   - **Note:** classification is coloured by the model's **validation-tuned per-class thresholds** (a class is positive only when prob ≥ its threshold), not a flat 50%. The free-text report is best with a real Subject/Study ID (KG grounding); without one it can be fluent but ungrounded.
   - **Classification tab** — 14 CheXpert class probabilities, colour-coded by severity (red > 50 %, amber 20–50 %, green < 20 %).
   - **Generated Report tab** — free-text radiology report produced by the language model. Use **Copy to clipboard** to paste it elsewhere.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| A running KnoCLIP-XAI backend | See [`backend/start.sh`](../backend/start.sh) and [`backend/config_example.env`](../backend/config_example.env) |
| A modern browser | Chrome 90+, Firefox 90+, Safari 15+, Edge 90+ |
| A chest X-ray image | JPEG or PNG, max 10 MB |

## No installation required

`frontend/index.html` is a single self-contained file. All styles and logic are embedded — no internet connection, no package manager, and no server needed to serve the HTML itself.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "Connection failed" | Backend not running or wrong port | Start backend; check API URL |
| "Model not loaded" badge | Backend started but model still loading | Wait ~30 s and test again |
| Blank report | Model loaded but report generation disabled in config | Check `enable_report_generation` in your YAML config |
| CORS error in browser console | Backend CORS settings | The backend already allows all origins by default |

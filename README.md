# KnoCLIP-XAI

## Self-Explainable Vision–Language Framework for Medical Imaging

> Master's Thesis — Vrije Universiteit Brussel (VUB)
> Compute: Hydra HPC Cluster (VSC Tier-2) | Dataset: MIMIC-CXR (PhysioNet) | Ontology: UMLS 2025AB

---

## Overview

This repository implements **KnoCLIP-XAI**, a knowledge-graph-infused, CLIP-based
Vision-Language Model (VLM) for **self-explainable** chest X-ray analysis. The framework is
designed as a two-phase neuro-symbolic pipeline:

- **Phase I** — Knowledge Graph Construction from radiology reports
- **Phase II** — Multi-task model training (pathology classification + report generation)

Core capabilities:
- **Self-explainable**: explanations are inherent to the model's decision process via
  cross-attention over medical concepts, not produced post-hoc
- **Ontology-grounded**: all medical entities are anchored to UMLS CUI codes via a
  hybrid knowledge graph
- **Report-generating**: produces clinically grounded radiology reports conditioned on
  visual and knowledge features
- **Statistically rigorous**: baseline vs. neuro-symbolic comparison with McNemar's test,
  bootstrap CIs, and Bonferroni correction

---

## Research Objectives

1. Integrate **prior medical knowledge** (ontologies + KGs) into representation learning
2. Embed knowledge representations directly into model architectures
3. Associate model activations with **explicit medical concepts**
4. Generate **clinically interpretable and explainable medical reports**
5. Provide rigorous **quantitative and qualitative evaluation** of explainability

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │       Radiology Report        │
                    └──────────────┬───────────────┘
                                   │ Phase I
                    ┌──────────────▼───────────────┐
                    │   RadGraph-XL Entity Extraction│
                    │   UMLS 2025AB Grounding       │
                    │   SapBERT Node Embedding      │
                    │   PyG Knowledge Graph (.pt)   │
                    └──────────────┬───────────────┘
                                   │
        ┌──────────────────────────┼───────────────────────┐
        │                          │                       │
  ┌───────────┐           ┌──────────────┐                 │
  │ Chest     │           │  Knowledge   │                 │
  │ X-Ray     │           │  Graph (.pt) │                 │
  │ (JPG/PNG) │           │              │                 │
  └─────┬─────┘           └──────┬───────┘                 │
        │                        │                         │
        ▼                        ▼                         │
  ┌───────────┐           ┌──────────────┐                 │
  │ BioMedCLIP│           │   GATv2      │                 │
  │ ViT-B/16  │           │   Encoder    │                 │
  │ (E_V)     │           │   (E_K)      │                 │
  └─────┬─────┘           └──────┬───────┘                 │
        │ Z_v (B,196,768)        │ Z_k (B,N,768)          │
        │                        │                         │
        └────────┬───────────────┘                         │
                 ▼                                         │
        ┌──────────────┐                                   │
        │Cross-Attention│                                  │
        │   Fusion      │                                  │
        └───────┬──────┘                                   │
                │ Z_fused (B,N,768)                        │
                │                                          │
        ┌───────┴────────────────────┐                     │
        ▼                            ▼                     │
  ┌───────────┐            ┌──────────────┐                │
  │ Self-Attn │            │ GPT-2 Report │                │
  │ Pooling + │            │  Decoder +   │◄───────────────┘
  │ Classif.  │            │  xattn       │ (teacher forcing)
  │ Head      │            │              │
  └─────┬─────┘            └──────┬───────┘
        │                         │
        ▼                         ▼
  14-class logits          Report tokens
  (CheXpert)
```

### Model Variants

| Configuration | Knowledge Graph | Visual Encoder | Outputs |
|--------------|----------------|---------------|---------|
| **Baseline** | No | BioMedCLIP ViT-B/16 | 14-class CheXpert classification |
| **Neuro-Symbolic** | Yes (GATv2 + Fusion) | BioMedCLIP ViT-B/16 | Classification + Report Generation |

---

## Dataset: MIMIC-CXR (PhysioNet)

This project uses the **MIMIC-CXR v2.0.0** dataset from [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/), **not** a HuggingFace proxy dataset. Access requires:

1. **PhysioNet Credentialed Access** — [apply here](https://physionet.org/settings/credentialing/)
2. **CITI Training** — Data or Specimens Only Research
3. **Data Use Agreement** — signed for MIMIC-CXR

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total images | 377,110 DICOM chest X-rays |
| Total studies | 227,835 radiographic studies |
| Total patients | 65,379 |
| Full DICOM size | ~4.7 TB |
| Report structure | Impression, Findings, Indication, Technique sections |
| Labels | 14-class CheXpert (from `mimic-cxr-2.0.0-chexpert.csv`) |
| Splits | Patient-level train/validate/test (from `mimic-cxr-2.0.0-split.csv`) |

### Storage Strategy

The full 4.7 TB DICOM dataset is stored on the Hydra HPC cluster (see [Infrastructure](#infrastructure)). Locally, a **subset of radiology reports** is used for Phase I knowledge graph development:

| Content | Location | Description |
|---------|----------|-------------|
| Reports (subset) | `data/MIMIC-CXR-RRG_reports/files/` | ~12,968 patients (p10 + p11 prefixes) |
| Record list | `data/MIMIC-CXR-RRG_reports/cxr-record-list.csv` | Full 377K record mapping |
| UMLS Metathesaurus | `data/umls-2025AB-metathesaurus-full/2025AB/META/` | MRCONSO.RRF (2.1 GB) + ancillary files |
| DICOM images (full) | `$VSC_SCRATCH/mimic-cxr/` (HPC only) | Full dataset for training |

> **Important**: The radiology report text files come directly from PhysioNet's `mimic-cxr-reports.zip`, not from any third-party source. While only p10 and p11 patient reports are available locally (~12,968 patients), the `cxr-record-list.csv` covers all 377,110 records across the full dataset. Phase I knowledge graph construction uses the locally available report subset for development, with full-dataset processing intended for the HPC cluster.

### Report Structure

Reports are stored as plain text files: `files/p{prefix}/p{subject_id}/s{study_id}.txt`

Each report may contain sections: **FINDINGS** (detailed observations), **IMPRESSION** (summary/conclusion), INDICATION, TECHNIQUE, COMPARISON. The pipeline prioritizes the **Impression** section and falls back to **Findings** if Impression is empty.

---

## Quick Start

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **PhysioNet credentialed access** to MIMIC-CXR
- **UMLS license** for the Metathesaurus (free account at [UTS](https://uts.nlm.nih.gov/uts/))
- **GPU** recommended for RadGraph extraction and model training (Phase I can run on CPU)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Clone\ repo
```

2. **Set up virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

For **local development** (CPU-only):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.txt
```

For **HPC/GPU**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install -r requirements.txt
```

> **Critical**: `transformers==4.57.3` is pinned because the RadGraph model (`StanfordAIMI/modern-radgraph-xl`) requires this exact version. Do not upgrade.

4. **Verify installation**
```bash
python -m main validate
```

Expected output:
```
VALIDATION MODE — running on CPU with dummy data
[1/7] Checking imports …        ✓ All imports succeeded.
[2/7] Baseline model forward …  ✓ Baseline forward pass OK.
[3/7] Neuro-symbolic forward …  ✓ Neuro-symbolic forward pass OK.
[4/7] Graph builder …           ✓ Graph builder OK.
[5/7] Loss computation …        ✓ Loss OK.
[6/7] Classification metrics …  ✓ Classification metrics OK.
[7/7] Generation metrics …      ✓ Generation metrics OK.
ALL VALIDATIONS PASSED ✓
```

### Data Setup

#### MIMIC-CXR Reports (Required for Phase I)

Download the radiology reports from PhysioNet (requires credentialed access):

```bash
# Option 1: Download reports archive from PhysioNet
wget -r -N -c -np --user YOUR_PHYSIONET_USER --ask-password \
    https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip

# Extract to data directory
unzip mimic-cxr-reports.zip -d data/MIMIC-CXR-RRG_reports/

# Also download the record list
wget --user YOUR_PHYSIONET_USER --ask-password \
    https://physionet.org/files/mimic-cxr/2.0.0/cxr-record-list.csv.gz \
    -O data/MIMIC-CXR-RRG_reports/cxr-record-list.csv.gz
gunzip data/MIMIC-CXR-RRG_reports/cxr-record-list.csv.gz
```

#### MIMIC-CXR DICOM Images (Required for Phase II)

On the HPC cluster, download the full DICOM dataset:

```bash
# On Hydra HPC — download to scratch storage
wget -r -N -c -np --user YOUR_PHYSIONET_USER --ask-password \
    https://physionet.org/files/mimic-cxr/2.0.0/ \
    -P $VSC_SCRATCH/mimic-cxr/
```

Or use the provided download script:
```bash
bash scripts/hydra/download_physionet.sh
```

#### UMLS Metathesaurus 2025AB (Required for Phase I)

1. Go to [UMLS Downloads](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html)
2. Download the **2025AB Full Release** (requires free UMLS account)
3. Extract and place under `data/umls-2025AB-metathesaurus-full/2025AB/META/`

Key files needed:

| File | Purpose | Size |
|------|---------|------|
| `MRCONSO.RRF` | Concept names and sources (entity → CUI mapping) | ~2.1 GB |
| `MRSTY.RRF` | Semantic types per CUI (type validation) | ~5 MB |
| `MRREL.RRF` | Concept relationships (ontology edges) | ~320 MB |

> **Note**: Get your UMLS API key from [UTS](https://uts.nlm.nih.gov/uts/) (free account). Store it in `.env`:
> ```bash
> echo "UMLS_API_KEY=your_key_here" >> .env
> ```

---

## Usage

### Local verification first

```bash
python -m main validate
python -m pytest tests/ -q
```

Current verified local result in this workspace:
- `116 passed, 3 skipped, 3 warnings`
- `python -m main validate` completed successfully on CPU

### Phase I — Knowledge graph construction

```bash
# Local smoke test / development run
python -m main --config configs/phase1_kg_local.yaml phase1 --max-studies 100

# Hydra full run
sbatch scripts/hydra/run_phase1.sh
```

Primary Phase I outputs in `outputs/KG/`:
- `extractions.pt`
- `grounding.pt`
- `embeddings.pt`
- `report_graphs.pt`
- `global_kg.pt`
- `grounding_summary.csv`
- `study_metadata.pt`
- `phase1_summary.json`

### Phase II — Training

```bash
# Baseline
python -m main --config configs/phase2_baseline.yaml train

# Main local/subset neuro-symbolic GPT-2 run
python -m main --config configs/phase2_neurosymbolic_gpt2.yaml train

# Hydra baseline
sbatch scripts/hydra/train_baseline.sh

# Hydra main neuro-symbolic run
sbatch scripts/hydra/train_neurosymbolic.sh
```

### Evaluation, explainability, audit, and comparison

```bash
# Evaluate a checkpoint and export metrics/plots
python -m main --config configs/phase2_neurosymbolic_gpt2.yaml evaluate \
  --checkpoint outputs/checkpoints/best_model.pt \
  --output-dir outputs/evaluation/manual_run

# Export sample-level explainability figures
python -m main --config configs/phase2_neurosymbolic_gpt2.yaml explain \
  --checkpoint outputs/checkpoints/best_model.pt \
  --split test \
  --max-samples 8 \
  --output-dir outputs/explainability/manual_run

# Audit split coverage and report distributions
python -m main --config configs/phase2_neurosymbolic_gpt2.yaml audit-data \
  --output-dir outputs/data_audit/manual_run

# Compare two evaluated runs
python -m main compare \
  --eval-a outputs/evaluation/baseline_run \
  --eval-b outputs/evaluation/ns_run \
  --label-a baseline \
  --label-b ns \
  --output-dir outputs/comparisons/baseline_vs_ns
```

### Single-image inference

```bash
python -m main --config configs/hydra_phase2_neurosymbolic_gpt2_jpg.yaml predict \
  --checkpoint outputs/checkpoints/best_model.pt \
  --image-path /absolute/path/to/image.jpg \
  --output-dir outputs/predict/manual_run \
  --save-explainability
```

### Knowledge graph viewer

```bash
python scripts/visualize_kg.py --artifact-dir outputs/KG --port 8502
```

---

## Real-Time Inference Web Application

A lightweight backend + frontend lets supervisors and collaborators run inference interactively without touching the command line.

### Backend (FastAPI)

```bash
# Required env vars
export MODEL_CONFIG=configs/hydra_phase2_neurosymbolic_gpt2_jpg.yaml
export MODEL_CHECKPOINT=outputs/checkpoints/neurosymbolic_gpt2_hydra_jpg/best_model.pt
export KG_ARTIFACTS_DIR=outputs/KG          # only for neuro-symbolic configs

# Start the API server (model loads once, all requests reuse it)
bash backend/start.sh
```

The server starts at `http://0.0.0.0:8000` by default.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness probe — returns model_loaded + device |
| `/predict` | POST | Upload a JPG/PNG and get classification + report |
| `/classes` | GET | List the 14 CheXpert class names |
| `/config` | GET | Non-sensitive model configuration info |

Full API documentation is auto-generated at `http://localhost:8000/docs` (Swagger UI).

Install backend extras before first launch:
```bash
pip install -r backend/requirements.txt
```

### Frontend

No installation required — open the HTML file in any modern browser:

```bash
# Option 1 — open directly
xdg-open frontend/index.html   # Linux
open frontend/index.html       # macOS

# Option 2 — serve (optional, avoids any browser CORS quirks)
python -m http.server 3000 --directory frontend
# then open http://localhost:3000
```

The UI lets you:
- Configure the backend API URL (defaults to `http://localhost:8000`)
- Test the connection with a single click
- Drag-and-drop or browse for a chest X-ray (JPG/PNG)
- Run analysis and see classification probabilities + the generated report
- Optionally request explainability artefacts

---

## Model Selection (Hydra — run after training)

After training all runs on Hydra, submit evaluation for all configurations at once:

```bash
# From the Hydra login node, inside the repo directory
bash scripts/hydra/submit_full_comparison.sh
```

This submits one evaluation job per model (9 total): impression, findings, and all ablations.

Once all eval jobs are complete, collect results locally:

```bash
# Rank all models by macro AUC-ROC (primary metric)
python scripts/local/select_best_model.py \
  --eval-dirs \
    outputs/evaluation/ns_impression_s1full/best_model \
    outputs/evaluation/ns_findings_s1full/best_model \
    outputs/evaluation/ablation_genw025/best_model \
    outputs/evaluation/ablation_gat1/best_model \
    outputs/evaluation/ablation_gat3/best_model \
    outputs/evaluation/ablation_no_kg_gpt2/best_model \
    outputs/evaluation/baseline_hydra_jpg/best_model \
  --primary-metric macro_auc \
  --output-json outputs/comparisons/ranking.json

# Generate a full Markdown comparison report
python scripts/local/model_comparison_report.py \
  --eval-dirs outputs/evaluation/*/best_model \
  --output-md outputs/comparisons/model_comparison_report.md
```

---

## Project Structure

```
Clone repo/
├── main.py                          # CLI entry point (phase1/train/evaluate/explain/audit-data/predict/compare/validate)
├── backend/                         # FastAPI real-time inference API
│   ├── app.py                       # REST endpoints (/health /predict /classes /config)
│   ├── start.sh                     # Launch script (reads MODEL_CONFIG, MODEL_CHECKPOINT env vars)
│   ├── requirements.txt             # fastapi + uvicorn + python-multipart
│   └── config_example.env           # Template env file
├── frontend/                        # Standalone web UI (no build step)
│   ├── index.html                   # Single-page app — open in any browser
│   └── README.md                    # Usage instructions
├── configs/
│   ├── phase1_kg.yaml
│   ├── phase1_kg_gpu.yaml
│   ├── phase1_kg_local.yaml
│   ├── phase2_baseline.yaml
│   ├── phase2_neurosymbolic.yaml
│   ├── phase2_neurosymbolic_gpt2.yaml
│   ├── hydra_phase2_baseline_jpg.yaml
│   ├── hydra_phase2_neurosymbolic_gpt2_jpg.yaml
│   ├── hydra_phase2_neurosymbolic_gpt2_jpg_findings.yaml
│   └── ablation_*.yaml
├── src/
│   ├── data/                        # dataset, splits, transforms, report/image loading
│   ├── knowledge/                   # extraction, grounding, embeddings, graph building
│   ├── models/                      # visual encoder, GATv2 encoder, fusion, classifier, decoder
│   ├── training/                    # losses, scheduler, trainer
│   ├── evaluation/                  # metrics, explainability, artifact export, statistics
│   └── pipelines/                   # training, explainability, inference, audit, comparison, phase1
├── scripts/
│   ├── hydra/                       # run_phase1, train/evaluate/explain launchers, ablation/comparison submitters
│   │   └── submit_full_comparison.sh # Submit eval jobs for all 9 model variants
│   ├── local/
│   │   ├── select_best_model.py     # Rank eval dirs and declare winner
│   │   └── model_comparison_report.py # Generate Markdown comparison report
│   └── visualize_kg.py              # KG viewer launcher
├── kg_viewer/                       # Streamlit KG viewer
├── tests/                           # 116 tests, 3 skipped in the current validated state
├── documents/                       # runbook, checklist, audits, methodology docs
└── .aiglobal/                       # AI-facing project guides and design notes
```

---

## Infrastructure

### Hydra HPC cluster

| Partition | Hardware | Typical use |
|-----------|----------|-------------|
| `ampere_gpu` | NVIDIA A100 40GB | Main Phase II training |
| `hopper_gpu` | NVIDIA H200 140GB | Large-batch experiments |
| `zen4` / `zen5` | CPU-only | Phase I and heavy preprocessing |

Storage conventions:

| Path | Purpose |
|------|---------|
| `$VSC_HOME` | Code, configs, lightweight scripts |
| `$VSC_DATA` | Persistent KG artifacts, checkpoints, environments |
| `$VSC_SCRATCH` | Large caches and dataset staging |

### Local environment

Local CPU execution is intended for code verification, dummy-data validation, synthetic explainability/export tests, and small subset runs.

---

## Testing

```bash
# Full suite
python -m pytest tests/ -q

# Focused runtime/launcher checks
python -m pytest tests/test_runtime_surfaces.py tests/test_hydra_scripts.py -q

# Focused figure/export checks
python -m pytest tests/test_explainability.py tests/test_evaluation.py tests/test_kg_viewer.py -q
```

Current validated status in this workspace:
- `116 passed, 3 skipped, 3 warnings`
- Hydra launcher regression tests pass
- Explainability exporter tests pass
- KG viewer tests pass

The skipped tests are network-dependent surfaces that require external model downloads.

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.x | Core ML framework |
| `transformers` | 4.57.3 (pinned) | RadGraph compatibility |
| `torch-geometric` | 2.7.0 | GATv2 knowledge encoder |
| `open-clip-torch` | current pinned repo version | BioMedCLIP backbone loading |
| `scikit-learn` | 1.x | Classification metrics |
| `rouge-score` | 0.1.x | ROUGE metrics |

---

## Configuration System

The repository uses dataclass defaults overlaid by YAML files.

```python
from src.core.config import load_config, get_baseline_config, get_neurosymbolic_config

config = load_config("configs/phase2_neurosymbolic_gpt2.yaml")
baseline = get_baseline_config()
neurosymbolic = get_neurosymbolic_config()
```

Important operational configs:
- `configs/phase2_baseline.yaml`: local/subset baseline
- `configs/phase2_neurosymbolic_gpt2.yaml`: main local/subset neuro-symbolic run
- `configs/hydra_phase2_baseline_jpg.yaml`: main Hydra baseline
- `configs/hydra_phase2_neurosymbolic_gpt2_jpg.yaml`: main Hydra neuro-symbolic run

---

## Canonical Documents

Start with these files:
- `documents/CURRENT_ARCHITECTURE_AND_RUNBOOK.md`
- `documents/THESIS_RESEARCH_CHECKLIST.md`
- `documents/EXPLAINABILITY_AND_FIGURE_AUDIT.md`
- `documents/DOCUMENTATION_AND_RUNBOOK_AUDIT.md`

These are the authoritative references for the current codebase. Older methodology and analysis documents in `documents/` should be treated as historical unless they explicitly state they are current.

---

## Troubleshooting

| Issue | Current guidance |
|-------|------------------|
| `transformers` mismatch | Keep `transformers==4.57.3` pinned for RadGraph compatibility |
| `torch_geometric` missing | Install the version from `requirements.txt` / `requirements-dev.txt` |
| OOM during training | Reduce `batch_size` or raise `accumulation_steps` |
| Missing UMLS files | Place 2025AB files under `data/umls-2025AB-metathesaurus-full/2025AB/META/` or the configured Hydra path |
| Missing optional generation metrics | `METEOR` needs NLTK `wordnet`; `BERTScore` needs `bert_score`; `CIDEr` needs `aac_metrics` |
| Hydra launcher mismatch | Use the maintained `scripts/hydra/*.sh` files; they are regression-tested by `tests/test_hydra_scripts.py` |

---

## References

### Key Papers

1. **RadGraph-XL**: Delbrouck et al., "RadGraph-XL: A Large-Scale Expert-Annotated Dataset for Entity and Relation Extraction from Radiology Reports", ACL 2024 Findings
2. **BioMedCLIP**: Zhang et al., "BiomedCLIP: A Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs", 2023
3. **GATv2**: Brody et al., "How Attentive are Graph Attention Networks?", ICLR 2022
4. **MIMIC-CXR**: Johnson et al., "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports", Scientific Data 2019
5. **SapBERT**: Liu et al., "Self-Alignment Pretraining for Biomedical Entity Representations", NAACL 2021
6. **F1-RadGraph**: Delbrouck et al., "Improving the Factual Correctness of Radiology Report Generation with Semantic Rewards", EMNLP 2022

### Datasets & Resources

| Resource | Source | Access |
|----------|--------|--------|
| MIMIC-CXR v2.0.0 | [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) | Credentialed |
| UMLS 2025AB Metathesaurus | [NLM](https://www.nlm.nih.gov/research/umls/) | Licensed (free) |
| CheXpert Labels | Bundled with MIMIC-CXR | Via `mimic-cxr-2.0.0-chexpert.csv` |

---

## License

See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **MIMIC-CXR**: Johnson et al., MIT Laboratory for Computational Physiology
- **RadGraph**: Stanford AIMI Center (Delbrouck et al.)
- **UMLS**: U.S. National Library of Medicine
- **BioMedCLIP**: Microsoft Research
- **SapBERT**: Cambridge Language Technology Lab
- **Hydra HPC**: VUB/ULB Computing Centre (VSC Tier-2)

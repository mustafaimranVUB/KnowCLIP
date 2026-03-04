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
  │ (DICOM)   │           │              │                 │
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
  │ Self-Attn │            │ Transformer  │                │
  │ Pooling + │            │   Report     │◄───────────────┘
  │ Classif.  │            │  Decoder     │ (teacher forcing)
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
python main.py validate
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

### Phase I — Knowledge Graph Construction

Phase I processes radiology reports to build a medical knowledge graph:

1. Loads report text (Impression → Findings fallback)
2. Extracts entities and relations via **RadGraph-XL** (ModernBERT backbone)
3. Filters out measurement entities
4. Grounds entities to **UMLS CUI codes** via exact match on MRCONSO.RRF
5. Embeds entity text with **SapBERT** (768-dim vectors)
6. Builds **PyTorch Geometric** graph objects (per-report + global)
7. Validates and saves all artifacts as `.pt` files

```bash
# Local (small run for development/testing)
python main.py phase1 --config configs/phase1_kg_local.yaml --max-studies 100

# Full run on HPC
sbatch scripts/hydra/run_phase1.sh
```

**Configuration** (`configs/phase1_kg.yaml`):
```yaml
kg_pipeline:
  mrconso_path: "${VSC_DATA}/umls/2025AB/META/MRCONSO.RRF"
  mrsty_path: "${VSC_DATA}/umls/2025AB/META/MRSTY.RRF"
  radgraph_model_type: "modern-radgraph-xl"
  top_k_candidates: 5
```

**Output** (`outputs/KG/`):

| Artifact | Format | Description |
|----------|--------|-------------|
| `extractions.pt` | `torch.save` | Per-report entities and triples from RadGraph-XL |
| `grounding.pt` | `torch.save` | Entity → UMLS CUI mapping results |
| `embeddings.pt` | `torch.save` | SapBERT 768-dim node embeddings |
| `report_graphs.pt` | `torch.save` | Per-study PyG `Data` objects |
| `global_kg.pt` | PyG `Data` | Merged global knowledge graph |
| `cui_coverage.csv` | CSV | Human-readable coverage statistics |
| `grounding_summary.csv` | CSV | Grounding quality metrics |
| `study_metadata.pt` | `torch.save` | Valid study list and metadata |
| `phase1_summary.json` | JSON | Pipeline run summary and validation results |

### Phase II — Model Training

#### Baseline Training (Classification Only)

```bash
# Local (CPU, for testing only)
python main.py train --config configs/phase2_baseline.yaml

# HPC (GPU)
sbatch scripts/hydra/train_baseline.sh
```

#### Neuro-Symbolic Training (Full KG + Report Generation)

```bash
# HPC (GPU)
sbatch scripts/hydra/train_neurosymbolic.sh
```

**Key config differences:**

| Parameter | Baseline | Neuro-Symbolic |
|-----------|----------|----------------|
| `use_kg` | `false` | `true` |
| `enable_report_generation` | `false` | `true` |
| `generation_loss_weight` | `0.0` | `1.0` |
| Knowledge encoder | None | GATv2 (2 layers, 4 heads) |
| Fusion module | None | Cross-attention (2 layers, 8 heads) |

### Evaluation

```bash
# Evaluate baseline
python main.py evaluate \
    --config configs/phase2_baseline.yaml \
    --checkpoint outputs/checkpoints/baseline/best_model.pt

# Evaluate neuro-symbolic
python main.py evaluate \
    --config configs/phase2_neurosymbolic.yaml \
    --checkpoint outputs/checkpoints/neurosymbolic/best_model.pt
```

**Metrics computed:**
- **Classification**: Per-class AUC-ROC, macro AUC-ROC, F1 at optimal threshold
- **Report Generation**: BLEU-1/2/4, ROUGE-1/2/L, BERTScore, F1-RadGraph
- **Statistical Tests**: McNemar's test (baseline vs NS), bootstrap 95% CI, Bonferroni correction

### Quick Validation (No Data Required)

```bash
python main.py validate
```

Runs 7 checks with dummy data on CPU — verifies imports, model forward passes, graph construction, loss computation, and metrics.

---

## Project Structure

```
Clone repo/
├── main.py                          # CLI entry point (phase1/train/evaluate/validate)
├── requirements.txt                 # Pinned production dependencies
├── requirements-dev.txt             # Dev/test dependencies
├── README.md                        # This file
├── LICENSE
│
├── configs/
│   ├── phase1_kg.yaml               # Phase I — KG pipeline (HPC paths)
│   ├── phase1_kg_local.yaml         # Phase I — KG pipeline (local paths)
│   ├── phase1_kg_gpu.yaml           # Phase I — KG pipeline (GPU partition)
│   ├── phase2_baseline.yaml         # Phase II — baseline training
│   └── phase2_neurosymbolic.yaml    # Phase II — neuro-symbolic training
│
├── src/
│   ├── __init__.py                  # Package root (version 0.2.0)
│   ├── core/
│   │   ├── config.py                # Dataclass configs + YAML overlay system
│   │   └── utils.py                 # Seed, device, logging utilities
│   ├── data/
│   │   ├── dataset.py               # MIMICCXRDataset + collate_mimic
│   │   ├── dicom_loader.py          # DICOM → preprocessed tensor
│   │   ├── report_loader.py         # Report text extraction and section parsing
│   │   ├── splits.py                # Patient-level data splitting
│   │   └── transforms.py            # Train/eval image augmentations
│   ├── knowledge/
│   │   ├── extraction.py            # RadGraph-XL entity/relation extraction
│   │   ├── ontology_grounding.py    # UMLS exact-match grounding (MRCONSO)
│   │   ├── embeddings.py            # SapBERT 768-dim node embeddings
│   │   └── graph_builder.py         # PyG graph construction + validation
│   ├── models/
│   │   ├── interfaces.py            # Abstract base classes (ABCs)
│   │   ├── visual_encoder.py        # CLIPVisualEncoder (BioMedCLIP ViT-B/16)
│   │   ├── knowledge_encoder.py     # GATv2KnowledgeEncoder
│   │   ├── fusion.py                # CrossAttentionFusion
│   │   ├── classification.py        # ClassificationHead (14-class CheXpert)
│   │   ├── decoder.py               # TransformerReportDecoder
│   │   └── model_factory.py         # MedicalVLM + build_model()
│   ├── training/
│   │   ├── losses.py                # MultiTaskLoss (BCE + CE)
│   │   ├── scheduler.py             # Linear warmup + cosine decay
│   │   └── trainer.py               # Training loop + checkpointing
│   ├── evaluation/
│   │   ├── classification_metrics.py # AUC-ROC, F1, thresholds
│   │   ├── generation_metrics.py    # BLEU, ROUGE, BERTScore
│   │   └── statistical_tests.py     # McNemar, bootstrap CI, Bonferroni
│   └── pipelines/
│       ├── hybrid_kg_pipeline.py    # Phase I orchestrator
│       └── training_pipeline.py     # Phase II orchestrator
│
├── tests/                           # 84 unit tests (pytest)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_evaluation.py
│   ├── test_extraction.py
│   ├── test_graph_builder.py
│   ├── test_model_factory.py
│   ├── test_modules.py
│   ├── test_training.py
│   └── test_transforms.py
│
├── scripts/
│   ├── hydra/                       # SLURM job scripts for Hydra HPC
│   │   ├── setup_env.sh
│   │   ├── download_physionet.sh
│   │   ├── preprocess_dicom.sh
│   │   ├── run_phase1.sh
│   │   ├── train_baseline.sh
│   │   ├── train_neurosymbolic.sh
│   │   └── umls_download.sh
│   └── local/
│       └── run.sh                   # Local development runner
│
├── notebooks/
│   ├── Completed/
│   │   └── PhaseI_Hybrid_KG_RadGraph_UMLS.ipynb
│   └── complementary/              # Exploration notebooks
│
├── data/                            # Data directory (gitignored)
│   ├── MIMIC-CXR-RRG_reports/      # Radiology report text files (PhysioNet)
│   │   ├── cxr-record-list.csv     # 377K record mapping
│   │   └── files/p10..p19/         # Report text by patient
│   └── umls-2025AB-metathesaurus-full/
│       └── 2025AB/META/            # MRCONSO.RRF, MRSTY.RRF, etc.
│
└── outputs/                         # Generated artifacts (gitignored)
    ├── KG/                          # Phase I knowledge graph artifacts
    ├── checkpoints/                 # Model checkpoints
    └── logs/                        # Training logs
```

---

## Infrastructure

### Hydra HPC Cluster (VUB VSC Tier-2)

All GPU training and large-scale processing runs on the Hydra HPC cluster:

| Partition | GPUs | VRAM | Use Case |
|-----------|------|------|----------|
| `ampere_gpu` | 2× NVIDIA A100 | 40 GB | Model training (primary) |
| `hopper_gpu` | 2× NVIDIA H200 | 140 GB | Large batch training |
| `pascal_gpu` | 2× NVIDIA P100 | 16 GB | Small experiments |
| `zen4` / `zen5` | CPU only | 384+ GB RAM | Phase I KG, DICOM preprocessing |

**Storage layout:**

| Path | Purpose | Persistence |
|------|---------|-------------|
| `$VSC_HOME` | Code, configs, scripts | Persistent, backed up |
| `$VSC_DATA` | UMLS, KG artifacts, checkpoints | Persistent, not backed up |
| `$VSC_SCRATCH` | MIMIC-CXR DICOMs, temp files | Auto-purged (~30 days) |

### Local Development

Used for code editing, small-scale debugging (≤100 samples, CPU), and documentation. Any experiment touching >500 samples or requiring GPU **must** run on Hydra.

---

## Testing

```bash
# Run all tests (84 tests)
python -m pytest tests/ -v

# Run specific modules
python -m pytest tests/test_extraction.py -v       # Entity extraction
python -m pytest tests/test_graph_builder.py -v    # KG construction
python -m pytest tests/test_model_factory.py -v    # Model build + forward

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

| Test File | Coverage |
|-----------|----------|
| `test_config.py` | Config dataclass defaults, YAML overlay, nested overrides |
| `test_extraction.py` | Text normalization, entity parsing, filtering |
| `test_graph_builder.py` | Graph construction, deduplication, validation |
| `test_model_factory.py` | Model build, forward pass shapes, gradient flow |
| `test_modules.py` | Individual model components |
| `test_training.py` | MultiTaskLoss, scheduler warmup curves |
| `test_evaluation.py` | Classification/generation metrics, statistical tests |
| `test_transforms.py` | Image transforms, determinism |

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.4.0 | Core ML framework |
| `transformers` | 4.57.3 (pinned!) | RadGraph-XL (ModernBERT backbone) |
| `torch-geometric` | 2.6.0 | GATv2Conv knowledge encoder |
| `radgraph` | ≥0.1.0 | Medical entity/relation extraction |
| `sentence-transformers` | ≥2.2.0 | SapBERT node embeddings |
| `scikit-learn` | ≥1.3.0 | AUC-ROC, F1 metrics |
| `nltk` | ≥3.8.0 | BLEU score computation |
| `rouge-score` | ≥0.1.2 | ROUGE metrics |

See [requirements.txt](requirements.txt) for the complete pinned dependency list.

---

## Configuration System

All hyperparameters are managed via Python dataclasses with YAML overlay:

```python
from src.core.config import load_config, get_baseline_config, get_neurosymbolic_config

# Load from YAML (overrides dataclass defaults)
config = load_config("configs/phase2_neurosymbolic.yaml")

# Or use preset factories
baseline = get_baseline_config()       # use_kg=False, classification only
ns = get_neurosymbolic_config()        # use_kg=True, classification + generation
```

YAML files only need to specify values that differ from defaults.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `AttributeError: TokenizersBackend` | Install exact version: `pip install transformers==4.57.3` |
| `KeyError: 'modernbert'` | Same — transformers version mismatch |
| `radgraph not installed` | `pip install --force-reinstall --no-cache-dir radgraph` |
| `torch_geometric not found` | `pip install torch-geometric` |
| `MRCONSO.RRF not found` | Download UMLS 2025AB from NLM and place in `data/umls-2025AB-metathesaurus-full/2025AB/META/` |
| `OOM during training` | Reduce `batch_size` in config or increase `accumulation_steps` |
| `OOM during graph construction` | Use `--max-studies N` to limit processing |

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

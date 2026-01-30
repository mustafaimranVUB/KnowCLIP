# KnoCLIP-XAI Complete Setup Guide

This is the comprehensive setup guide for the KnoCLIP-XAI project.

## Overview

The project requires:
- **MIMIC-CXR-RRG dataset** - Download using `download_dataset.py`
- **UMLS MRCONSO.RRF file** - Download manually from UMLS website (recommended) or use `UMLS_ontology_download.py` for version 2021AB only

> **Important**: `transformers==4.57.3` is pinned in requirements.txt for RadGraph compatibility.

## Table of Contents
1. [Installation](#installation)
2. [Dataset Download](#dataset-download)
3. [Quick Start](#quick-start)
4. [Complete Examples](#complete-examples)
5. [Command Reference](#command-reference)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- ~10-50 GB free disk space (depending on dataset size)
- UMLS account (free, from https://uts.nlm.nih.gov/uts/) for MRCONSO.RRF download

### Setup Steps
```bash
# 1. Clone repository
git clone <repository-url>
cd Clone\ repo

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python verify_setup.py
```

## Dataset Download

### Available Presets

| Preset | Size | Best For |
|--------|------|----------|
| **minimal** | ~1 GB | Testing, quick demos |
| **small** | ~3-4 GB | Development, experimentation |
| **medium** | ~7-8 GB | Training models |
| **large** | ~15 GB | Full dataset analysis |
| **findings-only** | ~7.5 GB | Findings section |
| **impressions-only** | ~7.5 GB | Impressions section |

### Download Options

**Download Dataset:**
```bash
# List all presets
python download_dataset.py --list-presets

# Download minimal dataset (for testing)
python download_dataset.py --preset minimal

# Download small dataset (recommended for development)
python download_dataset.py --preset small --output-dir data/MIMIC-CXR-RRG_small

# Download custom patterns
python download_dataset.py --custom-patterns \
  findings_section/test-00000-of-00016.parquet \
  impression_section/test-00000-of-00016.parquet \
  README.md
```

**Download UMLS MRCONSO.RRF:**

*Option A: Manual Download (Recommended)*
1. Go to [UMLS Downloads](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html)
2. Create a free UMLS account if you don't have one
3. Download the UMLS Metathesaurus
4. Extract and locate `MRCONSO.RRF` in the `META/` folder
5. Place it in your data directory (e.g., `data/umls/META/MRCONSO.RRF`)

*Option B: Using Download Script (2021AB version only)*
```bash
# Note: Only works for UMLS version 2021AB
python src/downloads/UMLS_ontology_download.py \
  --api-key YOUR_UMLS_API_KEY \
  --version 2021AB \
  --output-dir data/umls \
  --extract
```

## Quick Start

### Minimal Setup (Testing)
```bash
# 1. Download dataset
python download_dataset.py --preset minimal

# 2. Run pipeline (assuming MRCONSO.RRF is already downloaded)
python main.py --dataset-dir data/MIMIC-CXR-RRG_small --mrconso data/umls/META/MRCONSO.RRF
```

### Development Setup
```bash
# 1. Download dataset
python download_dataset.py --preset small --output-dir data/MIMIC-CXR-RRG_small

# 2. Run pipeline
python main.py --dataset-dir data/MIMIC-CXR-RRG_small --mrconso data/umls/META/MRCONSO.RRF
```

## Complete Examples

### Example 1: Complete Workflow
```bash
# Step 1: Download dataset
python download_dataset.py --preset small --output-dir data/MIMIC-CXR-RRG_small

# Step 2: Download MRCONSO.RRF manually from UMLS website
#         OR use download script for 2021AB:
python src/downloads/UMLS_ontology_download.py \
  --api-key YOUR_KEY \
  --version 2021AB \
  --output-dir data/umls \
  --extract

# Step 3: Run pipeline
python main.py \
  --dataset-dir data/MIMIC-CXR-RRG_small \
  --mrconso data/umls/2021AB/META/MRCONSO.RRF \
  --split test \
  --seed 42

# Check outputs
ls -lh outputs/KG/
```

### Example 2: Multiple Runs with Same Data
```bash
# Download data once
python download_dataset.py --preset medium --output-dir data/MIMIC-CXR-RRG_medium

# Run with different seeds
python main.py \
  --dataset-dir data/MIMIC-CXR-RRG_medium \
  --mrconso data/umls/META/MRCONSO.RRF \
  --split test --seed 42

python main.py \
  --dataset-dir data/MIMIC-CXR-RRG_medium \
  --mrconso data/umls/META/MRCONSO.RRF \
  --split test --seed 99 \
  --output-dir outputs/KG_seed99
```

### Example 3: Using Shell Scripts
```bash
# Linux/Mac
chmod +x run.sh
./run.sh --dataset-dir data/MIMIC-CXR-RRG_small --mrconso data/umls/META/MRCONSO.RRF

# Windows
run.bat --dataset-dir data\MIMIC-CXR-RRG_small --mrconso data\umls\META\MRCONSO.RRF
```

## Command Reference

### download_dataset.py
```bash
python download_dataset.py [OPTIONS]

Options:
  --preset              Dataset size preset (minimal/small/medium/large/findings-only/impressions-only)
  --output-dir          Where to save (default: data/MIMIC-CXR-RRG_small)
  --custom-patterns     Custom download patterns
  --list-presets        Show all available presets
  --force               Re-download if exists
```

### main.py
```bash
python main.py --dataset-dir DIR --mrconso FILE [OPTIONS]

Required:
  --dataset-dir DIR     Path to downloaded dataset
  --mrconso FILE        Path to MRCONSO.RRF file

Optional:
  --split SPLIT         Dataset split: test/train/val (default: test)
  --seed SEED           Random seed (default: 42)
  --model-type TYPE     RadGraph model (default: modern-radgraph-xl)
  --output-dir DIR      Output location (default: outputs/KG)
```

### UMLS_ontology_download.py (2021AB only)
```bash
python src/downloads/UMLS_ontology_download.py --api-key KEY --version 2021AB [OPTIONS]

Required:
  --api-key KEY         UMLS API key (from https://uts.nlm.nih.gov/uts/)
  --version VER         Must be 2021AB (other versions not supported)

Optional:
  --output-dir DIR      Output directory
  --extract             Extract after download
```

## Key Files

| File | Purpose |
|------|---------|
| `download_dataset.py` | Standalone MIMIC-CXR-RRG downloader |
| `main.py` | Main pipeline entry point |
| `run.sh` | Bash script (Linux/Mac) |
| `run.bat` | Batch script (Windows) |
| `src/downloads/mimic_downloader.py` | Dataset download module |
| `src/downloads/UMLS_ontology_download.py` | UMLS downloader (2021AB only) |
| `src/pipelines/hybrid_kg_pipeline.py` | KG pipeline |

## Outputs

After running, check `outputs/KG/`:

- **entities_with_cui.pkl** - Entities with UMLS mappings
- **mention2cui.pkl** - Mention-to-CUI mappings
- **cui_coverage.csv** - Coverage statistics

## Troubleshooting

### Download Issues

**Q: Download fails or is interrupted**
```bash
# Downloads are resumable; just re-run
python download_dataset.py --preset small
```

**Q: Not enough disk space**
```bash
# Use smaller preset
python download_dataset.py --preset minimal
```

### Transformers/RadGraph Issues

**Q: "AttributeError: TokenizersBackend has no attribute encode_plus"**
```bash
# Install the correct transformers version
pip install transformers==4.57.3
```

**Q: "KeyError: 'modernbert'"**
```bash
# Your transformers version is too old
pip install transformers==4.57.3
```

### Module Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or verify environment
python verify_setup.py
```

## Pipeline Steps

```
1. Download Dataset    (python download_dataset.py --preset small)
   ↓
2. Download MRCONSO    (manual from UMLS website or script for 2021AB)
   ↓
3. Entity Extraction   (RadGraph with modern-radgraph-xl)
   ↓
4. UMLS Grounding      (CUI mapping)
   ↓
5. Graph Construction  (PyTorch Geometric)
   ↓
6. Export Outputs      (PKL + CSV to outputs/KG/)
```

## Performance Tips

- Use `minimal` preset for testing (1 GB, <5 min)
- Use `small` preset for development (3-4 GB, 10-30 min)
- Dataset download speed depends on internet (typical: 1-50 MB/s)
- Processing speed depends on CPU/GPU
- Monitor disk space during downloads

---

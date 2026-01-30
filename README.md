# KnoCLIP-XAI
# Self-Explainable Vision–Language Framework for Medical Imaging

## Overview
This repository contains the code, experiments, and manuscript for a Master's thesis
focused on **Self-Explainable AI (S-XAI)** for medical imaging. The core objective is to
develop a **clinically grounded, self-explainable framework** that integrates:

- Vision–Language Models (CLIP-based)
- Medical ontologies and knowledge graphs
- Concept-based explainability
- Explainable medical report generation

The framework is designed to support **zero-shot localization** and **explainable clinical
report generation**, where explanations are *inherently linked* to the model’s internal
decision-making process rather than produced post-hoc.

---

## Research Objectives
1. Integrate **prior medical knowledge** (ontologies + KGs) into representation learning.
2. Embed knowledge representations directly into model architectures.
3. Associate model activations with **explicit medical concepts**.
4. Generate **clinically interpretable and explainable medical reports**.
5. Provide rigorous **quantitative and qualitative evaluation** of explainability.

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- MIMIC-CXR-RRG dataset (downloaded via `download_dataset.py`)
- UMLS MRCONSO.RRF file (downloaded manually or via `UMLS_ontology_download.py` for 2021AB only)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Clone\ repo
```

2. **Set up virtual environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Important**: The `requirements.txt` pins `transformers==4.57.3` which is required for compatibility with the RadGraph model.

### Usage

#### Step 1: Download the Dataset (Required)

Download the MIMIC-CXR-RRG dataset from HuggingFace using the provided script:

```bash
# List all available presets
python download_dataset.py --list-presets

# Download small dataset (~3-4 GB) - recommended for development
python download_dataset.py --preset small --output-dir data/MIMIC-CXR-RRG_small

# Download minimal dataset (~1 GB) - for testing
python download_dataset.py --preset minimal
```

**Available presets:**
| Preset | Size | Description |
|--------|------|-------------|
| `minimal` | ~1 GB | 1 findings + 2 impression shards (testing) |
| `small` | ~3-4 GB | 4 findings + 4 impression shards (development) |
| `medium` | ~7-8 GB | All findings shards |
| `large` | ~15 GB | Full dataset |
| `findings-only` | ~7.5 GB | Only findings section |
| `impressions-only` | ~7.5 GB | Only impression section |

#### Step 2: Download UMLS MRCONSO.RRF (Required)

You need the UMLS MRCONSO.RRF file for entity grounding. **Choose one option:**

**Option A: Manual Download (Recommended)**
1. Go to [UMLS Downloads](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html)
2. Download the UMLS Metathesaurus (requires free UMLS account)
3. Extract and locate `MRCONSO.RRF` in the `META/` folder
4. Place it in your data directory, e.g., `data/umls/META/MRCONSO.RRF`

**Option B: Using the Download Script (2021AB version only)**
```bash
# Note: This script only works for UMLS version 2021AB
python src/downloads/UMLS_ontology_download.py \
  --api-key YOUR_UMLS_API_KEY \
  --version 2021AB \
  --output-dir data/umls \
  --extract
```

> **Note**: Get your UMLS API key from [UTS](https://uts.nlm.nih.gov/uts/) (free account required).

#### Step 3: Run the Pipeline

**Using Python directly (Recommended):**
```bash
python main.py \
  --dataset-dir data/MIMIC-CXR-RRG_small \
  --mrconso data/umls/META/MRCONSO.RRF
```

**Using the shell script (Linux/Mac):**
```bash
chmod +x run.sh
./run.sh --dataset-dir data/MIMIC-CXR-RRG_small --mrconso data/umls/META/MRCONSO.RRF
```

**Using the batch script (Windows):**
```batch
run.bat --dataset-dir data\MIMIC-CXR-RRG_small --mrconso data\umls\META\MRCONSO.RRF
```

**Command-line options:**
| Option | Description |
|--------|-------------|
| `-d, --dataset-dir DIR` | Path to downloaded dataset directory |
| `-m, --mrconso FILE` | Path to MRCONSO.RRF file |
| `-s, --split SPLIT` | Dataset split: test/train/val (default: test) |
| `--seed SEED` | Random seed (default: 42) |
| `--model-type TYPE` | RadGraph model: radgraph, radgraph-xl, modern-radgraph-xl (default: modern-radgraph-xl) |
| `-o, --output-dir DIR` | Output directory (default: outputs/KG) |
| `--skip-install` | Skip dependency installation |
| `-h, --help` | Show help message |

#### Running the Pipeline Directly

You can also run the pipeline module directly:

```bash
python src/pipelines/hybrid_kg_pipeline.py \
  --dataset-dir data/MIMIC-CXR-RRG_small \
  --mrconso data/umls/META/MRCONSO.RRF \
  --split test \
  --output-dir outputs/KG
```

### Output

The pipeline generates the following outputs in the `data/outputs/hybrid_kg/` directory (or custom output directory):

- **`entities_with_cui.pkl`**: Extracted entities enriched with UMLS CUI mappings
- **`mention2cui.pkl`**: Mention-to-CUI mapping dictionary
- **`cui_coverage.csv`**: Coverage statistics and mapping quality metrics

### Example Workflow

**Complete pipeline from scratch:**

```bash
# 1. Download the dataset
python download_dataset.py --preset small --output-dir data/MIMIC-CXR-RRG_small

# 2. Download UMLS MRCONSO.RRF manually from UMLS website
#    OR use the download script (2021AB only):
python src/downloads/UMLS_ontology_download.py --api-key YOUR_KEY --version 2021AB --output-dir data/umls --extract

# 3. Run the pipeline
python main.py --dataset-dir data/MIMIC-CXR-RRG_small --mrconso data/umls/2021AB/META/MRCONSO.RRF

# 4. Check the outputs
ls -l outputs/KG/
```

**Windows example:**

```batch
REM 1. Download dataset
python download_dataset.py --preset small --output-dir data\MIMIC-CXR-RRG_small

REM 2. Run pipeline (assuming MRCONSO.RRF is already downloaded)
python main.py --dataset-dir data\MIMIC-CXR-RRG_small --mrconso data\umls\META\MRCONSO.RRF

REM 3. Check outputs
dir outputs\KG
```

---

## Project Structure

```
Clone repo/
├── main.py                          # Main entry point for complete pipeline
├── download_dataset.py              # Standalone dataset download script
├── run.sh                           # Bash script for automated execution
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── SETUP.md                         # Detailed setup guide
├── verify_setup.py                  # Environment verification script
├── LICENSE
├── .env.example                     # Environment variables template
│
├── notebooks/
│   └── Completed/
│       └── PhaseI_Hybrid_KG_RadGraph_UMLS.ipynb  # Research notebook
│
├── src/
│   ├── __init__.py
│   ├── core/                        # Core utilities
│   ├── downloads/                   # Download modules
│   │   ├── __init__.py
│   │   ├── mimic_downloader.py      # MIMIC-CXR-RRG downloader
│   │   └── UMLS_ontology_download.py  # UMLS download (2021AB only)
│   ├── evaluation/                  # Evaluation metrics (empty for now)
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── extraction.py            # RadGraph entity extraction
│   │   ├── graph_builder.py         # PyTorch Geometric graph construction
│   │   └── ontology_grounding.py    # UMLS grounding/mapping
│   ├── models/                      # ML models (empty for now)
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── hybrid_kg_pipeline.py    # End-to-end pipeline
│   └── training/                    # Training scripts (empty for now)
│
└── data/                            # Data directory (created at runtime)
    ├── MIMIC-CXR-RRG_small/         # Downloaded MIMIC dataset
    ├── umls_metathesaurus/          # Downloaded UMLS data
    └── outputs/
        └── hybrid_kg/               # Pipeline outputs
```

---

## API Key Setup

### Obtaining a UMLS API Key

1. Go to [UMLS Terminology Services (UTS)](https://uts.nlm.nih.gov/uts/)
2. Click "Sign Up" if you don't have an account
3. Fill in the required information and agree to the terms
4. Once logged in, go to "My Profile"
5. Under "API Keys", click "Generate new API Key"
6. Copy your API key

### Setting Up the API Key

**Method 1: Environment Variable (Recommended)**

On Linux/Mac:
```bash
export UMLS_API_KEY=your_api_key_here
```

On Windows (PowerShell):
```powershell
$env:UMLS_API_KEY="your_api_key_here"
```

**Method 2: .env File**

Create a `.env` file in the project root:
```bash
echo "UMLS_API_KEY=your_api_key_here" > .env
```

**Method 3: Command-line Argument**

Pass the API key directly to the script:
```bash
python main.py --api-key your_api_key_here ...
# or
./run.sh --api-key your_api_key_here ...
```

---

## Pipeline Components

### 1. UMLS Metathesaurus Download
Downloads and extracts the UMLS Metathesaurus, which provides:
- MRCONSO.RRF: Concept names and sources
- Medical concept mappings
- Semantic types and relationships

### 2. Entity Extraction (RadGraph)
Extracts medical entities and relationships from radiology reports:
- **Entity Types**: ANATOMY, OBSERVATION
- **Relation Types**: LOCATED_AT, MODIFY, SUGGESTIVE_OF, ASSOCIATED_WITH
- Uses the RadGraph model (modern-radgraph-xl by default)

### 3. UMLS Grounding
Maps extracted entities to UMLS Concept Unique Identifiers (CUIs):
- Exact-match algorithm with configurable preferences
- Filters measurement-like entities
- Ranks candidates by source (SAB), term type (TTY), and preference

### 4. Hybrid Knowledge Graph Construction
Builds a PyTorch Geometric graph combining:
- **Prior Knowledge** (V_prior): UMLS ontology concepts
- **Data-driven** (V_data): Extracted entities from reports
- Node features: 768-dim embeddings
- Edge types: RadGraph relations

---

## Dependencies

### Core Libraries
- **torch>=2.0.0**: Deep learning framework
- **torch-geometric>=2.3.0**: Graph neural network library
- **radgraph>=0.1.0**: Medical entity extraction
- **transformers==4.57.3**: Required version for RadGraph compatibility

### Data Processing
- **pandas>=2.0.0**: Data manipulation
- **numpy>=1.24.0**: Numerical operations
- **datasets>=2.14.0**: HuggingFace dataset loader

### Utilities
- **python-dotenv>=1.0.0**: Environment variable management
- **tqdm>=4.65.0**: Progress bars
- **huggingface-hub>=0.34.0,<1.0**: Model hub access

> **Important**: `transformers==4.57.3` is pinned because newer versions break RadGraph compatibility.

See [requirements.txt](requirements.txt) for complete list.


## Troubleshooting

### Common Issues

**Issue: "radgraph is not installed"**
```bash
pip install --force-reinstall --no-cache-dir radgraph
```

**Issue: "AttributeError: TokenizersBackend has no attribute encode_plus"**
- This means you have an incompatible transformers version
- Solution: `pip install transformers==4.57.3`

**Issue: "KeyError: 'modernbert'"**
- Your transformers version is too old for `modern-radgraph-xl`
- Solution: `pip install transformers==4.57.3`

**Issue: "Dataset directory does not exist"**
- Download the dataset first: `python download_dataset.py --preset small`
- Verify the path is correct
- Use absolute paths if relative paths don't work

**Issue: "MRCONSO.RRF not found"**
- Download UMLS MRCONSO.RRF manually from [UMLS](https://www.nlm.nih.gov/research/umls/)
- Or use: `python src/downloads/UMLS_ontology_download.py --api-key KEY --version 2021AB`
- Note: The download script only works for version 2021AB

**Issue: "Out of memory during graph construction"**
- Reduce the dataset size (use `--preset minimal`)
- Use a machine with more RAM
- Consider processing in batches


## License

See [LICENSE](LICENSE) file for details.

---

## Contributing

This is a research project. For questions or suggestions, please open an issue.

---

## Acknowledgments

- RadGraph: Chen et al. for the RadGraph model
- UMLS: National Library of Medicine
- MIMIC-CXR: Johnson et al. for the MIMIC-CXR dataset

---



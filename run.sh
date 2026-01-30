#!/bin/bash

# run.sh - Setup and run KnoCLIP-XAI Hybrid Knowledge Graph Pipeline
# This script automates the complete workflow from environment setup to pipeline execution

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run the KnoCLIP-XAI Hybrid Knowledge Graph Pipeline.

PREREQUISITES:
    1. Download dataset first:    python download_dataset.py --preset small
    2. Download MRCONSO.RRF:      Manual download from UMLS website recommended
                                  (or use UMLS_ontology_download.py for 2021AB only)

OPTIONS:
    -h, --help              Display this help message
    -d, --dataset-dir DIR   Path to dataset directory (required)
    -m, --mrconso FILE      Path to MRCONSO.RRF file (required)
    -s, --split SPLIT       Dataset split to use (default: test)
    --seed SEED             Random seed (default: 42)
    --model-type TYPE       RadGraph model type (default: modern-radgraph-xl)
    -o, --output-dir DIR    Output directory for results
    --skip-install          Skip dependency installation
    --venv DIR              Use specific virtual environment directory

EXAMPLES:
    # Run with downloaded dataset and MRCONSO
    $0 --dataset-dir data/MIMIC-CXR-RRG_small --mrconso data/umls/META/MRCONSO.RRF

    # Specify output directory
    $0 -d data/MIMIC-CXR-RRG_small -m data/umls/META/MRCONSO.RRF -o outputs/my_run

    # Use different model type
    $0 -d data/MIMIC-CXR-RRG_small -m data/umls/META/MRCONSO.RRF --model-type radgraph-xl

EOF
}

# Default values
DATASET_DIR=""
SPLIT="test"
SEED=42
MODEL_TYPE="modern-radgraph-xl"
SKIP_INSTALL=false
VENV_DIR=".venv"
MRCONSO=""
OUTPUT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -d|--dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        -m|--mrconso)
            MRCONSO="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Print banner
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         KnoCLIP-XAI Hybrid Knowledge Graph Pipeline       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check Python installation
print_msg "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION found"

# Step 2: Create and activate virtual environment
if [ "$SKIP_INSTALL" = false ]; then
    print_msg "Setting up virtual environment in '$VENV_DIR'..."
    
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_msg "Activating virtual environment..."
    source "$VENV_DIR/bin/activate" || {
        # Try Windows activation if Linux activation fails
        source "$VENV_DIR/Scripts/activate" 2>/dev/null || {
            print_error "Failed to activate virtual environment"
            exit 1
        }
    }
    print_success "Virtual environment activated"
    
    # Step 3: Install dependencies
    print_msg "Installing dependencies from requirements.txt..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt
    print_success "Dependencies installed"
else
    print_warning "Skipping installation (--skip-install flag set)"
fi

# Step 4: Validate arguments
print_msg "Validating configuration..."

if [ -z "$DATASET_DIR" ]; then
    print_error "Dataset directory is required"
    print_error "Use --dataset-dir or download first with: python download_dataset.py --preset small"
    usage
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    print_error "Dataset directory does not exist: $DATASET_DIR"
    print_error "Download first with: python download_dataset.py --preset small --output-dir $DATASET_DIR"
    exit 1
fi

if [ -z "$MRCONSO" ]; then
    print_error "MRCONSO path is required"
    print_error "Download MRCONSO.RRF manually from UMLS website"
    print_error "Or use: python src/downloads/UMLS_ontology_download.py --api-key KEY --version 2021AB"
    usage
    exit 1
fi

if [ ! -f "$MRCONSO" ]; then
    print_error "MRCONSO file does not exist: $MRCONSO"
    exit 1
fi

print_success "Configuration validated"

# Step 5: Build command
print_msg "Preparing to run pipeline..."

CMD="python3 main.py --dataset-dir \"$DATASET_DIR\" --split \"$SPLIT\" --seed $SEED --model-type \"$MODEL_TYPE\" --mrconso \"$MRCONSO\""

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

# Step 6: Run the pipeline
echo ""
print_msg "Starting pipeline with the following configuration:"
echo "  Dataset: $DATASET_DIR"
echo "  MRCONSO: $MRCONSO"
echo "  Split: $SPLIT"
echo "  Model: $MODEL_TYPE"
echo "  Seed: $SEED"
if [ -n "$OUTPUT_DIR" ]; then
    echo "  Output: $OUTPUT_DIR"
fi
echo ""

eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    print_success "Pipeline completed successfully!"
    echo ""
else
    echo ""
    print_error "Pipeline failed with errors"
    exit 1
fi

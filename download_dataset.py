"""
Standalone script to download MIMIC-CXR-RRG dataset from HuggingFace.

Usage:
    python download_dataset.py --preset minimal
    python download_dataset.py --preset small --output-dir data/my_dataset
    python download_dataset.py --list-presets
    python download_dataset.py --custom-patterns findings_section/test-00000-of-00016.parquet README.md
"""

from pathlib import Path
from src.downloads.mimic_downloader import main

if __name__ == "__main__":
    main()

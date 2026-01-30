"""
Main entry point for KnoCLIP-XAI Hybrid Knowledge Graph Pipeline.

This script orchestrates the complete workflow:
1. Download UMLS Metathesaurus (optional)
2. Run Hybrid KG Pipeline to extract entities and ground them to UMLS

Usage:
    # With environment variable
    export UMLS_API_KEY=your_api_key
    python main.py --dataset-dir data/MIMIC-CXR-RRG_small --mrconso data/umls_metathesaurus/2021AB/META/MRCONSO.RRF

    # With API key as argument
    python main.py --api-key YOUR_KEY --dataset-dir data/MIMIC-CXR-RRG_small --download-umls
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


def download_umls_data(
    api_key: str, version: str, output_dir: Path, extract: bool, force: bool
) -> Path:
    """Download and extract UMLS Metathesaurus."""
    from src.downloads.UMLS_ontology_download import ensure_metathesaurus

    print("\n" + "=" * 60)
    print("STEP 1: Downloading UMLS Metathesaurus")
    print("=" * 60)

    result_path = ensure_metathesaurus(
        api_key=api_key,
        version=version,
        output_dir=output_dir,
        extract=extract,
        force=force,
    )

    print(f"UMLS data available at: {result_path}")
    return result_path


def download_mimic_dataset(
    preset: str,
    output_dir: Path,
    custom_patterns: list[str] | None = None,
    force: bool = False,
) -> Path:
    """Download MIMIC-CXR-RRG dataset from HuggingFace."""
    from src.downloads.mimic_downloader import download_mimic_cxr_rrg

    print("\n" + "=" * 60)
    print("STEP 1: Downloading MIMIC-CXR-RRG Dataset")
    print("=" * 60)

    result_path = download_mimic_cxr_rrg(
        output_dir=output_dir,
        preset=preset,
        custom_patterns=custom_patterns,
        force=force,
    )

    print(f"Dataset available at: {result_path}")
    return result_path


def run_hybrid_kg_pipeline(
    dataset_dir: Path,
    mrconso_path: Path,
    split: str,
    seed: int,
    model_type: str,
    output_dir: Path | None,
) -> None:
    """Run the Hybrid Knowledge Graph extraction and grounding pipeline."""
    from src.pipelines.hybrid_kg_pipeline import run_pipeline

    print("\n" + "=" * 60)
    print("STEP 3: Running Hybrid KG Pipeline")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"MRCONSO path: {mrconso_path}")
    print(f"Split: {split}")
    print(f"Model: {model_type}")
    print(f"Seed: {seed}")

    run_pipeline(
        dataset_dir=dataset_dir,
        mrconso_path=mrconso_path,
        split=split,
        seed=seed,
        model_type=model_type,
        output_dir=output_dir,
    )

    print("\nPipeline completed successfully!")


def build_parser() -> argparse.ArgumentParser:
    """Build the main argument parser."""
    parser = argparse.ArgumentParser(
        description="KnoCLIP-XAI: Self-Explainable Vision-Language Framework for Medical Imaging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # API Key
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=os.getenv("UMLS_API_KEY"),
        help="UMLS/UTS API key. Falls back to UMLS_API_KEY environment variable if not provided.",
    )

    # Dataset Download Options
    dataset_group = parser.add_argument_group("Dataset Download Options")
    dataset_group.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download MIMIC-CXR-RRG dataset from HuggingFace before running the pipeline.",
    )
    dataset_group.add_argument(
        "--dataset-preset",
        default="minimal",
        choices=[
            "minimal",
            "small",
            "medium",
            "large",
            "findings-only",
            "impressions-only",
        ],
        help=(
            "Preset download configuration: "
            "minimal (~1GB), small (~3-4GB), medium (~7-8GB), large (~15GB), "
            "findings-only (~7.5GB), impressions-only (~7.5GB)"
        ),
    )
    dataset_group.add_argument(
        "--dataset-custom-patterns",
        nargs="+",
        default=None,
        help=(
            "Custom file patterns to download. Overrides --dataset-preset. "
            "Example: --dataset-custom-patterns findings_section/test-00000-of-00016.parquet README.md"
        ),
    )
    dataset_group.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/MIMIC-CXR-RRG_small"),
        help="Path to store downloaded MIMIC dataset.",
    )
    dataset_group.add_argument(
        "--force-dataset-download",
        action="store_true",
        help="Force re-download even if dataset already exists.",
    )

    # UMLS Download Options
    umls_group = parser.add_argument_group("UMLS Download Options")
    umls_group.add_argument(
        "--download-umls",
        action="store_true",
        help="Download UMLS Metathesaurus before running the pipeline.",
    )
    umls_group.add_argument(
        "--umls-version",
        default="2021AB",
        help="UMLS version to download (e.g., 2024AB).",
    )
    umls_group.add_argument(
        "--umls-output-dir",
        type=Path,
        default=Path("data/umls_metathesaurus"),
        help="Directory to store downloaded UMLS data.",
    )
    umls_group.add_argument(
        "--force-umls-download",
        action="store_true",
        help="Force re-download even if UMLS data already exists.",
    )

    # Pipeline Options
    pipeline_group = parser.add_argument_group("Pipeline Options")
    pipeline_group.add_argument(
        "--mrconso",
        type=Path,
        help="Path to MRCONSO.RRF file. If not provided, will use downloaded UMLS data.",
    )
    pipeline_group.add_argument(
        "--split",
        default="test",
        help="Dataset split to process (train/test/validation).",
    )
    pipeline_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    pipeline_group.add_argument(
        "--model-type",
        default="modern-radgraph-xl",
        help="RadGraph model type to use for entity extraction. Options: 'radgraph', 'radgraph-xl', 'modern-radgraph-xl' (requires newer transformers).",
    )
    pipeline_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/KG"),
        help="Directory to save pipeline outputs (entities, mappings, coverage stats).",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    # If downloading dataset, we don't need to check if it exists yet
    if not args.download_dataset:
        # Check if dataset directory exists when not downloading
        if not args.dataset_dir.exists():
            print(f"Error: Dataset directory does not exist: {args.dataset_dir}")
            print("       Use --download-dataset to download the dataset automatically")
            sys.exit(1)

    # If downloading UMLS, we need an API key
    if args.download_umls and not args.api_key:
        print("Error: --api-key is required when using --download-umls")
        print("       Alternatively, set the UMLS_API_KEY environment variable")
        sys.exit(1)

    # If not downloading UMLS, we need an existing MRCONSO path
    if not args.download_umls and not args.mrconso:
        print("Error: --mrconso is required when not downloading UMLS")
        print("       Use --download-umls to download UMLS data first")
        sys.exit(1)

    if args.mrconso and not args.mrconso.exists():
        print(f"Error: MRCONSO file does not exist: {args.mrconso}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    print("\n" + "=" * 60)
    print("KnoCLIP-XAI: Complete Pipeline")
    print("=" * 60)

    dataset_dir = args.dataset_dir

    # Step 1: Download dataset if requested
    if args.download_dataset:
        dataset_dir = download_mimic_dataset(
            preset=args.dataset_preset,
            output_dir=args.dataset_dir,
            custom_patterns=args.dataset_custom_patterns,
            force=args.force_dataset_download,
        )

    # Step 2: Download UMLS if requested
    mrconso_path = args.mrconso
    if args.download_umls:
        umls_dir = download_umls_data(
            api_key=args.api_key,
            version=args.umls_version,
            output_dir=args.umls_output_dir,
            extract=True,
            force=args.force_umls_download,
        )
        # Construct path to MRCONSO.RRF
        mrconso_path = args.umls_output_dir / args.umls_version / "META" / "MRCONSO.RRF"

        if not mrconso_path.exists():
            print(f"\nError: Could not find MRCONSO.RRF at {mrconso_path}")
            print("The UMLS download may have a different structure.")
            print("Please check the extracted files and specify --mrconso manually.")
            sys.exit(1)

    # Step 3: Run the Hybrid KG Pipeline
    run_hybrid_kg_pipeline(
        dataset_dir=dataset_dir,
        mrconso_path=mrconso_path,
        split=args.split,
        seed=args.seed,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("All steps completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

"""Download MIMIC-CXR-RRG dataset from HuggingFace."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, List

# 1. Try to import dotenv to load .env file
try:
    from dotenv import load_dotenv

    # Load environment variables from .env file immediately
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, we simply proceed without it
    pass

# 2. Update huggingface_hub import to include login
try:
    from huggingface_hub import snapshot_download, login
except ImportError:
    raise ImportError(
        "huggingface-hub is required. Install with: pip install huggingface-hub"
    )


# Preset download patterns for different data sizes
PRESET_PATTERNS = {
    "minimal": {
        "description": "Minimal dataset (~1 GB): 1 findings shard + 2 impression shards",
        "patterns": [
            "findings_section/test-00000-of-00016.parquet",
            "impression_section/test-00000-of-00016.parquet",
            "impression_section/test-00001-of-00016.parquet",
            "README.md",
        ],
    },
    "small": {
        "description": "Small dataset (~3-4 GB): 4 findings shards + 4 impression shards",
        "patterns": [
            "findings_section/test-00000-of-00016.parquet",
            "findings_section/test-00001-of-00016.parquet",
            "findings_section/test-00002-of-00016.parquet",
            "findings_section/test-00003-of-00016.parquet",
            "impression_section/test-00000-of-00016.parquet",
            "impression_section/test-00001-of-00016.parquet",
            "impression_section/test-00002-of-00016.parquet",
            "impression_section/test-00003-of-00016.parquet",
            "README.md",
        ],
    },
    "medium": {
        "description": "Medium dataset (~7-8 GB): All impression shards",
        "patterns": [
            "impression_section/*",
            "README.md",
        ],
    },
    "large": {
        "description": "Large dataset (~15 GB): All findings + impressions",
        "patterns": [
            "findings_section/*",
            "impression_section/*",
            "README.md",
        ],
    },
    "findings-only": {
        "description": "Findings section only (~7.52 GB)",
        "patterns": [
            "findings_section/*",
            "README.md",
            ".gitattributes",
        ],
    },
    "impressions-only": {
        "description": "Impression section only (~7.5 GB)",
        "patterns": [
            "impression_section/*",
            "README.md",
            ".gitattributes",
        ],
    },
}


def authenticate_huggingface() -> None:
    """
    Attempt to authenticate using HF_TOKEN from environment.
    """
    token = os.getenv("HF_TOKEN")
    if token:
        print("\n" + "-" * 40)
        print("Authentication")
        print("-" * 40)
        try:
            # Attempt login
            login(token=token, add_to_git_credential=False)
            print("Successfully authenticated with Hugging Face Hub")
        except Exception as e:
            print(f"Authentication failed: {e}")
            print("  Proceeding without authentication...")
    else:
        print("\nNo HF_TOKEN found in environment or .env file.")
        print("  Proceeding with unauthenticated download (rate limits may apply).")


def download_mimic_cxr_rrg(
    repo_id: str = "X-iZhang/MIMIC-CXR-RRG",
    output_dir: Path | str = "data/MIMIC-CXR-RRG_small",
    preset: str = "minimal",
    custom_patterns: Optional[List[str]] = None,
    force: bool = False,
) -> Path:
    """
    Download MIMIC-CXR-RRG dataset from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID (default: X-iZhang/MIMIC-CXR-RRG)
    output_dir : Path | str
        Directory to store the downloaded dataset
    preset : str
        Preset pattern configuration. One of: minimal, small, medium, large,
        findings-only, impressions-only. Ignored if custom_patterns is provided.
    custom_patterns : Optional[List[str]]
        Custom file patterns to download. If provided, overrides preset.
        Example: ["findings_section/test-00000-of-00016.parquet", "README.md"]
    force : bool
        Force re-download even if files already exist

    Returns
    -------
    Path
        Path to the downloaded dataset
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine patterns to use
    if custom_patterns is not None:
        patterns = custom_patterns
        preset_name = "custom"
    elif preset not in PRESET_PATTERNS:
        raise ValueError(
            f"Unknown preset: {preset}. "
            f"Valid options: {', '.join(PRESET_PATTERNS.keys())}"
        )
    else:
        patterns = PRESET_PATTERNS[preset]["patterns"]
        preset_name = preset

    print("\n" + "=" * 70)
    print("MIMIC-CXR-RRG Dataset Download")
    print("=" * 70)
    print(f"Repository: {repo_id}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Preset: {preset_name}")
    if preset_name != "custom":
        print(f"Description: {PRESET_PATTERNS[preset]['description']}")
    print(f"Files to download: {len(patterns)}")
    print("\nFile patterns:")
    for pattern in patterns:
        print(f"  - {pattern}")

    print("\nDownloading...")

    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            allow_patterns=patterns,
            force_download=force,
            # snapshot_download automatically picks up HF_TOKEN from env if present,
            # but explicit login in main() handles the "warning" display better.
        )

        result_path = Path(local_path)
        print("\n" + "=" * 70)
        print(f"Download completed successfully!")
        print(f"  Location: {result_path.resolve()}")
        print("=" * 70 + "\n")

        return result_path

    except Exception as e:
        print(f"\nDownload failed: {str(e)}")
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        description="Download MIMIC-CXR-RRG dataset from HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--repo-id",
        default="X-iZhang/MIMIC-CXR-RRG",
        help="HuggingFace repository ID",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/MIMIC-CXR-RRG_small"),
        help="Directory to store the downloaded dataset",
    )

    parser.add_argument(
        "--preset",
        choices=list(PRESET_PATTERNS.keys()),
        default="minimal",
        help="Preset download configuration",
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Show available presets and exit",
    )

    parser.add_argument(
        "--custom-patterns",
        nargs="+",
        default=None,
        help=(
            "Custom file patterns to download. Overrides --preset. "
            "Example: --custom-patterns findings_section/test-00000-of-00016.parquet README.md"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )

    return parser


def main() -> None:
    """Command-line entry point."""

    parser = build_argument_parser()
    args = parser.parse_args()

    # Handle --list-presets
    if args.list_presets:
        print("\nAvailable Presets:")
        print("=" * 70)
        for preset_name, preset_info in PRESET_PATTERNS.items():
            print(f"\n{preset_name}:")
            print(f"  Description: {preset_info['description']}")
            print(f"  Patterns: {len(preset_info['patterns'])} files")
            for pattern in preset_info["patterns"]:
                print(f"    - {pattern}")
        print("\n" + "=" * 70 + "\n")
        return

    # Attempt Authentication before download
    authenticate_huggingface()

    # Download dataset
    try:
        result = download_mimic_cxr_rrg(
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            preset=args.preset,
            custom_patterns=args.custom_patterns,
            force=args.force,
        )
        print(f"\nDataset ready at: {result}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()

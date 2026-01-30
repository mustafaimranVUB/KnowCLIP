from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from zipfile import ZipFile
from dotenv import load_dotenv

from umls_downloader import download_umls_metathesaurus

load_dotenv()


def ensure_metathesaurus(
    *,
    api_key: str,
    version: str | None,
    output_dir: Path,
    extract: bool,
    force: bool,
) -> Path:
    """Download (and optionally extract) the UMLS Metathesaurus archive.

    Parameters
    ----------
    api_key:
            UMLS/UTS API key.
    version:
            UMLS release to fetch (e.g., "2021AB"). If ``None``, the latest
            available version is resolved by ``umls_downloader``.
    output_dir:
            Destination directory for the downloaded ZIP (and extracted files).
    extract:
            When ``True``, unzip the archive into ``output_dir/version``.
    force:
            When ``True``, re-download even if the file already exists.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # ``download_umls_metathesaurus`` stores data under the pystow cache. We
    # copy the ensured ZIP into our project-local ``output_dir`` so downstream
    # code has a stable path.
    ensured_path = download_umls_metathesaurus(
        version=version,
        api_key=api_key,
        force=force,
    )

    local_zip = output_dir / ensured_path.name
    if ensured_path.resolve() != local_zip.resolve():
        shutil.copy2(ensured_path, local_zip)

    if extract:
        target_extract_dir = output_dir / (version or "latest")
        target_extract_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(local_zip) as zf:
            zf.extractall(target_extract_dir)
        return target_extract_dir

    return local_zip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the full UMLS Metathesaurus archive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=os.getenv("UMLS_API_KEY"),
        help=(
            "UMLS/UTS API key. Falls back to the UMLS_API_KEY env var if not "
            "passed explicitly."
        ),
    )
    parser.add_argument(
        "--version",
        dest="version",
        default="2021AB",
        help="UMLS release to fetch (e.g., 2024AB). Uses latest when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("data/umls_metathesaurus"),
        help="Where to place the ZIP (and extraction output).",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the archive after download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key is required (or set UMLS_API_KEY)")

    result_path = ensure_metathesaurus(
        api_key=args.api_key,
        version=args.version,
        output_dir=args.output_dir,
        extract=args.extract,
        force=args.force,
    )

    print(result_path)


if __name__ == "__main__":
    main()

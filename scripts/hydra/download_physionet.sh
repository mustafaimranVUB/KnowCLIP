#!/bin/bash
#SBATCH --job-name=physionet_full_dl
#SBATCH --partition=zen4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ── PhysioNet credentials ──────────────────────────────────────────────────
PHYSIONET_USER="mustafaimraan"
PHYSIONET_PASS="5.i:KKg5YBzP4g." 

# ── Output directory ───────────────────────────────────────────────────────
OUTDIR="${VSC_SCRATCH}/mimic-cxr"
mkdir -p "$OUTDIR"

echo "=== Download started: $(date) ==="
echo "Node: $(hostname)"

download_if_available () {
    local url="$1"
    local outdir="$2"
    wget -N -c --auth-no-challenge --user="$PHYSIONET_USER" --password="$PHYSIONET_PASS" \
         --directory-prefix="$outdir" "$url" && return 0
    return 1
}

# ── 1. Metadata (Small files, wget is fine here) ───────────────────────────
for file in cxr-record-list.csv.gz cxr-study-list.csv.gz cxr-provider-list.csv.gz mimic-cxr-reports.zip
do
    wget -N -c --auth-no-challenge --user="$PHYSIONET_USER" --password="$PHYSIONET_PASS" \
         --directory-prefix="$OUTDIR" \
         "https://physionet.org/files/mimic-cxr/2.1.0/$file"
done

# ── 1b. Split/CheXpert metadata for Phase II ───────────────────────────────
# Prefer mimic-cxr-jpg metadata tables, with version fallback.
for file in mimic-cxr-2.0.0-split.csv.gz mimic-cxr-2.0.0-chexpert.csv.gz
do
        download_if_available "https://physionet.org/files/mimic-cxr-jpg/2.0.0/$file" "$OUTDIR" \
            || download_if_available "https://physionet.org/files/mimic-cxr-jpg/2.1.0/$file" "$OUTDIR" \
            || echo "Warning: could not download $file from mimic-cxr-jpg metadata endpoints"
done

# ── 2. Download full DICOM tree (all studies/labels) ─────────────────────
echo "Downloading full MIMIC-CXR DICOM tree (all label coverage)..."

wget -r -N -c -np -R "index.html*" \
     --auth-no-challenge \
     --user="$PHYSIONET_USER" \
     --password="$PHYSIONET_PASS" \
     --directory-prefix="$OUTDIR" \
     --no-host-directories --cut-dirs=3 \
     "https://physionet.org/files/mimic-cxr/2.1.0/files/"

echo "=== Download finished: $(date) ==="
#!/bin/bash
#SBATCH --job-name=physionet_full_dl
#SBATCH --partition=zen4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tsalaar.2003@gmail.com

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SLURM copies the batch script to its spool dir; BASH_SOURCE[0] then points
# there instead of the repo.  Fall back to SLURM_SUBMIT_DIR when needed.
if [[ ! -f "${SCRIPT_DIR}/common_paths.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/hydra"
fi
# shellcheck source=./common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"

# ── PhysioNet credentials ─────────────────────────────────────────────────
# Set these in your environment (e.g., ~/.bashrc or via sbatch --export):
#   export PHYSIONET_USER="your_username"
#   export PHYSIONET_PASS="your_password"
#
# NEVER hardcode credentials in this script.
if [[ -z "${PHYSIONET_USER:-}" || -z "${PHYSIONET_PASS:-}" ]]; then
    echo "ERROR: PHYSIONET_USER and PHYSIONET_PASS must be set in the environment." >&2
    echo "  Add to ~/.bashrc:" >&2
    echo "    export PHYSIONET_USER=your_username" >&2
    echo "    export PHYSIONET_PASS=your_password" >&2
    exit 1
fi

# ── Output directory ───────────────────────────────────────────────────────
# Writes to $VSC_SCRATCH/mimic-cxr if the legacy scratch is available,
# otherwise to the new THESIS_SCRATCH_ROOT (defined in common_paths.sh).
OUTDIR="${THESIS_PHYSIONET_DOWNLOAD_DIR}"
mkdir -p "${OUTDIR}"

echo "=== PhysioNet Download started: $(date) ==="
echo "Node        : $(hostname)"
echo "Destination : ${OUTDIR}"

download_if_available() {
    local url="$1"
    local outdir="$2"
    wget -N -c --auth-no-challenge \
         --user="${PHYSIONET_USER}" --password="${PHYSIONET_PASS}" \
         --directory-prefix="${outdir}" "${url}" && return 0
    return 1
}

# ── 1. Metadata (small files) ─────────────────────────────────────────────
echo "Downloading metadata files …"
for file in cxr-record-list.csv.gz cxr-study-list.csv.gz cxr-provider-list.csv.gz mimic-cxr-reports.zip; do
    wget -N -c --auth-no-challenge \
         --user="${PHYSIONET_USER}" --password="${PHYSIONET_PASS}" \
         --directory-prefix="${OUTDIR}" \
         "https://physionet.org/files/mimic-cxr/2.1.0/${file}"
done

# ── 1b. Split/CheXpert metadata for Phase II ──────────────────────────────
for file in mimic-cxr-2.0.0-split.csv.gz mimic-cxr-2.0.0-chexpert.csv.gz; do
    download_if_available "https://physionet.org/files/mimic-cxr-jpg/2.0.0/${file}" "${OUTDIR}" \
        || download_if_available "https://physionet.org/files/mimic-cxr-jpg/2.1.0/${file}" "${OUTDIR}" \
        || echo "Warning: could not download ${file} from mimic-cxr-jpg metadata endpoints"
done

# ── 2. Full DICOM tree ────────────────────────────────────────────────────
echo "Downloading full MIMIC-CXR DICOM tree …"
wget -r -N -c -np -R "index.html*" \
     --auth-no-challenge \
     --user="${PHYSIONET_USER}" \
     --password="${PHYSIONET_PASS}" \
     --directory-prefix="${OUTDIR}" \
     --no-host-directories --cut-dirs=3 \
     "https://physionet.org/files/mimic-cxr/2.1.0/files/"

echo "=== Download finished: $(date) ==="

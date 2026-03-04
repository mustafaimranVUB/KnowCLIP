#!/bin/bash
#SBATCH --job-name=physionet_fast_dl
#SBATCH --partition=zen4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── PhysioNet credentials ──────────────────────────────────────────────────
PHYSIONET_USER="mustafaimraan"
PHYSIONET_PASS="5.i:KKg5YBzP4g." 

# ── Output directory ───────────────────────────────────────────────────────
OUTDIR="${VSC_SCRATCH}/mimic-cxr"
mkdir -p "$OUTDIR"

echo "=== Download started: $(date) ==="
echo "Node: $(hostname)"

# ── 1. Metadata (Small files, wget is fine here) ───────────────────────────
for file in cxr-record-list.csv.gz cxr-study-list.csv.gz cxr-provider-list.csv.gz mimic-cxr-reports.zip
do
    wget -N -c --auth-no-challenge --user="$PHYSIONET_USER" --password="$PHYSIONET_PASS" \
         --directory-prefix="$OUTDIR" \
         "https://physionet.org/files/mimic-cxr/2.1.0/$file"
done

# ── 2. Sample 5% of p10 patients from metadata ────────────────────────────
echo "Building 5% patient sample from cxr-record-list.csv..."

# Decompress record list (keep original)
gunzip -kf "$OUTDIR/cxr-record-list.csv.gz"

# Extract unique p10 subject IDs (start with "10"), sample every 20th → 5%
PATIENT_LIST="$OUTDIR/p10_5pct_patients.txt"
awk -F',' 'NR>1 && $1 ~ /^10/ {print $1}' "$OUTDIR/cxr-record-list.csv" \
    | sort -u \
    | awk 'NR % 18 == 1' \
    > "$PATIENT_LIST"

TOTAL=$(wc -l < "$PATIENT_LIST")
echo "Selected ${TOTAL} patients (~5% of p10)"

# ── 3. Download selected patient directories ───────────────────────────────
echo "Starting targeted 5% subset download..."

while IFS= read -r subject_id; do
    URL="https://physionet.org/files/mimic-cxr/2.1.0/files/p10/p${subject_id}/" 
    wget -r -N -c -np -R "index.html*" \
         --auth-no-challenge \
         --user="$PHYSIONET_USER" \
         --password="$PHYSIONET_PASS" \
         --directory-prefix="$OUTDIR" \
         --no-host-directories --cut-dirs=4 \
         "$URL" 2>&1 | grep -v "^--\|reusing\|200 OK\|saved\|Saving" || true
done < "$PATIENT_LIST"

echo "=== Download finished: $(date) ==="
#!/usr/bin/env python3
"""
Compute RaTEScore for all KnoCLIP-XAI evaluation runs.

Reads:  outputs/evaluation/<run>/best_model/generation_predictions.jsonl
Writes: outputs/evaluation/<run>/best_model/ratescore.json
        outputs/evaluation/ratescore_summary.json

Prerequisites (install once in thesis env):
    pip install medspacy ratescore

Usage (from repo root with thesis env active):
    python src/compute_ratescore.py

Reference: Zhao et al. (2024). RaTEScore: A Metric for Radiology Report Generation.
           EMNLP 2024, pages 15004-15019.
"""
import json
import os
import statistics
import torch
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
# On Hydra, THESIS_WRITE_OUTPUT_ROOT is set by common_paths.sh and points to
# the VO outputs directory (/data/brussel/vo/.../thesis/outputs).  Fall back
# to a repo-relative path when running locally.
REPO_ROOT = Path(__file__).resolve().parent.parent
_write_output_root = os.environ.get("THESIS_WRITE_OUTPUT_ROOT")
if _write_output_root:
    EVAL_ROOT = Path(_write_output_root) / "evaluation"
else:
    EVAL_ROOT = REPO_ROOT / "outputs" / "evaluation"

# Runs to evaluate — ablation_genw025 is the proposed model (λ_gen=0.25)
RUNS = [
    "ablation_genw025",             # Proposed model  ← primary thesis entry
    "neurosymbolic_gpt2_hydra_jpg", # Default NS (λ_gen=0.50)
    "ablation_genw100",             # λ_gen=1.00
    "ablation_gat1",                # GATv2 1-head
    "ablation_gat3",                # GATv2 3-head
    "ablation_no_kg_gpt2",          # No KG (vision-only)
]

BATCH_SIZE = 8
USE_GPU = torch.cuda.is_available()

# ── Load scorer ───────────────────────────────────────────────────────────────
print(f"[ratescore] Initialising RaTEScore (GPU={USE_GPU}, batch={BATCH_SIZE}) ...")
try:
    from RaTEScore import RaTEScore as RaTEScorer
except ImportError:
    raise SystemExit(
        "ratescore is not installed.  Run: pip install medspacy ratescore"
    )

scorer = RaTEScorer(use_gpu=USE_GPU, batch_size=BATCH_SIZE)
print("[ratescore] Scorer ready.\n")

results = {}

for run in RUNS:
    jsonl_path = EVAL_ROOT / run / "best_model" / "generation_predictions.jsonl"
    if not jsonl_path.exists():
        print(f"[SKIP] {run}: {jsonl_path} not found")
        results[run] = None
        continue

    references, hypotheses = [], []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            ref = entry.get("reference_report", "").strip()
            hyp = entry.get("generated_report", "").strip()
            if ref and hyp:
                references.append(ref)
                hypotheses.append(hyp)

    if not references:
        print(f"[SKIP] {run}: no valid (reference, hypothesis) pairs found")
        results[run] = None
        continue

    print(f"[{run}] Scoring {len(references)} pairs ...")
    raw = scorer.compute_score(hypotheses, references)

    # scorer.score() may return a float or a list depending on version
    if isinstance(raw, (list, tuple)):
        mean_score = statistics.mean(float(x) for x in raw)
    else:
        mean_score = float(raw)

    entry_out = {"ratescore": round(mean_score, 4), "n_samples": len(references)}
    results[run] = entry_out
    print(f"  → RaTEScore = {mean_score:.4f}  (n={len(references)})")

    out_path = EVAL_ROOT / run / "best_model" / "ratescore.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entry_out, indent=2))
    print(f"  → Saved to {out_path}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
summary_path = EVAL_ROOT / "ratescore_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(results, indent=2))
print(f"Summary written to {summary_path}\n")

print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
for run, r in results.items():
    if r:
        print(f"  {run:<45s}  RaTEScore = {r['ratescore']:.4f}  (n={r['n_samples']})")
    else:
        print(f"  {run:<45s}  SKIPPED (no predictions)")

# Highlight the proposed model
proposed = results.get("ablation_genw025")
if proposed:
    print(f"\n>>> Proposed model (λ_gen=0.25) RaTEScore = {proposed['ratescore']:.4f}")
    print(">>> Enter this value as the thesis Table entry for KnoCLIP-XAI.")
    if proposed["ratescore"] > 0.522:
        print(">>> Value exceeds MedGPT-OSS 0.522 — mark as \\textbf{} in the thesis.")
    else:
        print(">>> Value is below MedGPT-OSS 0.522 — do NOT use \\textbf{} in the thesis.")

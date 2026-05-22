"""
extract_cam_weights.py
======================
Extract per-study CAM_k (Layer 2), GAT alpha (Layer 1), concept pooling weights
(Layer 3), and decoder beta (Layer 4) from a saved KnoCLIP-XAI checkpoint.

Outputs (per study, in --output-dir/<study_id>/):
  cam_k.json          — CAM_k[concept] = List[float] of length 196 (14×14 grid)
  gat_alpha.json      — alpha_ij for every edge in each GAT layer
  pooling_weights.json— alpha^pool[k] per concept node
  decoder_beta.json   — beta[token_idx][memory_pos] for decoder attribution
  concept_meta.json   — node text, CUI, type, certainty for each concept k
  summary.json        — top-3 concepts per class + highest-activated patches

Usage (local test, single study):
    python src/extract_cam_weights.py \
      --config  configs/hydra_phase2_neurosymbolic_gpt2_jpg.yaml \
      --checkpoint outputs/checkpoints/neurosymbolic_gpt2_hydra_jpg/best_model.pt \
      --study   p10000980/s54935705 \
      --output-dir outputs/cam_export

Usage (Hydra, all test-set studies):
    python src/extract_cam_weights.py \
      --config  $CONFIG_PATH \
      --checkpoint $CHECKPOINT \
      --split   test \
      --max-samples 500 \
      --output-dir $OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Project-root on sys.path (works both from repo root and as a Hydra srun)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract KnoCLIP-XAI attention weights")
    p.add_argument("--config",      required=True,  help="YAML config path")
    p.add_argument("--checkpoint",  required=True,  help="Path to best_model.pt")
    p.add_argument("--output-dir",  required=True,  help="Root output directory")
    p.add_argument("--split",       default="test", help="Dataset split (train/validate/test)")
    p.add_argument("--max-samples", type=int, default=50,
                   help="Maximum number of studies to process (0 = all)")
    p.add_argument("--study",       default=None,
                   help="Extract a single study by its 'pXX/sYY' key (overrides --split)")
    p.add_argument("--device",      default="auto",
                   help="'auto', 'cpu', or 'cuda'")
    return p


def _normalize_study_selector(study: Optional[str]) -> Optional[str]:
    """Normalize user-provided study selectors to dataset study_key format.

    Supported inputs:
      - p10000980/s54935705  -> 10000980_54935705
      - 10000980_54935705    -> 10000980_54935705
      - s54935705            -> s54935705 (substring fallback)
    """
    if not study:
        return None
    value = study.strip()
    match = re.fullmatch(r"p(\d+)\s*/\s*s(\d+)", value)
    if match:
        return f"{int(match.group(1))}_{int(match.group(2))}"
    return value


def _normalize_split_name(split: str) -> str:
    normalized = (split or "test").strip().lower()
    if normalized in {"val", "valid", "validation"}:
        return "validate"
    if normalized in {"train", "validate", "test"}:
        return normalized
    return "test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _cam_k_from_fusion_trace(fusion_trace: Dict[str, Any], num_patches: int = 196) -> Dict[int, List[float]]:
    """
    Compute CAM_k = (1/H) * sum_h A^(L,h)[k, :] from fusion trace.

    fusion_trace["last_layer"] is shape (B, H, K, P).
    Returns {concept_idx: [p_0, ..., p_195]}  (B=1 assumed).
    """
    last_layer = fusion_trace["last_layer"]  # (B, H, K, P)
    # Take mean over heads → (B, K, P), then squeeze batch
    cam = last_layer.mean(dim=1).squeeze(0)  # (K, P)
    cam_k: Dict[int, List[float]] = {}
    for k in range(cam.shape[0]):
        cam_k[k] = cam[k].cpu().float().tolist()
    return cam_k


def _gat_alpha_from_graph_trace(graph_trace: List[Dict[str, Any]]) -> List[Dict]:
    """
    Convert graph_trace (list of layer dicts) to serialisable form.
    Each layer dict: {"edge_index": (2,E), "alpha": (E, H) or (E,)}.
    Returns list[{layer, edge_src, edge_dst, alpha_per_head}].
    """
    result = []
    for layer_idx, layer_info in enumerate(graph_trace):
        edge_index = layer_info["edge_index"]   # (2, E)
        alpha      = layer_info["alpha"]        # (E, H) or (E,)
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)         # (E, 1)
        src = edge_index[0].cpu().tolist()
        dst = edge_index[1].cpu().tolist()
        alp = alpha.cpu().float().tolist()
        result.append({
            "layer": layer_idx,
            "edges": [
                {"src": s, "dst": d, "alpha": a}
                for s, d, a in zip(src, dst, alp)
            ]
        })
    return result


def _decoder_beta_from_explainability(
    explainability: Dict[str, Any],
    token_labels: List[str],
    num_concepts: int,
) -> Optional[Dict]:
    """
    Extract decoder cross-attention weights if available.
    Returns dict: {token_idx: {token_text, concept_attributions, patch_attributions}}
    """
    dec = explainability.get("decoder")
    if dec is None:
        return None
    # dec["beta"] shape varies by implementation; handle gracefully
    beta = dec.get("cross_attn_weights") or dec.get("beta")
    if beta is None:
        return None
    if isinstance(beta, torch.Tensor):
        beta = beta.cpu().float()  # (T, M) or (L, T, M)
        if beta.dim() == 3:
            beta = beta.mean(dim=0)  # average over decoder layers → (T, M)
        result = {}
        for t_idx, t_weights in enumerate(beta):
            t_label = token_labels[t_idx] if t_idx < len(token_labels) else f"tok_{t_idx}"
            w = t_weights.tolist()
            result[t_idx] = {
                "token": t_label,
                "concept_attributions": w[:num_concepts],
                "patch_attributions":   w[num_concepts:],
            }
        return result
    return None


def _extract_one_study(
    model,
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Run one forward pass with return_attention=True and return raw traces."""
    with torch.no_grad():
        pixel_values = batch["image"].to(device)
        outputs = model(
            pixel_values=pixel_values,
            graph_x=batch.get("graph_x").to(device) if isinstance(batch.get("graph_x"), torch.Tensor) else None,
            graph_edge_index=batch.get("graph_edge_index").to(device) if isinstance(batch.get("graph_edge_index"), torch.Tensor) else None,
            graph_edge_type=batch.get("graph_edge_type").to(device) if isinstance(batch.get("graph_edge_type"), torch.Tensor) else None,
            graph_batch=batch.get("graph_batch").to(device) if isinstance(batch.get("graph_batch"), torch.Tensor) else None,
            graph_num_nodes_per_sample=batch.get("graph_num_nodes_per_sample"),
            return_attention=True,
        )

    return outputs


def _save_study_artefacts(
    outputs: Dict[str, Any],
    batch:   Dict[str, Any],
    study_dir: Path,
) -> None:
    """Parse outputs dict and write JSON artefacts to study_dir."""
    study_dir.mkdir(parents=True, exist_ok=True)

    xai = outputs.get("explainability", {})

    # ── concept metadata ────────────────────────────────────────────────────
    meta = {
        "node_texts":      batch.get("graph_node_texts",      [[]]),
        "node_cuis":       batch.get("graph_node_cuis",       [[]]),
        "node_types":      batch.get("graph_node_types",      [[]]),
        "node_certainties":batch.get("graph_node_certainties",[[]]),
    }
    # Take first sample from batch (batch size 1)
    concept_meta = {
        "node_texts":      meta["node_texts"][0]       if meta["node_texts"]      else [],
        "node_cuis":       meta["node_cuis"][0]        if meta["node_cuis"]       else [],
        "node_types":      meta["node_types"][0]       if meta["node_types"]      else [],
        "node_certainties":meta["node_certainties"][0] if meta["node_certainties"] else [],
    }
    (study_dir / "concept_meta.json").write_text(json.dumps(concept_meta, indent=2))

    num_concepts = len(concept_meta["node_texts"])

    # ── Layer 1: GAT alpha ──────────────────────────────────────────────────
    graph_trace = xai.get("graph", {})
    graph_layers = graph_trace.get("edge_attention_layers", []) if isinstance(graph_trace, dict) else []
    if graph_layers:
        gat_data = _gat_alpha_from_graph_trace(graph_layers)
        (study_dir / "gat_alpha.json").write_text(json.dumps(gat_data, indent=2))

    # ── Layer 2: CAM_k ──────────────────────────────────────────────────────
    fusion_trace = xai.get("fusion", {})
    if fusion_trace:
        cam_k = _cam_k_from_fusion_trace(fusion_trace)
        # Annotate with concept text where available
        texts = concept_meta["node_texts"]
        cam_named = {
            texts[k] if k < len(texts) else f"concept_{k}": cam_k[k]
            for k in cam_k
        }
        (study_dir / "cam_k.json").write_text(json.dumps(cam_named, indent=2))

    # ── Layer 3: pooling weights ─────────────────────────────────────────────
    pool_info = xai.get("pooling", {})
    if pool_info:
        pool_weights = pool_info.get("weights")
        if pool_weights is not None:
            if isinstance(pool_weights, torch.Tensor):
                pool_weights = pool_weights.squeeze(0).cpu().float().tolist()
            texts = concept_meta["node_texts"]
            pooling_out = {
                texts[k] if k < len(texts) else f"concept_{k}": float(pool_weights[k])
                for k in range(len(pool_weights))
            }
            (study_dir / "pooling_weights.json").write_text(json.dumps(pooling_out, indent=2))

    # ── Layer 4: decoder beta ────────────────────────────────────────────────
    token_ids   = outputs.get("generated_token_ids", [])
    token_labels = batch.get("generated_token_labels", [str(t) for t in token_ids])
    beta_data = _decoder_beta_from_explainability(xai, token_labels, num_concepts)
    if beta_data:
        (study_dir / "decoder_beta.json").write_text(json.dumps(beta_data, indent=2))

    # ── Summary: top concepts + patches per class ────────────────────────────
    cls_logits = outputs.get("classification_logits")
    cls_probs = []
    if cls_logits is not None:
        cls_probs = torch.sigmoid(cls_logits).squeeze(0).cpu().float().tolist()

    summary: Dict[str, Any] = {
        "study_key":    (batch.get("study_key", ["unknown"])[0] if isinstance(batch.get("study_key"), list) else batch.get("study_key", "unknown")),
        "class_probs":  cls_probs,
        "generated":    outputs.get("generated_report", ""),
        "reference":    batch.get("report_text", [""])[0] if batch.get("report_text") else "",
    }

    if fusion_trace:
        # Top-5 patches for each concept
        cam_k_flat = {k: v for k, v in cam_k.items()}
        texts = concept_meta["node_texts"]
        top_patches = {}
        for k, patches in cam_k_flat.items():
            arr = np.array(patches)
            top5_idx = arr.argsort()[-5:][::-1].tolist()
            name = texts[k] if k < len(texts) else f"concept_{k}"
            top_patches[name] = {
                "top5_patch_indices": top5_idx,
                "top5_patch_rows_cols": [
                    {"row": p // 14, "col": p % 14} for p in top5_idx
                ],
                "top5_cam_values": [float(patches[p]) for p in top5_idx],
            }
        summary["top_patches_per_concept"] = top_patches

    (study_dir / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_arg_parser().parse_args()
    device = _resolve_device(args.device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────────
    from src.core.config import load_config
    config = load_config(args.config)

    # ── Load model ───────────────────────────────────────────────────────────
    from src.models.model_factory import build_model
    model = build_model(config.model)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print(f"[extract] Loaded checkpoint: {args.checkpoint}")
    print(f"[extract] Model on: {device}")

    # ── Build dataset ────────────────────────────────────────────────────────
    from src.data.dataset import MIMICCXRDataset, collate_mimic
    from src.data.transforms import get_eval_transforms
    from torch.utils.data import DataLoader

    split_name = _normalize_split_name(args.split if args.study is None else "test")
    dc = config.data

    def _build_dataset(split: str) -> MIMICCXRDataset:
        return MIMICCXRDataset(
            split=_normalize_split_name(split),
            mimic_root=dc.mimic_root,
            reports_root=dc.reports_root,
            split_csv=dc.split_csv,
            chexpert_csv=dc.chexpert_csv,
            kg_artifacts_dir=dc.kg_artifacts_dir,
            transforms=get_eval_transforms(),
            include_graphs=config.model.use_kg,
            split_strategy=dc.split_strategy,
            subset_seed=dc.subset_seed,
            subset_train_ratio=dc.subset_train_ratio,
            subset_val_ratio=dc.subset_val_ratio,
            subset_test_ratio=dc.subset_test_ratio,
            auto_min_val_samples=dc.auto_min_val_samples,
            auto_min_test_samples=dc.auto_min_test_samples,
            image_suffixes=dc.image_suffixes,
            # For extraction we intentionally keep this off so real-world study IDs
            # (possibly with sparse labels) are still discoverable.
            enforce_all_labels_per_split=False,
        )

    dataset = _build_dataset(split_name)

    if args.study is not None:
        study_selector = _normalize_study_selector(args.study)
        # Filter to the requested study key
        filtered = [
            i for i, s in enumerate(dataset.samples)
            if (study_selector in str(s.get("study_key", "")))
        ]
        if not filtered:
            candidate_splits = [sp for sp in ["test", "validate", "train"] if sp != split_name]
            found_elsewhere = False
            for candidate_split in candidate_splits:
                candidate_dataset = _build_dataset(candidate_split)
                candidate_filtered = [
                    i for i, s in enumerate(candidate_dataset.samples)
                    if (study_selector in str(s.get("study_key", "")))
                ]
                if candidate_filtered:
                    print(
                        f"[extract] NOTE: study '{args.study}' not in '{split_name}', "
                        f"found in '{candidate_split}'."
                    )
                    dataset = candidate_dataset
                    split_name = candidate_split
                    filtered = candidate_filtered
                    found_elsewhere = True
                    break

            if not found_elsewhere:
                print(f"[extract] WARNING: study '{args.study}' not found in any split.")
                print("[extract] Available study keys (first 10 from test split):")
                for s in dataset.samples[:10]:
                    print(f"  {s.get('study_key', 'unknown')}")
                sys.exit(1)
        from torch.utils.data import Subset
        dataset = Subset(dataset, filtered)
        print(f"[extract] Filtered to {len(dataset)} study matching '{args.study}' in split '{split_name}'")

    max_n = args.max_samples if args.max_samples > 0 else len(dataset)
    dataset = torch.utils.data.Subset(dataset, list(range(min(max_n, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=functools.partial(
            collate_mimic,
            max_report_length=config.model.report_generation.max_report_length,
        ),
    )

    # ── Iterate and extract ──────────────────────────────────────────────────
    processed = 0
    for batch_idx, batch in enumerate(loader):
        study_key = (
            batch.get("study_key", [f"study_{batch_idx:04d}"])[0]
            if isinstance(batch.get("study_key"), (list, tuple))
            else batch.get("study_key", f"study_{batch_idx:04d}")
        )
        # Sanitise for filesystem
        safe_key = str(study_key).replace("/", "_").replace(" ", "_")
        study_dir = output_root / safe_key

        print(f"[extract] {batch_idx+1}/{len(loader)}  {study_key}")

        try:
            outputs = _extract_one_study(model, batch, device)
            _save_study_artefacts(outputs, batch, study_dir)
            processed += 1
        except Exception as exc:
            print(f"[extract] SKIP {study_key}: {exc}")
            (study_dir / "error.txt").parent.mkdir(parents=True, exist_ok=True)
            (study_dir / "error.txt").write_text(str(exc))

    print(f"[extract] Done. Processed {processed}/{len(loader)} studies → {output_root}")


if __name__ == "__main__":
    main()

"""Export self-explainability traces as figures and summaries."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.transforms import denormalise
from src.knowledge.graph_builder import EDGE_TYPE_MAP

EDGE_TYPE_LABELS = {value: key for key, value in EDGE_TYPE_MAP.items()}


@dataclass
class ExplainabilitySample:
    """Serializable payload for one explainability export."""

    study_key: str
    image: torch.Tensor
    explainability: Dict[str, Any]
    class_names: List[str] = field(default_factory=list)
    classification_probs: Optional[torch.Tensor] = None
    reference_report: str = ""
    generated_report: str = ""
    generated_token_ids: List[int] = field(default_factory=list)
    generated_token_labels: List[str] = field(default_factory=list)
    graph_node_texts: List[str] = field(default_factory=list)
    graph_node_cuis: List[str] = field(default_factory=list)
    graph_node_types: List[str] = field(default_factory=list)
    graph_node_certainties: List[str] = field(default_factory=list)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "sample"


def _detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_image_numpy(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    restored = denormalise(image.detach().cpu()).clamp(0.0, 1.0)
    return restored.permute(1, 2, 0).numpy()


def _truncate(label: str, max_length: int = 72) -> str:
    return label if len(label) <= max_length else f"{label[: max_length - 3]}..."


def _build_concept_labels(sample: ExplainabilitySample, concept_count: int) -> List[str]:
    labels: List[str] = []
    for index in range(concept_count):
        text = sample.graph_node_texts[index] if index < len(sample.graph_node_texts) else f"concept_{index}"
        cui = sample.graph_node_cuis[index] if index < len(sample.graph_node_cuis) else ""
        node_type = sample.graph_node_types[index] if index < len(sample.graph_node_types) else ""
        certainty = sample.graph_node_certainties[index] if index < len(sample.graph_node_certainties) else ""

        metadata = [value for value in (node_type, cui, certainty) if value]
        label = text
        if metadata:
            label = f"{label} ({', '.join(metadata)})"
        labels.append(_truncate(label))
    return labels


def _extract_pooling(sample: ExplainabilitySample) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    pooling = sample.explainability.get("pooling")
    if not isinstance(pooling, dict):
        return None, None

    weights = pooling.get("weights")
    mask = pooling.get("mask")
    if not isinstance(weights, torch.Tensor):
        return None, None

    weights_cpu = weights.detach().cpu()
    mask_cpu = mask.detach().cpu() if isinstance(mask, torch.Tensor) else None

    if weights_cpu.ndim == 2:
        weights_cpu = weights_cpu[0]
    if mask_cpu is not None and mask_cpu.ndim == 2:
        mask_cpu = mask_cpu[0]

    return weights_cpu, mask_cpu


def _top_indices(values: torch.Tensor, top_k: int) -> List[int]:
    if values.numel() == 0:
        return []
    count = min(top_k, int(values.numel()))
    return torch.topk(values, k=count).indices.tolist()


def _reshape_patch_map(values: torch.Tensor) -> Optional[np.ndarray]:
    patch_count = int(values.numel())
    side = int(math.isqrt(patch_count))
    if side * side != patch_count:
        return None
    return values.detach().cpu().reshape(side, side).numpy()


class ExplainabilityExporter:
    """Save sample-level explainability plots and summaries."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        top_k_pathologies: int = 5,
        top_k_concepts: int = 6,
        top_k_edges: int = 8,
        top_k_decoder_tokens: int = 4,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_k_pathologies = top_k_pathologies
        self.top_k_concepts = top_k_concepts
        self.top_k_edges = top_k_edges
        self.top_k_decoder_tokens = top_k_decoder_tokens

    def export_sample(self, sample: ExplainabilitySample) -> Dict[str, Any]:
        sample_dir = self.output_dir / _slugify(sample.study_key)
        sample_dir.mkdir(parents=True, exist_ok=True)

        image_np = _as_image_numpy(sample.image)
        image_path = sample_dir / "input_image.png"
        self._save_image(image_np, image_path)

        top_pathologies = self._save_classification_scores(sample, sample_dir)
        top_concepts = self._save_concept_importance(sample, sample_dir)
        self._save_visual_pooling_overlay(sample, sample_dir, image_np)
        self._save_cross_attention_overlays(sample, sample_dir, image_np, top_concepts)
        concept_cams = self._save_concept_cam_overlays(sample, sample_dir, image_np, top_concepts)
        fusion_summary = self._save_fusion_activation_summaries(sample, sample_dir, top_concepts)
        top_edges = self._save_graph_attention(sample, sample_dir)
        self._save_attention_vector(sample, sample_dir)
        self._save_decoder_attention(sample, sample_dir, image_np, top_concepts)

        trace_path = sample_dir / "trace_bundle.pt"
        torch.save(
            {
                "study_key": sample.study_key,
                "classification_probs": _detach_tree(sample.classification_probs),
                "generated_token_ids": sample.generated_token_ids,
                "generated_token_labels": sample.generated_token_labels,
                "graph_node_texts": sample.graph_node_texts,
                "graph_node_cuis": sample.graph_node_cuis,
                "graph_node_types": sample.graph_node_types,
                "graph_node_certainties": sample.graph_node_certainties,
                "explainability": _detach_tree(sample.explainability),
            },
            trace_path,
        )

        summary = {
            "study_key": sample.study_key,
            "sample_dir": str(sample_dir),
            "top_pathologies": top_pathologies,
            "top_concepts": top_concepts,
            "concept_cams": concept_cams,
            "fusion_summary": fusion_summary.get("stats", {}),
            "top_edges": top_edges,
            "reference_report": sample.reference_report,
            "generated_report": sample.generated_report,
            "generated_token_labels": sample.generated_token_labels,
            "files": {
                "input_image": str(image_path),
                "trace_bundle": str(trace_path),
            },
        }

        summary_json = sample_dir / "summary.json"
        summary_json.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        summary["files"]["summary_json"] = str(summary_json)

        for key, value in fusion_summary.get("files", {}).items():
            summary["files"][key] = str(value)

        # Register every PNG in the sample directory so the API can encode them all.
        for png_path in sorted(sample_dir.glob("*.png")):
            key = png_path.stem
            if key not in summary["files"]:
                summary["files"][key] = str(png_path)

        summary_md = sample_dir / "summary.md"
        self._write_summary_markdown(sample, summary, summary_md)
        summary["files"]["summary_markdown"] = str(summary_md)

        return summary

    def _save_image(self, image_np: np.ndarray, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np)
        ax.set_title("Input CXR")
        ax.axis("off")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_classification_scores(self, sample: ExplainabilitySample, sample_dir: Path) -> List[Dict[str, float]]:
        if sample.classification_probs is None or not sample.class_names:
            return []

        scores = sample.classification_probs.detach().cpu().flatten()
        count = min(self.top_k_pathologies, len(sample.class_names), int(scores.numel()))
        if count == 0:
            return []

        top_values, top_indices = torch.topk(scores, k=count)
        labels = [sample.class_names[idx] for idx in top_indices.tolist()]

        fig, ax = plt.subplots(figsize=(10, 4 + 0.35 * count), constrained_layout=True)
        ax.barh(labels[::-1], top_values.tolist()[::-1], color="#9e2a2b")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Predicted probability")
        ax.set_title("Top pathology probabilities")

        output_path = sample_dir / "classification_scores.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return [
            {"label": labels[i], "score": float(top_values[i])}
            for i in range(count)
        ]

    def _save_concept_importance(self, sample: ExplainabilitySample, sample_dir: Path) -> List[Dict[str, float]]:
        weights, mask = _extract_pooling(sample)
        if weights is None:
            return []

        if mask is not None:
            valid_count = int(mask.sum().item())
            weights = weights[:valid_count]

        if weights.numel() == 0 or not sample.graph_node_texts:
            return []

        concept_count = min(len(sample.graph_node_texts), int(weights.numel()))
        weights = weights[:concept_count]
        labels = _build_concept_labels(sample, concept_count)
        indices = _top_indices(weights, self.top_k_concepts)
        top_labels = [labels[index] for index in indices]
        top_values = [float(weights[index]) for index in indices]

        fig, ax = plt.subplots(figsize=(12, 4 + 0.45 * len(indices)), constrained_layout=True)
        ax.barh(top_labels[::-1], top_values[::-1], color="#386641")
        ax.set_xlabel("Attention pooling weight")
        ax.set_title("Top KG concepts")

        output_path = sample_dir / "concept_importance.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return [
            {"index": index, "label": labels[index], "weight": float(weights[index])}
            for index in indices
        ]

    def _save_visual_pooling_overlay(
        self,
        sample: ExplainabilitySample,
        sample_dir: Path,
        image_np: np.ndarray,
    ) -> None:
        if sample.graph_node_texts:
            return

        weights, mask = _extract_pooling(sample)
        if weights is None:
            return
        if mask is not None:
            weights = weights[: int(mask.sum().item())]
        heatmap = _reshape_patch_map(weights)
        if heatmap is None:
            return

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax.imshow(image_np)
        ax.imshow(heatmap, cmap="magma", alpha=0.55, interpolation="bilinear")
        ax.set_title("Visual pooling attention")
        ax.axis("off")
        output_path = sample_dir / "visual_pooling_overlay.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_cross_attention_overlays(
        self,
        sample: ExplainabilitySample,
        sample_dir: Path,
        image_np: np.ndarray,
        top_concepts: List[Dict[str, float]],
    ) -> None:
        fusion = sample.explainability.get("fusion")
        if not isinstance(fusion, dict) or not top_concepts:
            return

        last_layer = fusion.get("last_layer")
        if not isinstance(last_layer, torch.Tensor):
            return

        attn = last_layer.detach().cpu()
        if attn.ndim != 4:
            return

        averaged = attn[0].mean(dim=0)
        plot_count = min(len(top_concepts), self.top_k_concepts)
        fig, axes = plt.subplots(1, plot_count, figsize=(5 * plot_count, 5), squeeze=False, constrained_layout=True)

        for axis, concept in zip(axes[0], top_concepts[:plot_count]):
            concept_index = int(concept["index"])
            if concept_index >= averaged.shape[0]:
                axis.axis("off")
                continue
            heatmap = _reshape_patch_map(averaged[concept_index])
            if heatmap is None:
                axis.axis("off")
                continue
            axis.imshow(image_np)
            axis.imshow(heatmap, cmap="magma", alpha=0.55, interpolation="bilinear")
            axis.set_title(_truncate(concept["label"], 42))
            axis.axis("off")

        output_path = sample_dir / "cross_attention_overlays.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_concept_cam_overlays(
        self,
        sample: ExplainabilitySample,
        sample_dir: Path,
        image_np: np.ndarray,
        top_concepts: List[Dict[str, float]],
        top_k: int = 8,
    ) -> List[Dict[str, str]]:
        """Save one inherent Cam_k overlay per top concept (matches thesis Figure 3.6 style).

        Uses cross-attention weights from the fusion layer (not a post-hoc gradient method).
        Each output is saved as cam_{slug}.png and also registered in summary["files"].
        """
        fusion = sample.explainability.get("fusion")
        if not isinstance(fusion, dict) or not top_concepts:
            return []
        last_layer = fusion.get("last_layer")
        if not isinstance(last_layer, torch.Tensor) or last_layer.ndim != 4:
            return []

        # [batch, heads, K_concepts, P_patches] → avg over batch+heads → [K, P]
        attn = last_layer.detach().cpu()[0].mean(dim=0)

        h, w = image_np.shape[:2]
        side = int(math.isqrt(attn.shape[1]))  # 14 for 196 patches

        # Render on grayscale background (cleaner for heatmap overlay)
        gray = np.mean(image_np, axis=2)
        gray_rgb = np.stack([gray, gray, gray], axis=2)

        cam_files: List[Dict[str, str]] = []

        for concept in top_concepts[:top_k]:
            concept_index = int(concept["index"])
            if concept_index >= attn.shape[0]:
                continue
            heatmap = _reshape_patch_map(attn[concept_index])
            if heatmap is None:
                continue

            # Normalize to [0, 1]
            hm_min, hm_max = float(heatmap.min()), float(heatmap.max())
            if hm_max > hm_min:
                heatmap = (heatmap - hm_min) / (hm_max - hm_min)
            else:
                continue  # uniform attention — skip

            concept_text = concept["label"].split("(")[0].strip()
            slug = _slugify(concept_text)[:48]
            file_key = f"cam_{slug}"

            fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
            ax.imshow(gray_rgb)
            ax.imshow(
                heatmap,
                cmap="Reds",
                alpha=0.60,
                interpolation="bilinear",
                vmin=0,
                vmax=1,
                extent=[0, w, h, 0],
            )

            # Patch grid lines (same as thesis figure)
            for i in range(1, side):
                ax.axvline(x=i * w / side, color="white", linewidth=0.35, alpha=0.45)
                ax.axhline(y=i * h / side, color="white", linewidth=0.35, alpha=0.45)

            # Dashed bounding box around the hot region (≥ 0.5 normalised)
            hot = np.argwhere(heatmap >= 0.5)
            if len(hot) >= 2:
                r_min, c_min = hot.min(axis=0)
                r_max, c_max = hot.max(axis=0)
                rx = c_min * w / side
                ry = r_min * h / side
                rw = (c_max - c_min + 1) * w / side
                rh = (r_max - r_min + 1) * h / side
                rect = plt.Rectangle(
                    (rx, ry), rw, rh,
                    linewidth=1.6, edgecolor="white", facecolor="none",
                    linestyle="--", alpha=0.85,
                )
                ax.add_patch(rect)

            ax.set_title(f"Cam$_k$: {concept_text}", fontsize=11)
            ax.axis("off")

            out_path = sample_dir / f"{file_key}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            cam_files.append({
                "label": concept["label"],
                "concept_text": concept_text,
                "key": file_key,
                "path": str(out_path),
            })

        return cam_files

    def _save_fusion_activation_summaries(
        self,
        sample: ExplainabilitySample,
        sample_dir: Path,
        top_concepts: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        fusion = sample.explainability.get("fusion")
        if not isinstance(fusion, dict):
            return {"files": {}, "stats": {}}

        per_layer = fusion.get("per_layer")
        last_layer = fusion.get("last_layer")
        if not isinstance(per_layer, list) or not per_layer:
            return {"files": {}, "stats": {}}

        layer_tensors = [
            layer.detach().cpu()
            for layer in per_layer
            if isinstance(layer, torch.Tensor) and layer.ndim == 4
        ]
        if not layer_tensors:
            return {"files": {}, "stats": {}}

        files: Dict[str, str] = {}
        stats: Dict[str, Any] = {}

        # Layer x Head mean attention strengths (middle fusion activations).
        layer_head_strengths = []
        for layer in layer_tensors:
            # layer[0]: (H, K, P)
            layer_head_strengths.append(layer[0].mean(dim=(1, 2)).numpy())
        strength_matrix = np.stack(layer_head_strengths, axis=0)  # (L, H)

        fig, ax = plt.subplots(
            figsize=(1.2 * strength_matrix.shape[1] + 4, 0.8 * strength_matrix.shape[0] + 3),
            constrained_layout=True,
        )
        image = ax.imshow(strength_matrix, cmap="magma", aspect="auto")
        ax.set_xlabel("Head")
        ax.set_ylabel("Fusion layer")
        ax.set_title("Fusion activation strength (layer x head)")
        ax.set_xticks(np.arange(strength_matrix.shape[1]))
        ax.set_yticks(np.arange(strength_matrix.shape[0]))
        fig.colorbar(image, ax=ax, shrink=0.85)
        layer_head_path = sample_dir / "fusion_layer_head_attention.png"
        fig.savefig(layer_head_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        files["fusion_layer_head_attention"] = str(layer_head_path)

        if isinstance(last_layer, torch.Tensor) and last_layer.ndim == 4:
            last = last_layer.detach().cpu()[0]  # (H, K, P)
            mean_over_heads = last.mean(dim=0)  # (K, P)
            num_queries = int(mean_over_heads.shape[0])
            num_patches = int(mean_over_heads.shape[1])

            if top_concepts:
                selected_indices = [
                    int(concept["index"])
                    for concept in top_concepts
                    if 0 <= int(concept["index"]) < num_queries
                ]
                query_labels = [
                    _truncate(concept["label"], 48)
                    for concept in top_concepts
                    if 0 <= int(concept["index"]) < num_queries
                ]
            else:
                query_strength = mean_over_heads.mean(dim=1)
                selected_indices = _top_indices(query_strength, top_k=min(10, num_queries))
                query_labels = [f"query_{index}" for index in selected_indices]

            if selected_indices:
                selected = mean_over_heads[selected_indices].numpy()
                fig, ax = plt.subplots(
                    figsize=(min(16, 0.06 * selected.shape[1] + 6), 0.6 * selected.shape[0] + 3),
                    constrained_layout=True,
                )
                image = ax.imshow(selected, cmap="viridis", aspect="auto")
                ax.set_xlabel("Visual patch index")
                ax.set_ylabel("Concept/query")
                ax.set_title("Fusion query-to-patch attention (last layer, mean over heads)")
                ax.set_yticks(np.arange(len(query_labels)))
                ax.set_yticklabels(query_labels)
                fig.colorbar(image, ax=ax, shrink=0.85)
                query_patch_path = sample_dir / "fusion_query_patch_attention.png"
                fig.savefig(query_patch_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                files["fusion_query_patch_attention"] = str(query_patch_path)

            stats = {
                "num_fusion_layers": len(layer_tensors),
                "num_attention_heads": int(last.shape[0]),
                "num_queries": num_queries,
                "num_visual_patches": num_patches,
                "max_layer_head_attention": float(np.max(strength_matrix)),
                "mean_layer_head_attention": float(np.mean(strength_matrix)),
            }

        return {"files": files, "stats": stats}

    def _save_graph_attention(self, sample: ExplainabilitySample, sample_dir: Path) -> List[Dict[str, float]]:
        graph = sample.explainability.get("graph")
        if not isinstance(graph, dict):
            return []

        layers = graph.get("edge_attention_layers")
        if not isinstance(layers, list) or not layers:
            return []

        concept_count = len(sample.graph_node_texts)
        node_labels = _build_concept_labels(sample, concept_count)
        summaries: List[Dict[str, float]] = []

        fig, axes = plt.subplots(len(layers), 1, figsize=(14, 4 * len(layers)), squeeze=False, constrained_layout=True)

        for axis, layer in zip(axes[:, 0], layers):
            edge_index = layer.get("edge_index")
            alpha = layer.get("alpha")
            edge_type = layer.get("edge_type")
            if not isinstance(edge_index, torch.Tensor) or not isinstance(alpha, torch.Tensor):
                axis.axis("off")
                continue

            edge_index = edge_index.detach().cpu()
            alpha = alpha.detach().cpu()
            edge_type = edge_type.detach().cpu() if isinstance(edge_type, torch.Tensor) else None

            edge_scores = alpha.mean(dim=-1) if alpha.ndim > 1 else alpha
            if edge_scores.numel() == 0:
                axis.axis("off")
                continue

            top_indices = _top_indices(edge_scores, self.top_k_edges)
            labels: List[str] = []
            values: List[float] = []

            for edge_idx in top_indices:
                src = int(edge_index[0, edge_idx])
                dst = int(edge_index[1, edge_idx])
                relation = "edge"
                if edge_type is not None and edge_idx < edge_type.numel():
                    relation = EDGE_TYPE_LABELS.get(int(edge_type[edge_idx]), f"type_{int(edge_type[edge_idx])}")
                src_label = node_labels[src] if src < len(node_labels) else f"node_{src}"
                dst_label = node_labels[dst] if dst < len(node_labels) else f"node_{dst}"
                labels.append(_truncate(f"{src_label} -> {dst_label} [{relation}]", 84))
                values.append(float(edge_scores[edge_idx]))
                if layer is layers[-1]:
                    summaries.append({
                        "source": src_label,
                        "target": dst_label,
                        "relation": relation,
                        "score": float(edge_scores[edge_idx]),
                    })

            axis.barh(labels[::-1], values[::-1], color="#335c67")
            axis.set_title(f"Layer {int(layer.get('layer_index', 0))} top graph-attention edges")
            axis.set_xlabel("Mean edge attention")

        output_path = sample_dir / "graph_attention.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return summaries

    def _save_attention_vector(self, sample: ExplainabilitySample, sample_dir: Path) -> None:
        graph = sample.explainability.get("graph")
        if not isinstance(graph, dict):
            return

        layers = graph.get("edge_attention_layers")
        if not isinstance(layers, list) or not layers:
            return

        fig, axes = plt.subplots(len(layers), 1, figsize=(12, 3 * len(layers)), squeeze=False, constrained_layout=True)
        for axis, layer in zip(axes[:, 0], layers):
            vector = layer.get("attention_vector")
            if not isinstance(vector, torch.Tensor):
                axis.axis("off")
                continue

            values = vector.detach().cpu().squeeze(0)
            if values.ndim == 1:
                values = values.unsqueeze(0)
            image = axis.imshow(values.numpy(), cmap="coolwarm", aspect="auto")
            axis.set_ylabel("Head")
            axis.set_xlabel("Feature channel")
            axis.set_title(f"GATv2 attention vector a, layer {int(layer.get('layer_index', 0))}")
            fig.colorbar(image, ax=axis, shrink=0.85)

        output_path = sample_dir / "gat_attention_vector.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_decoder_attention(
        self,
        sample: ExplainabilitySample,
        sample_dir: Path,
        image_np: np.ndarray,
        top_concepts: List[Dict[str, float]],
    ) -> None:
        decoder = sample.explainability.get("decoder")
        if not isinstance(decoder, dict):
            return

        per_layer = decoder.get("per_layer")
        if not isinstance(per_layer, list) or not per_layer:
            return

        layer_tensors = [layer.detach().cpu() for layer in per_layer if isinstance(layer, torch.Tensor) and layer.ndim == 4]
        if not layer_tensors:
            return

        averaged = torch.stack([layer[0] for layer in layer_tensors], dim=0).mean(dim=(0, 1))
        concept_count = min(len(sample.graph_node_texts), averaged.shape[1])
        token_labels = sample.generated_token_labels[: averaged.shape[0]] or [f"token_{i}" for i in range(averaged.shape[0])]

        # Remove an explicit BOS row from the visualisations.
        if token_labels and token_labels[0].strip().lower() in {"<bos>", "bos"} and averaged.shape[0] > 1:
            averaged = averaged[1:]
            token_labels = token_labels[1:]

        if concept_count > 0 and top_concepts:
            selected = [concept for concept in top_concepts if int(concept["index"]) < concept_count]
            if selected:
                concept_indices = [int(concept["index"]) for concept in selected]
                concept_labels = [_truncate(concept["label"], 48) for concept in selected]
                concept_attention = averaged[:, concept_indices].numpy()

                fig, ax = plt.subplots(figsize=(1.2 * len(concept_indices) + 6, 0.45 * len(token_labels) + 4), constrained_layout=True)
                image = ax.imshow(concept_attention, cmap="viridis", aspect="auto")
                ax.set_yticks(np.arange(len(token_labels)))
                ax.set_yticklabels([_truncate(label, 36) for label in token_labels])
                ax.set_xticks(np.arange(len(concept_labels)))
                ax.set_xticklabels(concept_labels, rotation=45, ha="right")
                ax.set_title("Decoder token-to-concept attention")
                fig.colorbar(image, ax=ax, shrink=0.85)
                output_path = sample_dir / "decoder_concepts.png"
                fig.savefig(output_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

        visual_attention = averaged[:, concept_count:]
        if visual_attention.numel() == 0:
            return

        token_scores = visual_attention.sum(dim=-1)
        top_token_indices = _top_indices(token_scores, self.top_k_decoder_tokens)
        fig, axes = plt.subplots(1, len(top_token_indices), figsize=(5 * len(top_token_indices), 5), squeeze=False, constrained_layout=True)
        for axis, token_index in zip(axes[0], top_token_indices):
            heatmap = _reshape_patch_map(visual_attention[token_index])
            if heatmap is None:
                axis.axis("off")
                continue
            axis.imshow(image_np)
            axis.imshow(heatmap, cmap="magma", alpha=0.55, interpolation="bilinear")
            axis.set_title(_truncate(token_labels[token_index], 24))
            axis.axis("off")

        output_path = sample_dir / "decoder_visual_overlays.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _write_summary_markdown(
        self,
        sample: ExplainabilitySample,
        summary: Dict[str, Any],
        output_path: Path,
    ) -> None:
        lines = [
            f"# Explainability Summary: {sample.study_key}",
            "",
            "## Reports",
            f"- Reference: {sample.reference_report or 'N/A'}",
            f"- Generated: {sample.generated_report or 'N/A'}",
            "",
            "## Top Pathologies",
        ]

        if summary["top_pathologies"]:
            for item in summary["top_pathologies"]:
                lines.append(f"- {item['label']}: {item['score']:.4f}")
        else:
            lines.append("- No classification scores available.")

        lines.extend([
            "",
            "## Top KG Concepts",
        ])

        if summary["top_concepts"]:
            for item in summary["top_concepts"]:
                lines.append(f"- {item['label']}: {item['weight']:.4f}")
        else:
            lines.append("- No graph concepts available for this sample.")

        lines.extend([
            "",
            "## Fusion Activations",
        ])

        fusion_summary = summary.get("fusion_summary", {})
        if fusion_summary:
            lines.extend([
                f"- fusion layers: {fusion_summary.get('num_fusion_layers', 'N/A')}",
                f"- attention heads: {fusion_summary.get('num_attention_heads', 'N/A')}",
                f"- concepts/queries: {fusion_summary.get('num_queries', 'N/A')}",
                f"- visual patches: {fusion_summary.get('num_visual_patches', 'N/A')}",
                f"- mean layer-head attention: {float(fusion_summary.get('mean_layer_head_attention', 0.0)):.4f}",
                f"- max layer-head attention: {float(fusion_summary.get('max_layer_head_attention', 0.0)):.4f}",
            ])
        else:
            lines.append("- Fusion traces unavailable for this sample.")

        lines.extend([
            "",
            "## GATv2 Attention Vector",
            "- The learned attention vector a is exported from each GATv2 layer as the stored gat.att parameter.",
            "- Edge scores follow alpha_ij = softmax_j(LeakyReLU(((W h_i + W h_j + W_e e_ij) * a).sum(-1))).",
            "- The heatmap in gat_attention_vector.png shows which feature channels each head emphasises.",
            "",
            "## Saved Figures",
        ])

        for key, path in summary["files"].items():
            lines.append(f"- {key}: {Path(path).name}")

        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
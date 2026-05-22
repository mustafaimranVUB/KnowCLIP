"""Quantitative XAI metrics: Visual deletion/insertion curves, ablation ranking.

This module implements quantitative explainability metrics to validate that
the model's internal attention mechanisms (patch importance, visual attention)
actually correlate with model predictions and performance.

Simplified implementation focused on visual ablation (most practical metric).

Metrics:
- Visual Ablation Ranking: Drop top-k patches, measure AUC impact
- Patch Deletion Curves: Progressively remove patches by importance
- Patch Insertion Curves: Progressively add patches by importance (best-first)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_auc_score

logger = logging.getLogger(__name__)

# sklearn can emit this RuntimeWarning when internal casting encounters
# non-finite values in edge cases. We already sanitize inputs before AUC,
# so this warning is just noisy in quantitative-XAI batch logs.
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
    module=r"sklearn\.externals\.array_api_compat\.numpy\._aliases",
)


@dataclass
class QuantitativeXAIResults:
    """Results from quantitative XAI evaluation."""

    # Visual ablation metrics
    visual_ablation_auc_drop: float  # AUC drop after removing top-k patches
    visual_ablation_ranks: List[float]  # AUC drop for each k
    visual_deletion_curve_x: List[float]  # Fraction of patches removed
    visual_deletion_curve_y: List[float]  # Resulting AUC
    visual_deletion_auc: float  # Area under deletion curve
    visual_insertion_curve_x: List[float]  # Fraction of patches added (best-first)
    visual_insertion_curve_y: List[float]  # Resulting AUC
    visual_insertion_auc: float  # Area under insertion curve
    visual_preservation_rank: float  # Correlation: importance vs AUC drop


class QuantitativeXAIEvaluator:
    """Evaluates quantitative explainability of model predictions."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        num_visual_ablation_steps: int = 10,
    ):
        """Initialize quantitative XAI evaluator.

        Args:
            model: MedicalVLM model with explainability hooks.
            device: Computation device.
            num_visual_ablation_steps: Number of visual patch ablation steps.
        """
        self.model = model
        self.device = device
        self.num_visual_ablation_steps = num_visual_ablation_steps

    def evaluate(
        self,
        batch_images: torch.Tensor,
        batch_graphs: Optional[torch.Tensor],
        batch_graph_edge_index: Optional[torch.Tensor],
        batch_graph_edge_type: Optional[torch.Tensor],
        batch_graph_batch: Optional[torch.Tensor],
        batch_labels: torch.Tensor,
        label_index: int,
        explanations: Dict[str, torch.Tensor],
    ) -> QuantitativeXAIResults:
        """Compute quantitative XAI metrics for a batch.

        Args:
            batch_images: (B, 3, H, W) image batch.
            batch_graphs: (N_total, 768) node features or None.
            batch_graph_edge_index: (2, E_total) edge indices or None.
            batch_graph_edge_type: (E_total,) edge types or None.
            batch_graph_batch: (N_total,) batch vector for graph nodes or None.
            batch_labels: (B, 14) binary labels.
            label_index: Which label (0-13) to evaluate on.
            explanations: Dict with keys like 'pooling', 'fusion', 'decoder'.

        Returns:
            QuantitativeXAIResults with all metrics.
        """
        self.model.eval()
        batch_size = batch_images.shape[0]

        # Baseline metrics (all patches active)
        with torch.no_grad():
            logits_baseline = self._get_logits(
                batch_images, batch_graphs, batch_graph_edge_index, batch_graph_edge_type, batch_graph_batch
            )
            probs_baseline = torch.sigmoid(logits_baseline)[:, label_index]

            # For multi-label AUC (label_index), target is binary
            targets = batch_labels[:, label_index].float().cpu().numpy()
            baseline_auc = self._safe_binary_auc(
                targets,
                probs_baseline.detach().cpu().numpy(),
                fallback=0.5,
            )

        # Visual ablation (patches only)
        visual_results = self._visual_ablation_curves(
            batch_images,
            batch_graphs,
            batch_graph_edge_index,
            batch_graph_edge_type,
            batch_graph_batch,
            batch_labels,
            label_index,
            explanations,
            baseline_auc,
            targets,
        )

        return QuantitativeXAIResults(
            visual_ablation_auc_drop=visual_results["ablation_auc_drop"],
            visual_ablation_ranks=visual_results["ablation_ranks"],
            visual_deletion_curve_x=visual_results["deletion_x"],
            visual_deletion_curve_y=visual_results["deletion_y"],
            visual_deletion_auc=visual_results["deletion_auc"],
            visual_insertion_curve_x=visual_results["insertion_x"],
            visual_insertion_curve_y=visual_results["insertion_y"],
            visual_insertion_auc=visual_results["insertion_auc"],
            visual_preservation_rank=visual_results["preservation_rank"],
        )

    def _get_logits(
        self,
        batch_images: torch.Tensor,
        batch_graphs: Optional[torch.Tensor],
        batch_graph_edge_index: Optional[torch.Tensor],
        batch_graph_edge_type: Optional[torch.Tensor],
        batch_graph_batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Get classification logits."""
        with torch.no_grad():
            output = self.model(
                pixel_values=batch_images,
                graph_x=batch_graphs,
                graph_edge_index=batch_graph_edge_index,
                graph_edge_type=batch_graph_edge_type,
                graph_batch=batch_graph_batch,
            )
            return output.get("classification_logits", output)

    def _visual_ablation_curves(
        self,
        batch_images: torch.Tensor,
        batch_graphs: Optional[torch.Tensor],
        batch_graph_edge_index: Optional[torch.Tensor],
        batch_graph_edge_type: Optional[torch.Tensor],
        batch_graph_batch: Optional[torch.Tensor],
        batch_labels: torch.Tensor,
        label_index: int,
        explanations: Dict[str, torch.Tensor],
        baseline_auc: float,
        targets: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute visual (patch) ablation and deletion curves.

        Progressively remove image patches in order of importance.
        """
        batch_size, channels, height, width = batch_images.shape
        num_patches = 196  # ViT-B/16: (224/16)^2 = 14x14 patches

        # Get patch importance from visual pooling or fallback to uniform
        pooling_weights = explanations.get("pooling", {}).get("weights")
        if pooling_weights is not None and pooling_weights.ndim > 1 and pooling_weights.shape[1] >= num_patches:
            # Shape: (B, num_patches)
            patch_importance = pooling_weights[:, :num_patches]
        else:
            # Uniform importance fallback
            patch_importance = torch.ones(batch_size, num_patches) / num_patches

        # Normalize to [0, 1]
        if patch_importance.min() < 0 or patch_importance.max() > 1:
            patch_importance = (patch_importance - patch_importance.min()) / (
                patch_importance.max() - patch_importance.min() + 1e-8
            )

        # Patch ranking (descending by importance)
        patch_ranks = torch.argsort(patch_importance, dim=1, descending=True)  # (B, num_patches)

        ablation_aucs = []
        deletion_aucs = []
        insertion_aucs = []

        num_steps = self.num_visual_ablation_steps
        patch_size = 16  # 224 / 14 = 16 pixels per patch

        for step in range(num_steps + 1):
            k = int((step / num_steps) * num_patches)

            # Ablation: mask top-k patches by importance
            images_ablated = batch_images.clone()

            for b in range(batch_size):
                for patch_idx in range(min(k, num_patches)):
                    p = patch_ranks[b, patch_idx].item()
                    patch_row = int(p // 14)
                    patch_col = int(p % 14)
                    y_start = patch_row * patch_size
                    y_end = min(y_start + patch_size, height)
                    x_start = patch_col * patch_size
                    x_end = min(x_start + patch_size, width)
                    images_ablated[b, :, y_start:y_end, x_start:x_end] = 0.0

            with torch.no_grad():
                logits_ablated = self._get_logits(
                    images_ablated, batch_graphs, batch_graph_edge_index, batch_graph_edge_type, batch_graph_batch
                )
                probs_ablated = torch.sigmoid(logits_ablated)[:, label_index]

            auc_ablated = self._safe_binary_auc(
                targets,
                probs_ablated.detach().cpu().numpy(),
                fallback=baseline_auc,
            )

            ablation_aucs.append(baseline_auc - auc_ablated)

            # Deletion: remove top-k patches
            images_deleted = batch_images.clone()
            for b in range(batch_size):
                for patch_idx in range(min(k, num_patches)):
                    p = patch_ranks[b, patch_idx].item()
                    patch_row = int(p // 14)
                    patch_col = int(p % 14)
                    y_start = patch_row * patch_size
                    y_end = min(y_start + patch_size, height)
                    x_start = patch_col * patch_size
                    x_end = min(x_start + patch_size, width)
                    images_deleted[b, :, y_start:y_end, x_start:x_end] = 0.0

            with torch.no_grad():
                logits_deleted = self._get_logits(
                    images_deleted, batch_graphs, batch_graph_edge_index, batch_graph_edge_type, batch_graph_batch
                )
                probs_deleted = torch.sigmoid(logits_deleted)[:, label_index]

            auc_deleted = self._safe_binary_auc(
                targets,
                probs_deleted.detach().cpu().numpy(),
                fallback=0.5,
            )

            deletion_aucs.append(auc_deleted)

            # Insertion: add top-k patches only (best-first)
            images_inserted = torch.zeros_like(batch_images)
            for b in range(batch_size):
                for patch_idx in range(min(k, num_patches)):
                    p = patch_ranks[b, patch_idx].item()
                    patch_row = int(p // 14)
                    patch_col = int(p % 14)
                    y_start = patch_row * patch_size
                    y_end = min(y_start + patch_size, height)
                    x_start = patch_col * patch_size
                    x_end = min(x_start + patch_size, width)
                    images_inserted[b, :, y_start:y_end, x_start:x_end] = batch_images[
                        b, :, y_start:y_end, x_start:x_end
                    ]

            with torch.no_grad():
                logits_inserted = self._get_logits(
                    images_inserted, batch_graphs, batch_graph_edge_index, batch_graph_edge_type, batch_graph_batch
                )
                probs_inserted = torch.sigmoid(logits_inserted)[:, label_index]

            auc_inserted = self._safe_binary_auc(
                targets,
                probs_inserted.detach().cpu().numpy(),
                fallback=0.5,
            )

            insertion_aucs.append(auc_inserted)

        x_values = np.linspace(0, 1, num_steps + 1).tolist()
        try:
            deletion_auc_value = float(auc(x_values, deletion_aucs))
            insertion_auc_value = float(auc(x_values, insertion_aucs))
        except (ValueError, ZeroDivisionError):
            deletion_auc_value = 0.5
            insertion_auc_value = 0.5

        # Preservation rank: correlation between ablation importance and AUC drop
        if len(ablation_aucs) > 1 and np.std(ablation_aucs) > 1e-6:
            try:
                preservation_rank = float(np.corrcoef(np.arange(len(ablation_aucs)), ablation_aucs)[0, 1])
            except (ValueError, RuntimeWarning):
                preservation_rank = 0.5
        else:
            preservation_rank = 0.5

        if np.isnan(preservation_rank):
            preservation_rank = 0.5

        return {
            "ablation_auc_drop": float(np.mean(ablation_aucs)) if ablation_aucs else 0.0,
            "ablation_ranks": [float(x) for x in ablation_aucs],
            "deletion_x": x_values,
            "deletion_y": [float(x) for x in deletion_aucs],
            "deletion_auc": deletion_auc_value,
            "insertion_x": x_values,
            "insertion_y": [float(x) for x in insertion_aucs],
            "insertion_auc": insertion_auc_value,
            "preservation_rank": preservation_rank,
        }

    @staticmethod
    def _safe_binary_auc(
        targets: np.ndarray,
        scores: np.ndarray,
        fallback: float = 0.5,
    ) -> float:
        """Compute ROC-AUC for a binary target with robust NaN handling.

        Quantitative-XAI batches can contain uncertain labels encoded as NaN.
        sklearn's roc_auc_score emits RuntimeWarnings (and can fail) when NaNs
        leak into input arrays, so we filter to finite pairs first.
        """
        y_true = np.asarray(targets, dtype=np.float32).reshape(-1)
        y_score = np.asarray(scores, dtype=np.float32).reshape(-1)
        if y_true.size == 0 or y_score.size == 0:
            return float(fallback)

        valid = np.isfinite(y_true) & np.isfinite(y_score)
        if not np.any(valid):
            return float(fallback)

        y_true = y_true[valid]
        y_score = y_score[valid]

        # ROC-AUC is undefined for single-class targets.
        if y_true.size < 2 or np.unique(y_true).size < 2:
            return float(fallback)

        try:
            auc_value = float(roc_auc_score(y_true, y_score))
        except Exception:
            return float(fallback)

        if not np.isfinite(auc_value):
            return float(fallback)
        return auc_value

"""Quantitative XAI evaluation pipeline — measure explainability metrics.

This pipeline evaluates whether the model's internal attention mechanisms
(visual importance, cross-attention) actually correlate with predictions
using deletion/insertion curves and ablation ranking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.core.config import ProjectConfig
from src.evaluation.quantitative_xai_metrics import (
    QuantitativeXAIEvaluator,
    QuantitativeXAIResults,
)
from src.pipelines.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


class QuantitativeXAIPipeline:
    """Evaluate quantitative explainability metrics on test set."""

    def __init__(
        self,
        config: ProjectConfig,
        device: Optional[str] = None,
    ) -> None:
        self.config = config
        self.training_pipeline = TrainingPipeline(config=config, device=device)
        self.device = self.training_pipeline.device

    def run(
        self,
        checkpoint_path: str | Path,
        output_dir: Optional[str | Path] = None,
        max_samples: Optional[int] = None,
        label_indices: Optional[list[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate quantitative XAI metrics.

        Args:
            checkpoint_path: Path to trained model checkpoint.
            output_dir: Where to save results (JSON + plots).
            max_samples: Max test samples to evaluate (for speed).
            label_indices: Which labels to evaluate (default: all 14).

        Returns:
            Dictionary with results.
        """
        self.training_pipeline.setup()
        model = self.training_pipeline.build_model()
        self.training_pipeline.load_checkpoint(checkpoint_path)
        model = model.to(self.device)
        model.eval()

        _, _, test_loader = self.training_pipeline.build_dataloaders()

        if label_indices is None:
            label_indices = list(range(14))

        output_path = Path(output_dir) if output_dir else Path(self.config.training.log_dir).parent / "quantitative_xai"
        output_path.mkdir(parents=True, exist_ok=True)

        evaluator = QuantitativeXAIEvaluator(
            model=model,
            device=self.device,
            num_visual_ablation_steps=10,
        )

        all_results: Dict[int, list[QuantitativeXAIResults]] = {
            label_idx: [] for label_idx in label_indices
        }

        sample_count = 0
        logger.info(f"Running quantitative XAI evaluation on test set...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Quantitative XAI")):
                if max_samples and sample_count >= max_samples:
                    break

                images = batch.get("image")
                if images is None:
                    images = batch.get("images")
                if images is None:
                    images = batch.get("pixel_values")
                if images is None:
                    logger.warning(
                        "Skipping batch %d: no image tensor found (expected one of: image/images/pixel_values).",
                        batch_idx,
                    )
                    continue
                images = images.to(self.device)

                labels = batch.get("labels")
                if labels is None:
                    logger.warning("Skipping batch %d: labels tensor missing.", batch_idx)
                    continue
                labels = labels.to(self.device)

                # Optional KG data
                graphs = batch.get("graph_x")
                if graphs is None:
                    graphs = batch.get("graphs")
                if graphs is not None:
                    graphs = graphs.to(self.device)

                graph_edge_index = batch.get("graph_edge_index")
                if graph_edge_index is not None:
                    graph_edge_index = graph_edge_index.to(self.device)

                graph_edge_type = batch.get("graph_edge_type")
                if graph_edge_type is not None:
                    graph_edge_type = graph_edge_type.to(self.device)

                graph_batch = batch.get("graph_batch")
                if graph_batch is not None:
                    graph_batch = graph_batch.to(self.device)

                # Forward pass with attention capture
                output = model(
                    pixel_values=images,
                    graph_x=graphs,
                    graph_edge_index=graph_edge_index,
                    graph_edge_type=graph_edge_type,
                    graph_batch=graph_batch,
                    return_attention=True,
                )

                explanations = output.get("explainability", {})

                # Evaluate each label
                for label_idx in label_indices:
                    try:
                        result = evaluator.evaluate(
                            batch_images=images,
                            batch_graphs=graphs,
                            batch_graph_edge_index=graph_edge_index,
                            batch_graph_edge_type=graph_edge_type,
                            batch_graph_batch=graph_batch,
                            batch_labels=labels,
                            label_index=label_idx,
                            explanations=explanations,
                        )
                        all_results[label_idx].append(result)
                    except Exception as e:
                        logger.warning(
                            f"Failed to evaluate label {label_idx} on batch {batch_idx}: {e}"
                        )

                sample_count += images.shape[0]

        # Aggregate results
        aggregated = self._aggregate_results(all_results)

        # Save outputs
        self._save_results(aggregated, output_path)
        self._plot_results(aggregated, output_path)

        logger.info(f"Quantitative XAI evaluation complete. Results saved to {output_path}")
        return aggregated

    def _aggregate_results(
        self, all_results: Dict[int, list[QuantitativeXAIResults]]
    ) -> Dict[str, Any]:
        """Aggregate results across samples and labels."""
        class_names = self.config.model.classification_head.class_names

        aggregated = {
            "class_names": class_names,
            "by_class": {},
            "overall": {},
        }

        for label_idx, results_list in all_results.items():
            if not results_list:
                continue

            class_name = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"

            # Average across samples
            visual_ablation_drops = [r.visual_ablation_auc_drop for r in results_list]
            visual_deletion_aucs = [r.visual_deletion_auc for r in results_list]
            visual_insertion_aucs = [r.visual_insertion_auc for r in results_list]
            visual_preservation_ranks = [r.visual_preservation_rank for r in results_list]

            aggregated["by_class"][class_name] = {
                "num_samples": len(results_list),
                "visual_ablation_auc_drop": {
                    "mean": float(np.mean(visual_ablation_drops)),
                    "std": float(np.std(visual_ablation_drops)),
                    "min": float(np.min(visual_ablation_drops)),
                    "max": float(np.max(visual_ablation_drops)),
                },
                "visual_deletion_auc": {
                    "mean": float(np.mean(visual_deletion_aucs)),
                    "std": float(np.std(visual_deletion_aucs)),
                },
                "visual_insertion_auc": {
                    "mean": float(np.mean(visual_insertion_aucs)),
                    "std": float(np.std(visual_insertion_aucs)),
                },
                "visual_preservation_rank": {
                    "mean": float(np.mean(visual_preservation_ranks)),
                    "std": float(np.std(visual_preservation_ranks)),
                },
            }

        # Overall statistics
        all_visual_ablation_drops = []
        all_visual_deletion_aucs = []
        all_visual_insertion_aucs = []

        for class_stats in aggregated["by_class"].values():
            all_visual_ablation_drops.append(class_stats["visual_ablation_auc_drop"]["mean"])
            all_visual_deletion_aucs.append(class_stats["visual_deletion_auc"]["mean"])
            all_visual_insertion_aucs.append(class_stats["visual_insertion_auc"]["mean"])

        if all_visual_ablation_drops:
            aggregated["overall"] = {
                "visual_ablation_auc_drop_mean": float(np.mean(all_visual_ablation_drops)),
                "visual_deletion_auc_mean": float(np.mean(all_visual_deletion_aucs)),
                "visual_insertion_auc_mean": float(np.mean(all_visual_insertion_aucs)),
            }

        return aggregated

    def _save_results(self, aggregated: Dict[str, Any], output_path: Path) -> None:
        """Save results as JSON."""
        output_file = output_path / "quantitative_xai_metrics.json"
        with open(output_file, "w") as f:
            json.dump(aggregated, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    def _plot_results(self, aggregated: Dict[str, Any], output_path: Path) -> None:
        """Create visualization plots."""
        class_names = aggregated.get("class_names", [])
        by_class = aggregated.get("by_class", {})

        if not by_class:
            return

        # Plot 1: Ablation AUC drop by class
        fig, ax = plt.subplots(figsize=(12, 6))
        classes = list(by_class.keys())
        ablation_means = [by_class[c]["visual_ablation_auc_drop"]["mean"] for c in classes]
        ablation_stds = [by_class[c]["visual_ablation_auc_drop"]["std"] for c in classes]

        ax.bar(classes, ablation_means, yerr=ablation_stds, capsize=5, alpha=0.7)
        ax.set_ylabel("AUC Drop (higher = more important)")
        ax.set_title("Visual Ablation: AUC Drop by Class")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / "ablation_impact.png", dpi=100)
        plt.close()

        logger.info(f"Saved ablation impact plot to {output_path / 'ablation_impact.png'}")

        # Plot 2: Deletion and Insertion curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        deletion_means = [by_class[c]["visual_deletion_auc"]["mean"] for c in classes]
        insertion_means = [by_class[c]["visual_insertion_auc"]["mean"] for c in classes]

        ax1.bar(classes, deletion_means, alpha=0.7, label="Deletion")
        ax1.set_ylabel("AUC")
        ax1.set_title("Visual Deletion Curve (AUC)")
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(classes, insertion_means, alpha=0.7, label="Insertion", color="orange")
        ax2.set_ylabel("AUC")
        ax2.set_title("Visual Insertion Curve (AUC)")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / "deletion_insertion_curves.png", dpi=100)
        plt.close()

        logger.info(f"Saved deletion/insertion plot to {output_path / 'deletion_insertion_curves.png'}")

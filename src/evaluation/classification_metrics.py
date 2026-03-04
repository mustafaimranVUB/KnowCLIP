"""Classification evaluation metrics for CheXpert-14 multi-label detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Compute per-class and macro classification metrics.

    Parameters:
        class_names: Names of the 14 CheXpert labels.
        threshold: Decision threshold for converting probabilities.
    """

    CHEXPERT_LABELS: List[str] = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion",
        "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
    ]

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        self.class_names = class_names or self.CHEXPERT_LABELS
        self.threshold = threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute all classification metrics.

        Args:
            y_true: ``(N, C)`` ground-truth binary labels.
            y_score: ``(N, C)`` predicted probabilities (post-sigmoid).

        Returns:
            Dict with per-class and macro metrics.
        """
        from sklearn.metrics import (  # type: ignore
            roc_auc_score,
            f1_score,
            precision_score,
            recall_score,
            average_precision_score,
        )

        results: Dict[str, Any] = {"per_class": {}, "macro": {}}

        y_pred = (y_score >= self.threshold).astype(int)
        num_classes = y_true.shape[1]

        # Per-class metrics
        auc_scores = []
        f1_scores = []
        prec_scores = []
        rec_scores = []
        ap_scores = []

        for c in range(num_classes):
            name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            class_results: Dict[str, float] = {}

            # Skip classes with all-same labels
            if len(np.unique(y_true[:, c])) < 2:
                logger.warning("Class '%s' has only one label value — skipping AUC", name)
                class_results["auc_roc"] = float("nan")
            else:
                auc = float(roc_auc_score(y_true[:, c], y_score[:, c]))
                class_results["auc_roc"] = auc
                auc_scores.append(auc)

            f1 = float(f1_score(y_true[:, c], y_pred[:, c], zero_division=0))
            prec = float(precision_score(y_true[:, c], y_pred[:, c], zero_division=0))
            rec = float(recall_score(y_true[:, c], y_pred[:, c], zero_division=0))

            class_results.update({"f1": f1, "precision": prec, "recall": rec})
            f1_scores.append(f1)
            prec_scores.append(prec)
            rec_scores.append(rec)

            try:
                ap = float(average_precision_score(y_true[:, c], y_score[:, c]))
                class_results["average_precision"] = ap
                ap_scores.append(ap)
            except ValueError:
                class_results["average_precision"] = float("nan")

            results["per_class"][name] = class_results

        # Macro averages
        results["macro"] = {
            "auc_roc": float(np.nanmean(auc_scores)) if auc_scores else float("nan"),
            "f1": float(np.mean(f1_scores)),
            "precision": float(np.mean(prec_scores)),
            "recall": float(np.mean(rec_scores)),
            "average_precision": float(np.nanmean(ap_scores)) if ap_scores else float("nan"),
        }

        return results

    def find_optimal_thresholds(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> Dict[str, float]:
        """Find per-class optimal thresholds maximising F1.

        Args:
            y_true: ``(N, C)`` ground-truth binary labels.
            y_score: ``(N, C)`` predicted probabilities.

        Returns:
            Dict mapping class name to optimal threshold.
        """
        thresholds: Dict[str, float] = {}
        for c in range(y_true.shape[1]):
            name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"

            best_f1 = 0.0
            best_t = 0.5
            for t in np.arange(0.1, 0.9, 0.05):
                preds = (y_score[:, c] >= t).astype(int)
                from sklearn.metrics import f1_score as _f1  # type: ignore

                f1 = _f1(y_true[:, c], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            thresholds[name] = round(best_t, 2)

        return thresholds

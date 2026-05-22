"""Classification evaluation metrics for CheXpert-14 multi-label detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score as _sklearn_f1  # type: ignore

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
        thresholds: Optional[Dict[str, float] | np.ndarray] = None,
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

        results: Dict[str, Any] = {"per_class": {}, "macro": {}, "micro": {}, "overall": {}}

        threshold_array = self._resolve_thresholds(y_score.shape[1], thresholds)
        y_pred = (y_score >= threshold_array.reshape(1, -1)).astype(int)
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

            # Keep only binary labels for metrics. CheXpert uncertain labels (-1)
            # and missing labels (NaN) are excluded from evaluation.
            y_true_col = y_true[:, c]
            valid_mask = (~np.isnan(y_true_col)) & np.isin(y_true_col, [0.0, 1.0])
            if not np.any(valid_mask):
                logger.warning(
                    "Class '%s' has no valid binary labels after masking — skipping",
                    name,
                )
                class_results.update(
                    {
                        "auc_roc": float("nan"),
                        "f1": float("nan"),
                        "precision": float("nan"),
                        "recall": float("nan"),
                        "average_precision": float("nan"),
                    }
                )
                results["per_class"][name] = class_results
                continue

            y_true_c = y_true[valid_mask, c].astype(int)
            y_score_c = y_score[valid_mask, c]
            y_pred_c = y_pred[valid_mask, c]

            # Skip classes with all-same labels
            if len(np.unique(y_true_c)) < 2:
                logger.warning(
                    "Class '%s' has only one label value (%d samples, all=%d) — "
                    "skipping AUC and AP (insufficient label diversity in test set)",
                    name,
                    len(y_true_c),
                    int(y_true_c[0]),
                )
                class_results["auc_roc"] = float("nan")
                class_results["average_precision"] = float("nan")
            else:
                auc = float(roc_auc_score(y_true_c, y_score_c))
                class_results["auc_roc"] = auc
                auc_scores.append(auc)

            f1 = float(f1_score(y_true_c, y_pred_c, zero_division=0))
            prec = float(precision_score(y_true_c, y_pred_c, zero_division=0))
            rec = float(recall_score(y_true_c, y_pred_c, zero_division=0))

            class_results.update({"f1": f1, "precision": prec, "recall": rec})
            f1_scores.append(f1)
            prec_scores.append(prec)
            rec_scores.append(rec)

            if "average_precision" not in class_results:
                try:
                    ap = float(average_precision_score(y_true_c, y_score_c))
                    class_results["average_precision"] = ap
                    ap_scores.append(ap)
                except ValueError:
                    class_results["average_precision"] = float("nan")

            results["per_class"][name] = class_results

        # Log summary of evaluable vs skipped classes
        evaluable_auc = len(auc_scores)
        skipped_auc = num_classes - evaluable_auc
        if skipped_auc > 0:
            logger.warning(
                "AUC/AP computed for %d/%d classes; %d skipped due to single-label test samples. "
                "Consider increasing valid binary label coverage (or adjusting missing/uncertain-label handling) "
                "for more reliable evaluation.",
                evaluable_auc,
                num_classes,
                skipped_auc,
            )

        # Macro averages
        results["macro"] = {
            "auc_roc": float(np.nanmean(auc_scores)) if auc_scores else float("nan"),
            "f1": float(np.mean(f1_scores)),
            "precision": float(np.mean(prec_scores)),
            "recall": float(np.mean(rec_scores)),
            "average_precision": float(np.nanmean(ap_scores)) if ap_scores else float("nan"),
        }

        valid_mask = (~np.isnan(y_true)) & np.isin(y_true, [0.0, 1.0])
        if np.any(valid_mask):
            from sklearn.metrics import (  # type: ignore
                average_precision_score,
                f1_score,
                hamming_loss,
                jaccard_score,
                matthews_corrcoef,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            flat_true = y_true[valid_mask].astype(int)
            flat_score = y_score[valid_mask]
            flat_pred = y_pred[valid_mask]

            if len(np.unique(flat_true)) >= 2:
                results["micro"]["auc_roc"] = float(roc_auc_score(flat_true, flat_score))
                results["micro"]["average_precision"] = float(average_precision_score(flat_true, flat_score))
            else:
                results["micro"]["auc_roc"] = float("nan")
                results["micro"]["average_precision"] = float("nan")

            results["micro"].update({
                "f1": float(f1_score(flat_true, flat_pred, zero_division=0)),
                "precision": float(precision_score(flat_true, flat_pred, zero_division=0)),
                "recall": float(recall_score(flat_true, flat_pred, zero_division=0)),
            })

            subset_matches = []
            for sample_idx in range(y_true.shape[0]):
                sample_mask = valid_mask[sample_idx]
                if not np.any(sample_mask):
                    continue
                subset_matches.append(float(np.all(y_pred[sample_idx, sample_mask] == y_true[sample_idx, sample_mask])))

            results["overall"] = {
                "subset_accuracy": float(np.mean(subset_matches)) if subset_matches else float("nan"),
                "hamming_loss": float(hamming_loss(flat_true, flat_pred)),
                "jaccard_micro": float(jaccard_score(flat_true, flat_pred, zero_division=0)),
                "matthews_corrcoef": float(matthews_corrcoef(flat_true, flat_pred)) if len(np.unique(flat_pred)) > 1 or len(np.unique(flat_true)) > 1 else 0.0,
                "brier_score": float(np.mean((flat_score - flat_true) ** 2)),
                "ece": float(self._expected_calibration_error(flat_true, flat_score)),
            }
        else:
            results["micro"] = {
                "auc_roc": float("nan"),
                "average_precision": float("nan"),
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
            }
            results["overall"] = {
                "subset_accuracy": float("nan"),
                "hamming_loss": float("nan"),
                "jaccard_micro": float("nan"),
                "matthews_corrcoef": float("nan"),
                "brier_score": float("nan"),
                "ece": float("nan"),
            }

        results["thresholds"] = {
            self.class_names[index] if index < len(self.class_names) else f"class_{index}": float(threshold_array[index])
            for index in range(num_classes)
        }

        return results

    def _resolve_thresholds(
        self,
        num_classes: int,
        thresholds: Optional[Dict[str, float] | np.ndarray],
    ) -> np.ndarray:
        if thresholds is None:
            return np.full((num_classes,), float(self.threshold), dtype=np.float32)
        if isinstance(thresholds, dict):
            return np.asarray([
                float(thresholds.get(self.class_names[index], self.threshold))
                for index in range(num_classes)
            ], dtype=np.float32)
        arr = np.asarray(thresholds, dtype=np.float32).reshape(-1)
        if arr.shape[0] != num_classes:
            raise ValueError(f"Expected {num_classes} thresholds, got {arr.shape[0]}")
        return arr

    @staticmethod
    def _expected_calibration_error(
        y_true: np.ndarray,
        y_score: np.ndarray,
        num_bins: int = 15,
    ) -> float:
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        total = max(len(y_true), 1)
        for left, right in zip(bins[:-1], bins[1:]):
            if right == 1.0:
                mask = (y_score >= left) & (y_score <= right)
            else:
                mask = (y_score >= left) & (y_score < right)
            if not np.any(mask):
                continue
            confidence = float(np.mean(y_score[mask]))
            accuracy = float(np.mean(y_true[mask]))
            ece += abs(confidence - accuracy) * (np.sum(mask) / total)
        return float(ece)

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

            y_true_col = y_true[:, c]
            valid_mask = (~np.isnan(y_true_col)) & np.isin(y_true_col, [0.0, 1.0])
            if not np.any(valid_mask):
                thresholds[name] = 0.5
                continue

            y_true_c = y_true[valid_mask, c].astype(int)
            y_score_c = y_score[valid_mask, c]

            best_f1 = 0.0
            best_t = 0.5
            for t in np.arange(0.1, 0.9, 0.05):
                preds = (y_score_c >= t).astype(int)
                f1 = _sklearn_f1(y_true_c, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            thresholds[name] = round(best_t, 2)

        return thresholds

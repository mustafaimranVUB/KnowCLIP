"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.artifact_export import save_evaluation_artifacts
from src.evaluation.classification_metrics import ClassificationEvaluator
from src.evaluation.statistical_tests import (
    bonferroni_correction,
    bootstrap_ci,
    mcnemar_test,
)


class TestClassificationEvaluator:
    def test_evaluate_returns_per_class(self):
        evaluator = ClassificationEvaluator()
        y_true = np.random.randint(0, 2, (100, 14))
        y_score = np.random.rand(100, 14)

        result = evaluator.evaluate(y_true=y_true, y_score=y_score)
        assert "per_class" in result
        assert "macro" in result
        assert len(result["per_class"]) == 14

    def test_macro_auc_range(self):
        evaluator = ClassificationEvaluator()
        y_true = np.random.randint(0, 2, (100, 14))
        y_score = np.random.rand(100, 14)

        result = evaluator.evaluate(y_true=y_true, y_score=y_score)
        auc = result["macro"]["auc_roc"]
        # AUC might be NaN for some random splits, but if computed it should be 0-1
        if not np.isnan(auc):
            assert 0.0 <= auc <= 1.0

    def test_perfect_prediction(self):
        evaluator = ClassificationEvaluator(threshold=0.5)
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        # Perfect scores
        y_score = y_true.astype(float)

        result = evaluator.evaluate(y_true=y_true, y_score=y_score)
        assert result["macro"]["f1"] == 1.0

    def test_find_optimal_thresholds(self):
        evaluator = ClassificationEvaluator()
        y_true = np.random.randint(0, 2, (100, 3))
        y_score = np.random.rand(100, 3)

        thresholds = evaluator.find_optimal_thresholds(y_true, y_score)
        assert len(thresholds) == 3
        for name, t in thresholds.items():
            assert 0.1 <= t <= 0.85

    def test_evaluate_with_nan_labels(self):
        evaluator = ClassificationEvaluator(class_names=["c0", "c1", "c2"])
        y_true = np.array(
            [
                [1.0, 0.0, np.nan],
                [0.0, 1.0, np.nan],
                [1.0, np.nan, np.nan],
                [0.0, 0.0, np.nan],
            ],
            dtype=float,
        )
        y_score = np.array(
            [
                [0.9, 0.2, 0.1],
                [0.1, 0.8, 0.3],
                [0.7, 0.4, 0.6],
                [0.2, 0.3, 0.7],
            ],
            dtype=float,
        )

        result = evaluator.evaluate(y_true=y_true, y_score=y_score)
        assert "per_class" in result and "macro" in result
        # Entirely-NaN class should not crash and should remain NaN.
        assert np.isnan(result["per_class"]["c2"]["auc_roc"])

    def test_evaluate_with_uncertain_minus_one_labels(self):
        evaluator = ClassificationEvaluator(class_names=["c0", "c1"])
        y_true = np.array(
            [
                [1.0, -1.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [1.0, -1.0],
            ],
            dtype=float,
        )
        y_score = np.array(
            [
                [0.9, 0.8],
                [0.2, 0.7],
                [0.4, 0.3],
                [0.8, 0.6],
            ],
            dtype=float,
        )

        result = evaluator.evaluate(y_true=y_true, y_score=y_score)
        assert "per_class" in result and "macro" in result
        # c1 has both 0 and 1 after excluding uncertain (-1), so AUC should be valid.
        assert not np.isnan(result["per_class"]["c1"]["auc_roc"])


class TestMcNemarTest:
    def test_same_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        preds_a = np.array([1, 0, 1, 0, 1])
        preds_b = np.array([1, 0, 1, 0, 1])

        result = mcnemar_test(y_true, preds_a, preds_b)
        # Same predictions → no discordance
        assert result["p_value"] >= 0

    def test_different_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
        preds_a = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)  # perfect
        preds_b = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5)  # inverted

        result = mcnemar_test(y_true, preds_a, preds_b)
        assert "statistic" in result
        assert "p_value" in result


class TestBootstrapCI:
    def test_returns_ci(self):
        data = np.random.rand(50, 14)
        labels = np.random.randint(0, 2, (50, 14))

        def metric_fn(y_true, y_score):
            return float(np.mean(y_score))

        result = bootstrap_ci(metric_fn, labels, data, n_resamples=100)
        assert "point_estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]


class TestBonferroniCorrection:
    def test_correction(self):
        p_values = [0.01, 0.02, 0.05, 0.10]
        corrected = bonferroni_correction(p_values)
        assert len(corrected) == 4
        # Each result is a dict with original_p, corrected_p, significant
        assert corrected[0]["corrected_p"] == pytest.approx(0.04, abs=1e-6)
        assert all(r["corrected_p"] <= 1.0 for r in corrected)
        # First should be significant (0.01 * 4 = 0.04 < 0.05)
        assert corrected[0]["significant"] is True
        # Second should NOT be significant (0.02 * 4 = 0.08 > 0.05)
        assert corrected[1]["significant"] is False


class TestEvaluationArtifactExport:
    def test_save_evaluation_artifacts_creates_outputs(self, tmp_path):
        results = {
            "classification": {
                "macro": {"auc_roc": 0.81, "f1": 0.73},
                "micro": {"auc_roc": 0.84, "f1": 0.76},
                "per_class": {
                    "atelectasis": {"auc_roc": 0.82, "f1": 0.71, "average_precision": 0.77},
                    "cardiomegaly": {"auc_roc": 0.88, "f1": 0.79, "average_precision": 0.83},
                },
            },
            "generation": {
                "bleu_1": 0.22,
                "rouge_l": 0.31,
                "bertscore_f1": 0.85,
                "f1_radgraph": 0.19,
            },
        }

        artifacts = save_evaluation_artifacts(results, tmp_path)

        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "summary.md").exists()
        assert (tmp_path / "classification_metrics.png").exists()
        assert (tmp_path / "generation_metrics.png").exists()
        assert artifacts["metrics_json"].endswith("metrics.json")

    def test_save_evaluation_artifacts_exports_prediction_rows(self, tmp_path):
        results = {
            "classification": {
                "macro": {"auc_roc": 0.81, "f1": 0.73},
                "micro": {"auc_roc": 0.84, "f1": 0.76},
                "per_class": {
                    "atelectasis": {"auc_roc": 0.82, "f1": 0.71, "average_precision": 0.77},
                },
            },
            "generation": {"bleu_1": 0.22, "rouge_l": 0.31},
        }
        classification_rows = [
            {
                "study_key": "s1",
                "true_atelectasis": 1.0,
                "score_atelectasis": 0.9,
                "pred_atelectasis": 1,
            },
            {
                "study_key": "s2",
                "true_atelectasis": 0.0,
                "score_atelectasis": 0.2,
                "pred_atelectasis": 0,
            },
        ]
        generation_rows = [
            {"study_key": "s1", "reference_report": "normal heart", "generated_report": "normal heart"},
            {"study_key": "s2", "reference_report": "no effusion", "generated_report": "no pleural effusion"},
        ]

        artifacts = save_evaluation_artifacts(
            results,
            tmp_path,
            classification_rows=classification_rows,
            generation_rows=generation_rows,
        )

        assert (tmp_path / "classification_predictions.csv").exists()
        assert (tmp_path / "classification_reliability.png").exists()
        assert (tmp_path / "generation_predictions.jsonl").exists()
        assert artifacts["classification_predictions"].endswith("classification_predictions.csv")
        assert artifacts["generation_predictions"].endswith("generation_predictions.jsonl")

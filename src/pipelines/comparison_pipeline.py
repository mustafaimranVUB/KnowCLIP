"""Compare two evaluation artifact directories for ablation reporting."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from src.evaluation.statistical_tests import bootstrap_ci, bonferroni_correction, mcnemar_test, wilcoxon_signed_rank


class ComparisonPipeline:
    """Run pairwise significance and delta reports over two evaluation outputs."""

    def run(
        self,
        eval_dir_a: str | Path,
        eval_dir_b: str | Path,
        *,
        label_a: str = "model_a",
        label_b: str = "model_b",
        output_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        dir_a = Path(eval_dir_a)
        dir_b = Path(eval_dir_b)
        metrics_a = self._load_json(dir_a / "metrics.json")
        metrics_b = self._load_json(dir_b / "metrics.json")

        results: Dict[str, Any] = {
            "models": {"a": label_a, "b": label_b},
            "metrics": {label_a: metrics_a, label_b: metrics_b},
        }

        cls_path_a = dir_a / "classification_predictions.csv"
        cls_path_b = dir_b / "classification_predictions.csv"
        if cls_path_a.exists() and cls_path_b.exists():
            results["classification"] = self._compare_classification(cls_path_a, cls_path_b)

        gen_path_a = dir_a / "generation_predictions.jsonl"
        gen_path_b = dir_b / "generation_predictions.jsonl"
        if gen_path_a.exists() and gen_path_b.exists():
            results["generation"] = self._compare_generation(gen_path_a, gen_path_b)

        output_path = Path(output_dir) if output_dir is not None else dir_a / f"compare_vs_{dir_b.name}"
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        self._write_markdown(results, label_a, label_b, output_path / "comparison.md")
        return results

    def _compare_classification(self, path_a: Path, path_b: Path) -> Dict[str, Any]:
        rows_a = self._load_csv(path_a)
        rows_b = self._load_csv(path_b)
        aligned = self._align_rows(rows_a, rows_b)
        if not aligned:
            return {}

        first_row = aligned[0][0]
        label_keys = sorted(key.removeprefix("true_") for key in first_row if key.startswith("true_"))
        per_class: Dict[str, Any] = {}
        p_values: List[float] = []
        flat_true = []
        flat_pred_a = []
        flat_pred_b = []

        for label_key in label_keys:
            y_true = []
            pred_a = []
            pred_b = []
            for row_a, row_b in aligned:
                true_value = row_a.get(f"true_{label_key}", "")
                if true_value in ("", None):
                    continue
                y_true.append(int(float(true_value)))
                pred_a.append(int(row_a[f"pred_{label_key}"]))
                pred_b.append(int(row_b[f"pred_{label_key}"]))
            if not y_true:
                continue
            test_result = mcnemar_test(np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b))
            per_class[label_key] = test_result
            p_values.append(test_result["p_value"])
            flat_true.extend(y_true)
            flat_pred_a.extend(pred_a)
            flat_pred_b.extend(pred_b)

        flat_true_arr = np.asarray(flat_true)
        flat_pred_a_arr = np.asarray(flat_pred_a)
        flat_pred_b_arr = np.asarray(flat_pred_b)

        def _f1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            tp = float(np.sum((y_true == 1) & (y_pred == 1)))
            fp = float(np.sum((y_true == 0) & (y_pred == 1)))
            fn = float(np.sum((y_true == 1) & (y_pred == 0)))
            denom = (2 * tp + fp + fn)
            return 0.0 if denom == 0 else (2 * tp) / denom

        overall = mcnemar_test(flat_true_arr, flat_pred_a_arr, flat_pred_b_arr) if flat_true else {}
        model_a_ci = bootstrap_ci(_f1_metric, flat_true_arr, flat_pred_a_arr, n_resamples=300) if flat_true else {}
        model_b_ci = bootstrap_ci(_f1_metric, flat_true_arr, flat_pred_b_arr, n_resamples=300) if flat_true else {}

        return {
            "overall_mcnemar": overall,
            "per_class_mcnemar": per_class,
            "per_class_bonferroni": bonferroni_correction(p_values) if p_values else [],
            "micro_f1_ci": {"model_a": model_a_ci, "model_b": model_b_ci},
        }

    def _compare_generation(self, path_a: Path, path_b: Path) -> Dict[str, Any]:
        rows_a = self._load_jsonl(path_a)
        rows_b = self._load_jsonl(path_b)
        aligned = self._align_rows(rows_a, rows_b)
        if not aligned:
            return {}

        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores_a = []
        scores_b = []
        for row_a, row_b in aligned:
            reference = row_a.get("reference_report", "")
            hyp_a = row_a.get("generated_report", "")
            hyp_b = row_b.get("generated_report", "")
            scores_a.append(scorer.score(reference, hyp_a)["rougeL"].fmeasure)
            scores_b.append(scorer.score(reference, hyp_b)["rougeL"].fmeasure)

        return {
            "rouge_l_mean": {"model_a": float(np.mean(scores_a)), "model_b": float(np.mean(scores_b))},
            "wilcoxon_rouge_l": wilcoxon_signed_rank(np.asarray(scores_a), np.asarray(scores_b)),
            "n_pairs": len(scores_a),
        }

    def _write_markdown(self, results: Dict[str, Any], label_a: str, label_b: str, output_path: Path) -> None:
        lines = [
            "# Evaluation Comparison",
            "",
            f"- model_a: {label_a}",
            f"- model_b: {label_b}",
            "",
        ]
        classification = results.get("classification", {})
        if classification:
            lines.extend([
                "## Classification",
                f"- overall McNemar p-value: {classification.get('overall_mcnemar', {}).get('p_value', float('nan'))}",
                "",
            ])
        generation = results.get("generation", {})
        if generation:
            lines.extend([
                "## Generation",
                f"- Rouge-L Wilcoxon p-value: {generation.get('wilcoxon_rouge_l', {}).get('p_value', float('nan'))}",
                "",
            ])
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            logger.warning(
                "metrics.json not found at %s — run 'python -m main evaluate' for this model first "
                "before using compare.",
                path,
            )
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_csv(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _align_rows(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        lookup_b = {str(row.get("study_key")): row for row in rows_b}
        return [(row_a, lookup_b[str(row_a.get("study_key"))]) for row_a in rows_a if str(row_a.get("study_key")) in lookup_b]
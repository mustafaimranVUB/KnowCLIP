"""Helpers for exporting evaluation metrics as JSON and plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_summary(results: Dict[str, Any], summary_path: Path) -> None:
    lines = ["# Evaluation Summary", ""]

    classification = results.get("classification")
    if isinstance(classification, dict):
        macro = classification.get("macro", {})
        micro = classification.get("micro", {})
        lines.extend([
            "## Classification",
            f"- Macro AUC-ROC: {float(macro.get('auc_roc', float('nan'))):.4f}",
            f"- Macro F1: {float(macro.get('f1', float('nan'))):.4f}",
            f"- Micro AUC-ROC: {float(micro.get('auc_roc', float('nan'))):.4f}",
            f"- Micro F1: {float(micro.get('f1', float('nan'))):.4f}",
            "",
        ])

    generation = results.get("generation")
    if isinstance(generation, dict) and generation:
        lines.append("## Generation")
        for key, value in generation.items():
            if isinstance(value, (int, float, np.generic)):
                lines.append(f"- {key}: {float(value):.4f}")
        lines.append("")

    summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _save_classification_plot(classification: Dict[str, Any], output_path: Path) -> None:
    per_class = classification.get("per_class", {})
    class_names = list(per_class.keys())
    auc_values = [float(per_class[name].get("auc_roc", np.nan)) for name in class_names]
    f1_values = [float(per_class[name].get("f1", np.nan)) for name in class_names]
    ap_values = [float(per_class[name].get("average_precision", np.nan)) for name in class_names]

    macro = classification.get("macro", {})
    micro = classification.get("micro", {})

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    summary_labels = ["macro_auc", "macro_f1", "micro_auc", "micro_f1"]
    summary_values = [
        float(macro.get("auc_roc", np.nan)),
        float(macro.get("f1", np.nan)),
        float(micro.get("auc_roc", np.nan)),
        float(micro.get("f1", np.nan)),
    ]
    axes[0].bar(summary_labels, summary_values, color=["#335c67", "#9e2a2b", "#0a9396", "#bb3e03"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Classification Overview")
    axes[0].set_ylabel("Score")

    x = np.arange(len(class_names))
    width = 0.25
    axes[1].bar(x - width, auc_values, width=width, label="AUC-ROC", color="#335c67")
    axes[1].bar(x, f1_values, width=width, label="F1", color="#9e2a2b")
    axes[1].bar(x + width, ap_values, width=width, label="Average Precision", color="#e09f3e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Per-Class Metrics")
    axes[1].legend()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_generation_plot(generation: Dict[str, Any], output_path: Path) -> None:
    numeric_items = [
        (key, float(value))
        for key, value in generation.items()
        if isinstance(value, (int, float, np.generic))
    ]
    if not numeric_items:
        return

    keys = [item[0] for item in numeric_items]
    values = [item[1] for item in numeric_items]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar(keys, values, color="#386641")
    ax.set_ylim(min(0.0, min(values, default=0.0)), max(1.0, max(values, default=1.0)))
    ax.set_title("Generation Metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=45)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_reliability_plot(classification_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not classification_rows:
        return

    score_cols = sorted(key for key in classification_rows[0] if key.startswith("score_"))
    true_cols = [f"true_{column.removeprefix('score_')}" for column in score_cols]
    y_score = []
    y_true = []

    for row in classification_rows:
        for score_key, true_key in zip(score_cols, true_cols):
            true_value = row.get(true_key, "")
            if true_value in ("", None):
                continue
            y_true.append(float(true_value))
            y_score.append(float(row[score_key]))

    if not y_score:
        return

    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    bins = np.linspace(0.0, 1.0, 11)
    confidences = []
    accuracies = []

    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_score_arr >= left) & (y_score_arr <= right) if right == 1.0 else (y_score_arr >= left) & (y_score_arr < right)
        if not np.any(mask):
            continue
        confidences.append(float(np.mean(y_score_arr[mask])))
        accuracies.append(float(np.mean(y_true_arr[mask])))

    if not confidences:
        return

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="Ideal")
    ax.plot(confidences, accuracies, marker="o", color="#0a9396", label="Model")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram")
    ax.legend()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_artifacts(
    results: Dict[str, Any],
    output_dir: Path | str,
    *,
    classification_rows: Optional[list[dict[str, Any]]] = None,
    generation_rows: Optional[list[dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Persist evaluation results as JSON, Markdown, and plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}

    metrics_path = output_path / "metrics.json"
    metrics_path.write_text(json.dumps(_to_jsonable(results), indent=2), encoding="utf-8")
    artifacts["metrics_json"] = str(metrics_path)

    summary_path = output_path / "summary.md"
    _write_summary(results, summary_path)
    artifacts["summary_markdown"] = str(summary_path)

    classification = results.get("classification")
    if isinstance(classification, dict) and classification.get("per_class"):
        cls_plot_path = output_path / "classification_metrics.png"
        _save_classification_plot(classification, cls_plot_path)
        artifacts["classification_plot"] = str(cls_plot_path)

    if classification_rows:
        cls_rows_path = output_path / "classification_predictions.csv"
        with cls_rows_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(classification_rows[0].keys()))
            writer.writeheader()
            writer.writerows(classification_rows)
        artifacts["classification_predictions"] = str(cls_rows_path)

        reliability_path = output_path / "classification_reliability.png"
        _save_reliability_plot(classification_rows, reliability_path)
        if reliability_path.exists():
            artifacts["classification_reliability_plot"] = str(reliability_path)

    generation = results.get("generation")
    if isinstance(generation, dict) and generation:
        gen_plot_path = output_path / "generation_metrics.png"
        _save_generation_plot(generation, gen_plot_path)
        artifacts["generation_plot"] = str(gen_plot_path)

    if generation_rows:
        gen_rows_path = output_path / "generation_predictions.jsonl"
        with gen_rows_path.open("w", encoding="utf-8") as f:
            for row in generation_rows:
                f.write(json.dumps(_to_jsonable(row)) + "\n")
        artifacts["generation_predictions"] = str(gen_rows_path)

    return artifacts
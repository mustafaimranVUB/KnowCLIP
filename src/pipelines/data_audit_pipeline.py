"""Dataset split and report-distribution auditing for thesis reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.core.config import ProjectConfig, load_config
from src.data.dataset import MIMICCXRDataset
from src.data.splits import SplitManager

logger = logging.getLogger(__name__)


class DataAuditPipeline:
    """Compute split-level label and report-distribution summaries."""

    def __init__(self, config: ProjectConfig | str | Path | None = None) -> None:
        if isinstance(config, (str, Path)):
            self.config = load_config(Path(config))
        elif config is None:
            self.config = ProjectConfig()
        else:
            self.config = config

    def run(self, output_dir: Path | str | None = None) -> Dict[str, Any]:
        dc = self.config.data
        datasets = {
            split: MIMICCXRDataset(
                split=split,
                mimic_root=dc.mimic_root,
                reports_root=dc.reports_root,
                split_csv=dc.split_csv,
                chexpert_csv=dc.chexpert_csv,
                kg_artifacts_dir=dc.kg_artifacts_dir,
                include_graphs=False,
                split_strategy=dc.split_strategy,
                subset_seed=dc.subset_seed,
                subset_train_ratio=dc.subset_train_ratio,
                subset_val_ratio=dc.subset_val_ratio,
                subset_test_ratio=dc.subset_test_ratio,
                auto_min_val_samples=dc.auto_min_val_samples,
                auto_min_test_samples=dc.auto_min_test_samples,
                image_suffixes=dc.image_suffixes,
                enforce_all_labels_per_split=dc.enforce_all_labels_per_split,
            )
            for split in ("train", "validate", "test")
        }

        summary = {
            "splits": {split: ds.get_distribution_summary() for split, ds in datasets.items()},
            "all_labels_present": {
                split: len(ds.index_summary.get("missing_positive_labels", [])) == 0
                for split, ds in datasets.items()
            },
            "official_summary": SplitManager(dc.split_csv, dc.chexpert_csv).summary(),
        }

        output_path = Path(output_dir) if output_dir is not None else self.config.training.log_dir.parent / "data_audit"
        output_path.mkdir(parents=True, exist_ok=True)

        (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._write_markdown(summary, output_path / "summary.md")
        self._plot_label_prevalence(summary, output_path / "label_prevalence.png")
        self._plot_report_lengths(summary, output_path / "report_lengths.png")

        logger.info("Data audit written to %s", output_path)
        return summary

    def _write_markdown(self, summary: Dict[str, Any], output_path: Path) -> None:
        lines = ["# Data Audit", ""]
        for split, split_summary in summary["splits"].items():
            reports = split_summary.get("reports", {})
            lines.extend([
                f"## {split.title()}",
                f"- studies: {split_summary.get('num_studies', 0)}",
                f"- subjects: {split_summary.get('num_subjects', 0)}",
                f"- missing positive labels: {', '.join(split_summary.get('missing_positive_labels', [])) or 'none'}",
                f"- mean report tokens: {reports.get('mean_tokens', 0.0):.2f}",
                f"- p90 report tokens: {reports.get('p90_tokens', 0.0):.2f}",
                f"- unique report ratio: {reports.get('unique_report_ratio', 0.0):.4f}",
                "",
            ])
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _plot_label_prevalence(self, summary: Dict[str, Any], output_path: Path) -> None:
        labels = SplitManager.CHEXPERT_LABELS
        splits = ["train", "validate", "test"]
        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
        colors = {"train": "#335c67", "validate": "#9e2a2b", "test": "#386641"}
        for offset, split in enumerate(splits):
            prevalences = [
                float(summary["splits"][split].get("labels", {}).get(label, {}).get("prevalence", 0.0))
                for label in labels
            ]
            ax.bar(x + (offset - 1) * width, prevalences, width=width, label=split, color=colors[split])

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Positive prevalence")
        ax.set_title("Label prevalence by split")
        ax.legend()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_report_lengths(self, summary: Dict[str, Any], output_path: Path) -> None:
        splits = ["train", "validate", "test"]
        means = [float(summary["splits"][split].get("reports", {}).get("mean_tokens", 0.0)) for split in splits]
        p90s = [float(summary["splits"][split].get("reports", {}).get("p90_tokens", 0.0)) for split in splits]

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        x = np.arange(len(splits))
        ax.bar(x - 0.15, means, width=0.3, label="mean", color="#0a9396")
        ax.bar(x + 0.15, p90s, width=0.3, label="p90", color="#ee9b00")
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylabel("Tokens")
        ax.set_title("Report length distribution by split")
        ax.legend()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
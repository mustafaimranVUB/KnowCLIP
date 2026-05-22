"""Phase II — Neuro-Symbolic Model Training Pipeline.

Orchestrates:
1. Configuration loading
2. Dataset & DataLoader construction
3. Model assembly (via factory)
4. Training loop invocation
5. Evaluation
"""

from __future__ import annotations

import csv
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.core.config import ProjectConfig, load_config
from src.core.utils import get_device, set_seed, setup_logging
from src.data.dataset import MIMICCXRDataset, collate_mimic
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.model_factory import MedicalVLM, build_model
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end Phase II training orchestrator.

    Parameters:
        config: Full project configuration (or path to YAML).
        device: Torch device override.
    """

    def __init__(
        self,
        config: ProjectConfig | str | Path | None = None,
        device: Optional[str] = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.config = load_config(Path(config))
        elif config is None:
            self.config = ProjectConfig()
        else:
            self.config = config

        self.device = torch.device(device) if device else get_device()
        self._model: Optional[MedicalVLM] = None
        self._trainer: Optional[Trainer] = None
        self._report_tokenizer = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise seed, logging, and GPU determinism."""
        set_seed(self.config.training.seed)
        setup_logging(
            log_dir=self.config.training.log_dir,
            experiment_name="phase2_training",
        )
        logger.info("Device: %s", self.device)
        logger.info("Config: %s", self.config.to_dict())
        self._kg_quality_preflight()

    def _kg_quality_preflight(self) -> None:
        """Validate minimal KG quality requirements before Phase II when KG is enabled."""
        if not self.config.model.use_kg:
            return

        kg_dir = Path(self.config.data.kg_artifacts_dir)
        threshold = float(self.config.kg_pipeline.min_grounding_coverage)
        if threshold <= 0.0:
            return

        coverage = self._read_grounding_coverage(kg_dir)
        if coverage is None:
            logger.warning(
                "KG preflight: could not determine grounding coverage from %s; continuing.",
                kg_dir,
            )
            return

        logger.info("KG preflight: grounding coverage=%.4f (required>=%.4f)", coverage, threshold)
        if coverage < threshold and self.config.kg_pipeline.fail_on_low_grounding_coverage:
            raise RuntimeError(
                f"KG grounding coverage {coverage:.4f} is below threshold {threshold:.4f}."
            )

    def _read_grounding_coverage(self, kg_dir: Path) -> Optional[float]:
        summary_path = kg_dir / "phase1_summary.json"
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                # Handle nested grounding_coverage dict (e.g. {"coverage_pct": 95.56, ...}).
                gc = data.get("grounding_coverage")
                if isinstance(gc, dict):
                    for sub_key in ("coverage_pct", "coverage", "grounded_fraction"):
                        if sub_key in gc:
                            val = float(gc[sub_key])
                            # Normalise percentage (>1) to fraction.
                            return val / 100.0 if val > 1.0 else val
                # Flat scalar keys.
                for key in (
                    "grounding_coverage",
                    "coverage",
                    "cui_coverage",
                    "grounded_fraction",
                ):
                    if key in data and not isinstance(data[key], dict):
                        val = float(data[key])
                        return val / 100.0 if val > 1.0 else val
            except Exception:
                pass

        csv_path = kg_dir / "grounding_summary.csv"
        if csv_path.exists():
            try:
                with csv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        metric = row.get("metric", "").lower()
                        if metric in ("coverage_pct", "coverage", "grounding_coverage"):
                            val = float(row.get("value", ""))
                            return val / 100.0 if val > 1.0 else val
            except Exception:
                pass

        return None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def build_dataloaders(
        self,
        batch_size: Optional[int] = None,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build train / val / test DataLoaders.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        dc = self.config.data

        train_ds = MIMICCXRDataset(
            split="train",
            mimic_root=dc.mimic_root,
            reports_root=dc.reports_root,
            split_csv=dc.split_csv,
            chexpert_csv=dc.chexpert_csv,
            kg_artifacts_dir=dc.kg_artifacts_dir,
            transforms=get_train_transforms(),
            dev_subset_frac=dc.dev_subset_frac,
            include_graphs=self.config.model.use_kg,
            split_strategy=dc.split_strategy,
            subset_seed=dc.subset_seed,
            subset_train_ratio=dc.subset_train_ratio,
            subset_val_ratio=dc.subset_val_ratio,
            subset_test_ratio=dc.subset_test_ratio,
            auto_min_val_samples=dc.auto_min_val_samples,
            auto_min_test_samples=dc.auto_min_test_samples,
            image_suffixes=dc.image_suffixes,
            enforce_all_labels_per_split=dc.enforce_all_labels_per_split,
            report_section_preference=dc.report_section_preference,
        )
        val_ds = MIMICCXRDataset(
            split="validate",
            mimic_root=dc.mimic_root,
            reports_root=dc.reports_root,
            split_csv=dc.split_csv,
            chexpert_csv=dc.chexpert_csv,
            kg_artifacts_dir=dc.kg_artifacts_dir,
            transforms=get_eval_transforms(),
            dev_subset_frac=dc.dev_subset_frac,
            include_graphs=self.config.model.use_kg,
            split_strategy=dc.split_strategy,
            subset_seed=dc.subset_seed,
            subset_train_ratio=dc.subset_train_ratio,
            subset_val_ratio=dc.subset_val_ratio,
            subset_test_ratio=dc.subset_test_ratio,
            auto_min_val_samples=dc.auto_min_val_samples,
            auto_min_test_samples=dc.auto_min_test_samples,
            image_suffixes=dc.image_suffixes,
            enforce_all_labels_per_split=dc.enforce_all_labels_per_split,
            report_section_preference=dc.report_section_preference,
        )
        test_ds = MIMICCXRDataset(
            split="test",
            mimic_root=dc.mimic_root,
            reports_root=dc.reports_root,
            split_csv=dc.split_csv,
            chexpert_csv=dc.chexpert_csv,
            kg_artifacts_dir=dc.kg_artifacts_dir,
            transforms=get_eval_transforms(),
            include_graphs=self.config.model.use_kg,
            split_strategy=dc.split_strategy,
            subset_seed=dc.subset_seed,
            subset_train_ratio=dc.subset_train_ratio,
            subset_val_ratio=dc.subset_val_ratio,
            subset_test_ratio=dc.subset_test_ratio,
            auto_min_val_samples=dc.auto_min_val_samples,
            auto_min_test_samples=dc.auto_min_test_samples,
            image_suffixes=dc.image_suffixes,
            enforce_all_labels_per_split=dc.enforce_all_labels_per_split,
            report_section_preference=dc.report_section_preference,
        )

        if len(train_ds) == 0:
            raise RuntimeError(
                "No training samples available after filtering for existing JPG/PNG files. "
                "For subset runs, ensure at least part of the train split is downloaded "
                "(e.g., p10 subset + matching metadata)."
            )
        if len(val_ds) == 0:
            logger.warning("Validation split is empty for current subset; validation metrics will be unstable.")
        if len(test_ds) == 0:
            logger.warning("Test split is empty for current subset; final evaluation will be skipped/empty.")

        report_collate = partial(
            collate_mimic,
            max_report_length=self.config.model.report_generation.max_report_length,
        )

        common_kwargs = dict(
            batch_size=batch_size or dc.batch_size,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory and torch.cuda.is_available(),
            collate_fn=report_collate,
            persistent_workers=dc.num_workers > 0,
        )

        train_sampler = self._build_train_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            **common_kwargs,
        )
        val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
        test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

        logger.info(
            "Datasets — train: %d, val: %d, test: %d",
            len(train_ds),
            len(val_ds),
            len(test_ds),
        )
        logger.info("Train split summary: %s", train_ds.index_summary)
        logger.info("Validation split summary: %s", val_ds.index_summary)
        logger.info("Test split summary: %s", test_ds.index_summary)

        return train_loader, val_loader, test_loader

    def build_eval_dataloaders(
        self,
        batch_size: Optional[int] = None,
    ) -> tuple[DataLoader, DataLoader]:
        """Build validation and test DataLoaders for evaluation-only flows.

        This avoids materialising the train split and its graph artifacts,
        which significantly reduces host memory usage during checkpoint eval.
        """
        dc = self.config.data

        val_ds = MIMICCXRDataset(
            split="validate",
            mimic_root=dc.mimic_root,
            reports_root=dc.reports_root,
            split_csv=dc.split_csv,
            chexpert_csv=dc.chexpert_csv,
            kg_artifacts_dir=dc.kg_artifacts_dir,
            transforms=get_eval_transforms(),
            dev_subset_frac=dc.dev_subset_frac,
            include_graphs=self.config.model.use_kg,
            split_strategy=dc.split_strategy,
            subset_seed=dc.subset_seed,
            subset_train_ratio=dc.subset_train_ratio,
            subset_val_ratio=dc.subset_val_ratio,
            subset_test_ratio=dc.subset_test_ratio,
            auto_min_val_samples=dc.auto_min_val_samples,
            auto_min_test_samples=dc.auto_min_test_samples,
            image_suffixes=dc.image_suffixes,
            enforce_all_labels_per_split=dc.enforce_all_labels_per_split,
            report_section_preference=dc.report_section_preference,
        )
        test_ds = MIMICCXRDataset(
            split="test",
            mimic_root=dc.mimic_root,
            reports_root=dc.reports_root,
            split_csv=dc.split_csv,
            chexpert_csv=dc.chexpert_csv,
            kg_artifacts_dir=dc.kg_artifacts_dir,
            transforms=get_eval_transforms(),
            include_graphs=self.config.model.use_kg,
            split_strategy=dc.split_strategy,
            subset_seed=dc.subset_seed,
            subset_train_ratio=dc.subset_train_ratio,
            subset_val_ratio=dc.subset_val_ratio,
            subset_test_ratio=dc.subset_test_ratio,
            auto_min_val_samples=dc.auto_min_val_samples,
            auto_min_test_samples=dc.auto_min_test_samples,
            image_suffixes=dc.image_suffixes,
            enforce_all_labels_per_split=dc.enforce_all_labels_per_split,
            report_section_preference=dc.report_section_preference,
        )

        if len(val_ds) == 0:
            logger.warning("Validation split is empty for current subset; threshold tuning will be skipped.")
        if len(test_ds) == 0:
            logger.warning("Test split is empty for current subset; evaluation output will be empty.")

        report_collate = partial(
            collate_mimic,
            max_report_length=self.config.model.report_generation.max_report_length,
        )

        common_kwargs = dict(
            batch_size=batch_size or dc.batch_size,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory and torch.cuda.is_available(),
            collate_fn=report_collate,
            persistent_workers=dc.num_workers > 0,
        )

        val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
        test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

        logger.info(
            "Eval datasets — val: %d, test: %d",
            len(val_ds),
            len(test_ds),
        )
        logger.info("Validation split summary: %s", val_ds.index_summary)
        logger.info("Test split summary: %s", test_ds.index_summary)

        return val_loader, test_loader

    def _build_train_sampler(self, train_ds: MIMICCXRDataset) -> Optional[WeightedRandomSampler]:
        if self.config.training.sampler_strategy == "none" or len(train_ds) == 0:
            return None

        label_matrix = torch.stack([sample["labels"] for sample in train_ds.samples]).float()
        valid = (~torch.isnan(label_matrix)) & (label_matrix >= 0.0) & (label_matrix <= 1.0)
        positives = (label_matrix == 1.0) & valid
        pos_counts = positives.sum(dim=0).float().clamp_min(1.0)
        inverse_frequency = (len(train_ds) / pos_counts).pow(self.config.training.sampler_rare_label_power)

        class_names = self.config.model.classification_head.class_names
        no_finding_index = class_names.index("No Finding") if "No Finding" in class_names else None

        weights = []
        for row in label_matrix:
            positive_indices = torch.nonzero(row == 1.0, as_tuple=False).flatten()
            if positive_indices.numel() > 0:
                sample_weight = float(inverse_frequency[positive_indices].mean().item())
                if no_finding_index is not None and positive_indices.numel() == 1 and int(positive_indices[0]) == no_finding_index:
                    sample_weight *= 0.35
            else:
                sample_weight = 0.5
            sample_weight = max(0.05, min(float(self.config.training.sampler_max_weight), sample_weight))
            weights.append(sample_weight)

        weights_tensor = torch.tensor(weights, dtype=torch.double)
        logger.info(
            "Using balanced_multilabel sampler: min=%.3f median=%.3f max=%.3f",
            float(weights_tensor.min().item()),
            float(weights_tensor.median().item()),
            float(weights_tensor.max().item()),
        )
        return WeightedRandomSampler(weights_tensor, num_samples=len(train_ds), replacement=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def build_model(self) -> MedicalVLM:
        """Construct the model via the factory."""
        self._model = build_model(self.config.model)
        return self._model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            resume_from: Optional checkpoint path to resume from.

        Returns:
            Training history dict.
        """
        if self._model is None:
            self.build_model()

        assert self._model is not None

        self._trainer = Trainer(
            model=self._model,
            config=self.config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
        )

        if resume_from:
            self._trainer.load_checkpoint(resume_from)
            logger.info("Resumed from checkpoint: %s", resume_from)

        history = self._trainer.train()
        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_path: Optional[str | Path] = None,
        artifact_dir: Optional[str | Path] = None,
        threshold_loader: Optional[DataLoader] = None,
        generation_max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate on the test set.

        Args:
            test_loader: Test DataLoader.
            checkpoint_path: Path to best model checkpoint.
            artifact_dir: Optional directory for plots and JSON exports.

        Returns:
            Evaluation metrics dict.
        """
        if self._model is None:
            self.build_model()

        assert self._model is not None

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self._model.to(self.device)
        self._model.eval()

        # Run classification evaluation
        from src.evaluation.classification_metrics import ClassificationEvaluator

        cls_evaluator = ClassificationEvaluator(
            class_names=self.config.model.classification_head.class_names,
        )

        results: Dict[str, Any] = {}

        thresholds = None
        if threshold_loader is not None and self.config.training.optimize_thresholds_on_validation:
            threshold_payload = self._collect_classification_outputs(threshold_loader)
            if threshold_payload["scores"] is not None and threshold_payload["labels"] is not None:
                thresholds = cls_evaluator.find_optimal_thresholds(
                    threshold_payload["labels"],
                    threshold_payload["scores"],
                )
                results["classification_thresholds"] = thresholds

        cls_payload = self._collect_classification_outputs(test_loader)

        classification_rows = []
        if cls_payload["scores"] is not None and cls_payload["labels"] is not None:
            cls_results = cls_evaluator.evaluate(
                y_true=cls_payload["labels"],
                y_score=cls_payload["scores"],
                thresholds=thresholds,
            )
            results["classification"] = cls_results
            logger.info("Classification results: %s", cls_results)
            classification_rows = self._build_classification_rows(
                study_keys=cls_payload["study_keys"],
                labels=cls_payload["labels"],
                scores=cls_payload["scores"],
                thresholds=cls_results.get("thresholds"),
            )

        # Report generation evaluation (if applicable)
        generation_rows = []
        if self.config.model.enable_report_generation:
            generation_results, generation_rows = self._evaluate_generation(
                test_loader,
                return_predictions=artifact_dir is not None,
                max_samples=generation_max_samples if generation_max_samples is not None else self.config.training.generation_eval_max_samples,
            )
            results["generation"] = generation_results

        if artifact_dir is not None:
            from src.evaluation.artifact_export import save_evaluation_artifacts

            results["artifacts"] = save_evaluation_artifacts(
                results,
                artifact_dir,
                classification_rows=classification_rows,
                generation_rows=generation_rows,
            )

        return results

    def _collect_classification_outputs(self, loader: DataLoader) -> Dict[str, Any]:
        assert self._model is not None

        study_keys: list[str] = []
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                pixel_values = batch["image"].to(self.device)
                labels = batch.get("labels")
                graph_x = self._tensor_to_device(batch.get("graph_x"))
                graph_edge_index = self._tensor_to_device(batch.get("graph_edge_index"))
                graph_edge_type = self._tensor_to_device(batch.get("graph_edge_type"))
                graph_batch = self._tensor_to_device(batch.get("graph_batch"))

                outputs = self._model(
                    pixel_values=pixel_values,
                    graph_x=graph_x,
                    graph_edge_index=graph_edge_index,
                    graph_edge_type=graph_edge_type,
                    graph_batch=graph_batch,
                )

                if "classification_logits" in outputs and labels is not None:
                    all_preds.append(torch.sigmoid(outputs["classification_logits"]).cpu())
                    all_labels.append(labels.cpu())
                    study_keys.extend([str(value) for value in batch.get("study_key", [])])

        return {
            "study_keys": study_keys,
            "scores": torch.cat(all_preds, dim=0).numpy() if all_preds else None,
            "labels": torch.cat(all_labels, dim=0).numpy() if all_labels else None,
        }

    def _build_classification_rows(
        self,
        study_keys: list[str],
        labels: np.ndarray,
        scores: np.ndarray,
        thresholds: Optional[Dict[str, float]],
    ) -> list[dict[str, Any]]:
        if scores.size == 0:
            return []

        class_names = self.config.model.classification_head.class_names
        threshold_array = np.asarray([
            float(thresholds.get(name, 0.5)) if thresholds else 0.5
            for name in class_names
        ], dtype=np.float32)
        preds = (scores >= threshold_array.reshape(1, -1)).astype(int)

        rows: list[dict[str, Any]] = []
        for row_index, study_key in enumerate(study_keys):
            row: dict[str, Any] = {"study_key": study_key}
            for class_index, class_name in enumerate(class_names):
                safe_name = class_name.lower().replace(" ", "_").replace("-", "_")
                label_value = labels[row_index, class_index]
                row[f"true_{safe_name}"] = "" if np.isnan(label_value) else float(label_value)
                row[f"score_{safe_name}"] = float(scores[row_index, class_index])
                row[f"pred_{safe_name}"] = int(preds[row_index, class_index])
            rows.append(row)
        return rows

    def load_checkpoint(self, checkpoint_path: str | Path) -> Path:
        """Load a checkpoint with the same corruption fallback used for evaluation."""
        if self._model is None:
            self.build_model()

        assert self._model is not None

        checkpoint_path = Path(checkpoint_path)
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self._model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded checkpoint: %s", checkpoint_path)
            return checkpoint_path
        except RuntimeError as exc:
            if "zip archive" not in str(exc).lower() and "central directory" not in str(exc).lower():
                raise

            ckpt_dir = checkpoint_path.parent
            periodic = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"), reverse=True)
            for fallback in periodic:
                try:
                    ckpt = torch.load(str(fallback), map_location=self.device, weights_only=False)
                    self._model.load_state_dict(ckpt["model_state_dict"])
                    logger.warning(
                        "Checkpoint %s is corrupt (%s); fell back to %s",
                        checkpoint_path,
                        exc,
                        fallback.name,
                    )
                    return fallback
                except Exception:
                    continue

            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is corrupt and no valid periodic checkpoints found in {ckpt_dir}"
            ) from exc

    def _evaluate_generation(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False,
        max_samples: Optional[int] = None,
    ) -> tuple[Dict[str, Any], list[dict[str, Any]]]:
        """Generate reports and compute NLG metrics."""
        from src.evaluation.generation_metrics import GenerationEvaluator

        assert self._model is not None

        gen_evaluator = GenerationEvaluator()
        references: list[str] = []
        hypotheses: list[str] = []
        prediction_rows: list[dict[str, Any]] = []
        debug_budget = max(0, int(self.config.training.generation_debug_samples))
        debug_count = 0
        evaluated_samples = 0
        if max_samples is not None and max_samples > 0:
            logger.warning(
                "Generation evaluation capped at %d samples to avoid long-running jobs.",
                max_samples,
            )

        with torch.no_grad():
            for batch in test_loader:
                batch_size = int(batch["image"].shape[0])
                if max_samples is not None and evaluated_samples + batch_size > max_samples:
                    break
                pixel_values = batch["image"].to(self.device)
                ref_texts = batch.get("report_text", [])
                graph_x = self._tensor_to_device(batch.get("graph_x"))
                graph_edge_index = self._tensor_to_device(batch.get("graph_edge_index"))
                graph_edge_type = self._tensor_to_device(batch.get("graph_edge_type"))
                graph_batch = self._tensor_to_device(batch.get("graph_batch"))

                generated = self._model.generate_report(
                    pixel_values=pixel_values,
                    graph_x=graph_x,
                    graph_edge_index=graph_edge_index,
                    graph_edge_type=graph_edge_type,
                    graph_batch=graph_batch,
                    max_length=self.config.model.report_generation.max_report_length,
                )

                batch_hypotheses: list[str] = []
                if isinstance(generated, list):
                    for item in generated:
                        if isinstance(item, str):
                            batch_hypotheses.append(item)
                        elif isinstance(item, list):
                            batch_hypotheses.append(self._decode_generated_tokens(item))
                elif isinstance(generated, str):
                    batch_hypotheses.append(generated)
                hypotheses.extend(batch_hypotheses)

                if isinstance(ref_texts, list):
                    references.extend(ref_texts)
                elif isinstance(ref_texts, str):
                    references.append(ref_texts)

                if return_predictions:
                    batch_keys = [str(value) for value in batch.get("study_key", [])]
                    for sample_index, hypothesis in enumerate(batch_hypotheses):
                        reference = ref_texts[sample_index] if isinstance(ref_texts, list) and sample_index < len(ref_texts) else ""
                        prediction_rows.append({
                            "study_key": batch_keys[sample_index] if sample_index < len(batch_keys) else str(len(prediction_rows)),
                            "reference_report": reference,
                            "generated_report": hypothesis,
                            "reference_length": len(reference.split()),
                            "generated_length": len(hypothesis.split()),
                        })

                evaluated_samples += len(batch_hypotheses)

                # Debug raw decoded generations to diagnose mode collapse.
                while debug_count < debug_budget and debug_count < len(hypotheses):
                    ref = references[debug_count] if debug_count < len(references) else ""
                    logger.info(
                        "Generation debug sample %d | hyp='%s' | ref='%s'",
                        debug_count,
                        hypotheses[debug_count],
                        ref,
                    )
                    debug_count += 1

        if not references or not hypotheses:
            logger.warning("No references/hypotheses for generation evaluation.")
            return {}, prediction_rows

        gen_results = gen_evaluator.evaluate(hypotheses, references)
        gen_results.update(self._generation_health_diagnostics(hypotheses, references))
        logger.info("Generation results: %s", gen_results)
        return gen_results, prediction_rows

    def _generation_health_diagnostics(self, hypotheses: list[str], references: Optional[list[str]] = None) -> Dict[str, float]:
        """Compute lightweight collapse diagnostics for generated text."""
        if not hypotheses:
            return {
                "gen_repetition_ratio": 1.0,
                "gen_distinct_1": 0.0,
                "gen_distinct_2": 0.0,
                "gen_average_length": 0.0,
                "gen_unique_report_ratio": 0.0,
            }

        total_tokens = 0
        repeated_tokens = 0
        unigram_set = set()
        bigram_set = set()
        unique_reports = {hyp.strip() for hyp in hypotheses if hyp.strip()}

        for hyp in hypotheses:
            toks = hyp.split()
            total_tokens += len(toks)
            unigram_set.update(toks)
            if len(toks) > 1:
                bigrams = list(zip(toks[:-1], toks[1:]))
                bigram_set.update(bigrams)

            seen = set()
            for t in toks:
                if t in seen:
                    repeated_tokens += 1
                else:
                    seen.add(t)

        total_tokens = max(total_tokens, 1)
        total_bigrams = max(total_tokens - len(hypotheses), 1)

        diagnostics = {
            "gen_repetition_ratio": repeated_tokens / total_tokens,
            "gen_distinct_1": len(unigram_set) / total_tokens,
            "gen_distinct_2": len(bigram_set) / total_bigrams,
            "gen_average_length": total_tokens / max(len(hypotheses), 1),
            "gen_unique_report_ratio": len(unique_reports) / max(len(hypotheses), 1),
        }
        if references:
            reference_tokens = sum(len(ref.split()) for ref in references)
            diagnostics["gen_length_ratio"] = total_tokens / max(reference_tokens, 1)
        return diagnostics

    def _tensor_to_device(self, value: Any) -> Any:
        """Move tensor-like values to pipeline device, pass through non-tensors."""
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return value

    def _decode_generated_tokens(self, token_ids: list[int]) -> str:
        """Decode generated token IDs for human-readable debugging."""
        if len(token_ids) == 0:
            return ""

        if self._report_tokenizer is None:
            try:
                from transformers import AutoTokenizer  # type: ignore

                self._report_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception:
                self._report_tokenizer = False

        if self._report_tokenizer not in (None, False):
            try:
                text = self._report_tokenizer.decode(token_ids, skip_special_tokens=True)
                return text.strip()
            except Exception:
                pass

        return " ".join(str(tok) for tok in token_ids)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        resume_from: Optional[str | Path] = None,
        evaluate_after: bool = True,
    ) -> Dict[str, Any]:
        """Execute the full Phase II pipeline.

        Args:
            resume_from: Optional checkpoint path.
            evaluate_after: Whether to run evaluation after training.

        Returns:
            Combined results dict.
        """
        self.setup()

        train_loader, val_loader, test_loader = self.build_dataloaders()
        self.build_model()

        history = self.train(train_loader, val_loader, resume_from=resume_from)

        results: Dict[str, Any] = {"training_history": history}

        if evaluate_after:
            best_ckpt_from_run = history.get("best_checkpoint_path") if isinstance(history, dict) else None
            if best_ckpt_from_run and Path(best_ckpt_from_run).exists():
                eval_results = self.evaluate(
                    test_loader,
                    checkpoint_path=Path(best_ckpt_from_run),
                    threshold_loader=val_loader,
                )
            else:
                best_ckpt = self.config.training.checkpoint_dir / "best_model.pt"
                if best_ckpt.exists():
                    logger.warning(
                        "No best checkpoint saved during current run; using existing best checkpoint on disk: %s",
                        best_ckpt,
                    )
                    eval_results = self.evaluate(
                        test_loader,
                        checkpoint_path=best_ckpt,
                        threshold_loader=val_loader,
                    )
                    results["evaluation_checkpoint_warning"] = (
                        "Evaluation used pre-existing best_model.pt because current run did not save one."
                    )
                else:
                    eval_results = self.evaluate(test_loader, threshold_loader=val_loader)

            results["evaluation"] = eval_results

        return results

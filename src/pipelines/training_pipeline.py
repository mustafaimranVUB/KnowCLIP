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
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

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
        )

        if len(train_ds) == 0:
            raise RuntimeError(
                "No training samples available after filtering for existing DICOM files. "
                "For subset runs, ensure at least part of the train split is downloaded "
                "(e.g., p10 subset + matching metadata)."
            )
        if len(val_ds) == 0:
            logger.warning("Validation split is empty for current subset; validation metrics will be unstable.")
        if len(test_ds) == 0:
            logger.warning("Test split is empty for current subset; final evaluation will be skipped/empty.")

        common_kwargs = dict(
            batch_size=dc.batch_size,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory and torch.cuda.is_available(),
            collate_fn=collate_mimic,
            persistent_workers=dc.num_workers > 0,
        )

        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
        test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

        logger.info(
            "Datasets — train: %d, val: %d, test: %d",
            len(train_ds),
            len(val_ds),
            len(test_ds),
        )

        return train_loader, val_loader, test_loader

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
    ) -> Dict[str, Any]:
        """Evaluate on the test set.

        Args:
            test_loader: Test DataLoader.
            checkpoint_path: Path to best model checkpoint.

        Returns:
            Evaluation metrics dict.
        """
        if self._model is None:
            self.build_model()

        assert self._model is not None

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self._model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded checkpoint for evaluation: %s", checkpoint_path)

        self._model.to(self.device)
        self._model.eval()

        # Run classification evaluation
        from src.evaluation.classification_metrics import ClassificationEvaluator

        cls_evaluator = ClassificationEvaluator(
            class_names=self.config.model.classification_head.class_names,
        )

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
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
                    probs = torch.sigmoid(outputs["classification_logits"]).cpu()
                    all_preds.append(probs)
                    all_labels.append(labels)

        results: Dict[str, Any] = {}

        if all_preds:
            preds_tensor = torch.cat(all_preds, dim=0).numpy()
            labels_tensor = torch.cat(all_labels, dim=0).numpy()
            cls_results = cls_evaluator.evaluate(y_true=labels_tensor, y_score=preds_tensor)
            results["classification"] = cls_results
            logger.info("Classification results: %s", cls_results)

        # Report generation evaluation (if applicable)
        if self.config.model.enable_report_generation and self.config.model.use_kg:
            results["generation"] = self._evaluate_generation(test_loader)

        return results

    def _evaluate_generation(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Generate reports and compute NLG metrics."""
        from src.evaluation.generation_metrics import GenerationEvaluator

        assert self._model is not None

        gen_evaluator = GenerationEvaluator()
        references: list[str] = []
        hypotheses: list[str] = []
        debug_budget = max(0, int(self.config.training.generation_debug_samples))
        debug_count = 0

        with torch.no_grad():
            for batch in test_loader:
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
                )

                if isinstance(generated, list):
                    for item in generated:
                        if isinstance(item, str):
                            hypotheses.append(item)
                        elif isinstance(item, list):
                            hypotheses.append(self._decode_generated_tokens(item))
                elif isinstance(generated, str):
                    hypotheses.append(generated)

                if isinstance(ref_texts, list):
                    references.extend(ref_texts)
                elif isinstance(ref_texts, str):
                    references.append(ref_texts)

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
            return {}

        gen_results = gen_evaluator.evaluate(hypotheses, references)
        gen_results.update(self._generation_health_diagnostics(hypotheses))
        logger.info("Generation results: %s", gen_results)
        return gen_results

    def _generation_health_diagnostics(self, hypotheses: list[str]) -> Dict[str, float]:
        """Compute lightweight collapse diagnostics for generated text."""
        if not hypotheses:
            return {
                "gen_repetition_ratio": 1.0,
                "gen_distinct_1": 0.0,
                "gen_distinct_2": 0.0,
            }

        total_tokens = 0
        repeated_tokens = 0
        unigram_set = set()
        bigram_set = set()

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

        return {
            "gen_repetition_ratio": repeated_tokens / total_tokens,
            "gen_distinct_1": len(unigram_set) / total_tokens,
            "gen_distinct_2": len(bigram_set) / total_bigrams,
        }

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
                eval_results = self.evaluate(test_loader, checkpoint_path=Path(best_ckpt_from_run))
            else:
                best_ckpt = self.config.training.checkpoint_dir / "best_model.pt"
                if best_ckpt.exists():
                    logger.warning(
                        "No best checkpoint saved during current run; using existing best checkpoint on disk: %s",
                        best_ckpt,
                    )
                    eval_results = self.evaluate(test_loader, checkpoint_path=best_ckpt)
                    results["evaluation_checkpoint_warning"] = (
                        "Evaluation used pre-existing best_model.pt because current run did not save one."
                    )
                else:
                    eval_results = self.evaluate(test_loader)

            results["evaluation"] = eval_results

        return results

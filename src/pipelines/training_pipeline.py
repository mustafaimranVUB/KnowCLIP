"""Phase II — Neuro-Symbolic Model Training Pipeline.

Orchestrates:
1. Configuration loading
2. Dataset & DataLoader construction
3. Model assembly (via factory)
4. Training loop invocation
5. Evaluation
"""

from __future__ import annotations

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
        )
        test_ds = MIMICCXRDataset(
            split="test",
            mimic_root=dc.mimic_root,
            reports_root=dc.reports_root,
            split_csv=dc.split_csv,
            chexpert_csv=dc.chexpert_csv,
            kg_artifacts_dir=dc.kg_artifacts_dir,
            transforms=get_eval_transforms(),
        )

        common_kwargs = dict(
            batch_size=dc.batch_size,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory and torch.cuda.is_available(),
            collate_fn=collate_mimic,
        )

        train_loader = DataLoader(train_ds, shuffle=True, **common_kwargs)
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
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch.get("labels")

                outputs = self._model(
                    pixel_values=pixel_values,
                    graph_x=batch.get("graph_x"),
                    graph_edge_index=batch.get("graph_edge_index"),
                    graph_edge_type=batch.get("graph_edge_type"),
                    graph_batch=batch.get("graph_batch"),
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

        with torch.no_grad():
            for batch in test_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                ref_texts = batch.get("report_text", [])

                generated = self._model.generate_report(
                    pixel_values=pixel_values,
                    graph_x=batch.get("graph_x"),
                    graph_edge_index=batch.get("graph_edge_index"),
                    graph_edge_type=batch.get("graph_edge_type"),
                    graph_batch=batch.get("graph_batch"),
                )

                if isinstance(generated, list):
                    hypotheses.extend(generated)
                else:
                    hypotheses.append(generated)

                if isinstance(ref_texts, list):
                    references.extend(ref_texts)
                elif isinstance(ref_texts, str):
                    references.append(ref_texts)

        if not references or not hypotheses:
            logger.warning("No references/hypotheses for generation evaluation.")
            return {}

        gen_results = gen_evaluator.evaluate(references, hypotheses)
        logger.info("Generation results: %s", gen_results)
        return gen_results

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
            best_ckpt = self.config.training.checkpoint_dir / "best_model.pt"
            if best_ckpt.exists():
                eval_results = self.evaluate(test_loader, checkpoint_path=best_ckpt)
            else:
                eval_results = self.evaluate(test_loader)
            results["evaluation"] = eval_results

        return results

"""Main training loop for KnoCLIP-XAI models.

Handles:
- Mixed-precision training (FP16)
- Gradient accumulation & clipping
- Checkpointing & early stopping
- Logging via Python logging + optional WandB / TensorBoard
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler  # type: ignore
from torch.utils.data import DataLoader

from src.core.config import ProjectConfig, TrainingConfig
from src.core.utils import get_device, get_git_hash, set_seed
from src.models.model_factory import MedicalVLM
from src.training.losses import MultiTaskLoss
from src.training.scheduler import get_linear_warmup_cosine_decay

logger = logging.getLogger(__name__)


def _autocast_ctx(enabled: bool):
    """Compatibility wrapper for AMP autocast across torch versions."""
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except AttributeError:
        from torch.cuda.amp import autocast  # type: ignore

        return autocast(enabled=enabled)


class Trainer:
    """Training orchestrator for :class:`MedicalVLM`.

    Parameters:
        model: The assembled model.
        config: Full project configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Torch device.
    """

    def __init__(
        self,
        model: MedicalVLM,
        config: ProjectConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.tc = config.training
        self.device = device or get_device()

        self.model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        self.criterion = MultiTaskLoss(
            classification_weight=self.tc.classification_loss_weight,
            generation_weight=self.tc.generation_loss_weight,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.tc.learning_rate,
            weight_decay=self.tc.weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * self.tc.num_epochs // self.tc.accumulation_steps
        self.scheduler = get_linear_warmup_cosine_decay(
            self.optimizer,
            warmup_steps=self.tc.warmup_steps,
            total_steps=total_steps,
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=self.tc.mixed_precision)

        # State
        self.global_step = 0
        self.best_val_metric = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dict with training summary (best_metric, epochs, etc.).
        """
        logger.info("Starting training: %d epochs, device=%s", self.tc.num_epochs, self.device)
        logger.info("Git hash: %s", get_git_hash())
        logger.info("Config: %s", self.config.to_dict())

        self.model.train()
        history: Dict[str, list] = {"train_loss": [], "val_loss": []}

        for epoch in range(self.start_epoch, self.tc.num_epochs):
            epoch_loss = self._train_epoch(epoch)
            history["train_loss"].append(epoch_loss)

            # Validation
            val_loss = None
            if self.val_loader is not None and (epoch + 1) % self.tc.eval_every_n_epochs == 0:
                val_loss = self._validate(epoch)
                history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < self.best_val_metric:
                    self.best_val_metric = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.tc.early_stopping_patience:
                        logger.warning(
                            "Early stopping at epoch %d (patience=%d)",
                            epoch,
                            self.tc.early_stopping_patience,
                        )
                        break

            # Periodic checkpoint
            if (epoch + 1) % self.tc.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%s | best=%.4f | patience=%d/%d",
                epoch + 1,
                self.tc.num_epochs,
                epoch_loss,
                f"{val_loss:.4f}" if val_loss is not None else "N/A",
                self.best_val_metric,
                self.patience_counter,
                self.tc.early_stopping_patience,
            )

        return {
            "best_val_metric": self.best_val_metric,
            "epochs_trained": epoch + 1,
            "history": history,
        }

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self._train_step(batch, batch_idx)
            total_loss += loss
            num_batches += 1

            if (batch_idx + 1) % self.tc.log_every_n_steps == 0:
                avg = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "  Step %d | loss=%.4f | avg=%.4f | lr=%.2e",
                    self.global_step,
                    loss,
                    avg,
                    lr,
                )

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: Dict[str, Any], batch_idx: int) -> float:
        """Execute a single training step."""
        # Move data to device
        images = batch["image"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Build forward kwargs
        forward_kwargs: Dict[str, Any] = {"pixel_values": images}

        # TODO: Add graph data handling when KG is available
        # For now, classification-only forward pass

        with _autocast_ctx(enabled=self.tc.mixed_precision):
            outputs = self.model(**forward_kwargs)

            losses = self.criterion(
                classification_logits=outputs.get("classification_logits"),
                classification_targets=labels,
                generation_logits=outputs.get("generation_logits"),
                generation_targets=batch.get("target_ids"),
            )
            loss = losses["total"] / self.tc.accumulation_steps

        # Backward
        self.scaler.scale(loss).backward()

        # Accumulation step
        if (batch_idx + 1) % self.tc.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tc.gradient_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

        return loss.item() * self.tc.accumulation_steps

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["labels"].to(self.device)

            forward_kwargs: Dict[str, Any] = {"pixel_values": images}

            outputs = self.model(**forward_kwargs)
            losses = self.criterion(
                classification_logits=outputs.get("classification_logits"),
                classification_targets=labels,
            )
            total_loss += losses["total"].item()
            num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a training checkpoint."""
        ckpt_dir = Path(self.tc.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_metric": self.best_val_metric,
            "config": self.config.to_dict(),
            "seed": self.tc.seed,
            "git_hash": get_git_hash(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "timestamp": datetime.now().isoformat(),
        }

        path = ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt"
        torch.save(checkpoint, str(path))
        logger.info("Saved checkpoint: %s", path)

        if is_best:
            best_path = ckpt_dir / "best_model.pt"
            torch.save(checkpoint, str(best_path))
            logger.info("Saved best model: %s", best_path)

        # Cleanup old checkpoints
        self._cleanup_checkpoints(ckpt_dir)

    def _cleanup_checkpoints(self, ckpt_dir: Path) -> None:
        """Keep only the last N checkpoints + best."""
        keep = self.tc.keep_last_n_checkpoints
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
        if len(ckpts) > keep:
            for old in ckpts[: len(ckpts) - keep]:
                old.unlink()
                logger.debug("Removed old checkpoint: %s", old)

    def load_checkpoint(self, path: Path | str) -> None:
        """Resume training from a checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        logger.info("Loading checkpoint: %s", path)
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_metric = ckpt.get("best_val_metric", float("inf"))

        logger.info(
            "Resumed from epoch %d, step %d, best=%.4f",
            self.start_epoch,
            self.global_step,
            self.best_val_metric,
        )

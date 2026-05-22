"""Main training loop for KnoCLIP-XAI models.

Handles:
- Mixed-precision training (FP16)
- Gradient accumulation & clipping
- Checkpointing & early stopping
- Logging via Python logging + optional WandB / TensorBoard
"""

from __future__ import annotations

import errno
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.config import ProjectConfig, TrainingConfig
from src.core.utils import get_device, get_git_hash, set_seed
from src.models.model_factory import MedicalVLM
from src.training.losses import MultiTaskLoss
from src.training.scheduler import get_linear_warmup_cosine_decay

logger = logging.getLogger(__name__)


def _safe_torch_save(obj: Dict[str, Any], path: Path) -> bool:
    """Atomically save a checkpoint, returning False instead of raising on I/O errors.

    After writing, the checkpoint is verified by partially loading it to
    confirm the zip archive is intact (guards against HPC filesystem
    corruption).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a per-process unique temp path to avoid collisions when multiple
    # workers/processes attempt to save the same checkpoint path.
    tmp_name = f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp_path = path.with_name(tmp_name)
    try:
        torch.save(obj, str(tmp_path))
        # Verify the file is a valid zip archive before promoting it.
        torch.load(str(tmp_path), map_location="cpu", weights_only=False)
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

        # Do not crash training because checkpoint I/O failed.
        err_no = getattr(exc, "errno", None)
        if err_no == errno.ENOSPC or "file write failed" in str(exc).lower():
            logger.error(
                "Checkpoint save failed (disk full/quota?) at %s: %s",
                path,
                exc,
            )
        else:
            logger.error("Checkpoint save failed at %s: %s", path, exc)
        return False


def _autocast_ctx(enabled: bool, device: torch.device | None = None):
    """Compatibility wrapper for AMP autocast across torch versions.

    Dynamically selects device_type so the same code works on both
    CUDA (Hydra) and CPU-only (local dev).  AMP autocast on CPU uses
    bfloat16 when available and is a no-op otherwise.
    """
    device_type = "cuda" if (device is not None and device.type == "cuda") else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # AMP on CPU is only meaningful with bfloat16; disable if not available.
    if device_type == "cpu" and not torch.cpu.is_available():
        enabled = False
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except AttributeError:
        if device_type == "cuda":
            from torch.cuda.amp import autocast  # type: ignore
            return autocast(enabled=enabled)
        # CPU fallback: no-op context manager
        import contextlib
        return contextlib.nullcontext()


def _build_grad_scaler(enabled: bool, device: torch.device | None = None):
    """Compatibility wrapper for GradScaler across torch versions.

    GradScaler is only useful on CUDA; on CPU we disable it
    regardless of the ``enabled`` flag.
    """
    device_type = "cuda" if (device is not None and device.type == "cuda") else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device_type != "cuda":
        enabled = False
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except AttributeError:
        from torch.cuda.amp import GradScaler  # type: ignore
        return GradScaler(enabled=enabled)


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

        class_pos_weight = self._compute_class_pos_weight() if self.tc.use_pos_weight else None

        # Loss
        self.criterion = MultiTaskLoss(
            classification_weight=self.tc.classification_loss_weight,
            generation_weight=self.tc.generation_loss_weight,
            label_smoothing=self.config.model.report_generation.label_smoothing,
            classification_loss_type=self.tc.classification_loss_type,
            class_pos_weight=class_pos_weight,
            focal_gamma=self.tc.focal_gamma,
            focal_alpha=self.tc.focal_alpha,
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
        self.scaler = _build_grad_scaler(enabled=self.tc.mixed_precision, device=self.device)

        # Stage scheduling state.
        self._stage_mode: Optional[str] = None
        self._base_classification_loss_weight = self.criterion.classification_weight
        self._warned_missing_generation_targets = False
        self._report_prompt_prefix_ids = self._build_report_prompt_prefix_ids()

        # State
        self.global_step = 0
        self.best_val_metric = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0
        self.last_best_checkpoint_path: Optional[Path] = None
        self._run_manifest_written = False

    def _build_report_prompt_prefix_ids(self) -> list[int]:
        """Resolve configured generation prompt prefix into token IDs."""
        prefix = str(getattr(self.config.model.report_generation, "prompt_prefix", "") or "").strip()
        if not prefix:
            return []

        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained("gpt2")
            token_ids = [int(t) for t in tok.encode(prefix, add_special_tokens=False)]
            if token_ids:
                logger.info("Using report prompt prefix '%s' (%d tokens)", prefix, len(token_ids))
            return token_ids
        except Exception as exc:
            logger.warning("Could not tokenize report prompt prefix '%s': %s", prefix, exc)
            return []

    def _apply_report_prompt_prefix(
        self,
        target_ids: Optional[torch.Tensor],
        generation_targets: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepend fixed prompt tokens after BOS for report generation targets."""
        if target_ids is None or not self._report_prompt_prefix_ids:
            return target_ids, generation_targets

        seq_len = int(target_ids.shape[1])
        if seq_len <= 1:
            return target_ids, generation_targets

        prefix_len = min(len(self._report_prompt_prefix_ids), seq_len - 1)
        prefix_tensor = torch.tensor(
            self._report_prompt_prefix_ids[:prefix_len],
            dtype=target_ids.dtype,
            device=target_ids.device,
        )

        prefixed_target = target_ids.clone()
        prefixed_target[:, 1 + prefix_len :] = target_ids[:, 1 : seq_len - prefix_len]
        prefixed_target[:, 1 : 1 + prefix_len] = prefix_tensor.unsqueeze(0)

        if generation_targets is None:
            return prefixed_target, generation_targets

        prefixed_generation = generation_targets.clone()
        prefixed_generation[:, prefix_len:] = generation_targets[:, : seq_len - prefix_len]
        prefixed_generation[:, :prefix_len] = prefix_tensor.unsqueeze(0)
        return prefixed_target, prefixed_generation

    def _compute_class_pos_weight(self) -> Optional[torch.Tensor]:
        """Estimate per-class positive weights from training labels.

        Computes ``neg / pos`` using only valid binary labels (0/1), and clamps
        the result to ``[1, max_pos_weight]`` to avoid destabilising gradients.
        """
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "samples"):
            logger.warning("Could not infer dataset samples for pos_weight; disabling pos_weight.")
            return None

        labels: list[torch.Tensor] = []
        for sample in getattr(dataset, "samples", []):
            y = sample.get("labels") if isinstance(sample, dict) else None
            if isinstance(y, torch.Tensor):
                labels.append(y.float())

        if not labels:
            logger.warning("No labels found to compute pos_weight; disabling pos_weight.")
            return None

        label_mat = torch.stack(labels, dim=0)
        valid = (~torch.isnan(label_mat)) & (label_mat >= 0.0) & (label_mat <= 1.0)

        pos = ((label_mat == 1.0) & valid).sum(dim=0).float()
        neg = ((label_mat == 0.0) & valid).sum(dim=0).float()
        pos_weight = neg / pos.clamp_min(1.0)
        pos_weight = torch.clamp(pos_weight, min=1.0, max=float(self.tc.max_pos_weight))

        logger.info("Using class pos_weight (clamped): %s", [round(float(x), 3) for x in pos_weight])
        return pos_weight

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
        self._write_run_manifest()

        self.model.train()
        history: Dict[str, list] = {
            "train_loss": [],
            "train_cls_loss": [],
            "train_gen_loss": [],
            "train_weighted_cls_loss": [],
            "train_weighted_gen_loss": [],
            "val_loss": [],
            "val_cls_loss": [],
            "val_gen_loss": [],
            "val_weighted_cls_loss": [],
            "val_weighted_gen_loss": [],
        }

        epoch = self.start_epoch - 1  # guard: keeps epoch defined even if the loop body never runs
        for epoch in range(self.start_epoch, self.tc.num_epochs):
            self._apply_two_stage_schedule(epoch)
            epoch_stats = self._train_epoch(epoch)
            epoch_loss = epoch_stats["total"]
            history["train_loss"].append(epoch_stats["total"])
            history["train_cls_loss"].append(epoch_stats["classification"])
            history["train_gen_loss"].append(epoch_stats["generation"])
            history["train_weighted_cls_loss"].append(epoch_stats["weighted_classification"])
            history["train_weighted_gen_loss"].append(epoch_stats["weighted_generation"])
            saved_this_epoch = False

            # Validation
            val_loss = None
            if self.val_loader is not None and (epoch + 1) % self.tc.eval_every_n_epochs == 0:
                val_stats = self._validate(epoch)
                val_loss = val_stats["total"]
                history["val_loss"].append(val_stats["total"])
                history["val_cls_loss"].append(val_stats["classification"])
                history["val_gen_loss"].append(val_stats["generation"])
                history["val_weighted_cls_loss"].append(val_stats["weighted_classification"])
                history["val_weighted_gen_loss"].append(val_stats["weighted_generation"])

                in_stage1 = self.tc.two_stage_training and (epoch + 1) <= self.tc.stage1_epochs
                if in_stage1 and not self.tc.stage1_enable_early_stopping:
                    # Stage-1 can be configured to always run fully before stage 2.
                    self.patience_counter = 0
                    logger.info(
                        "Stage-1 epoch %d/%d: early stopping disabled by config.",
                        epoch + 1,
                        self.tc.stage1_epochs,
                    )
                    continue

                # Early stopping / best-checkpoint selection
                selection_metric = self._selection_metric_from_val(val_stats)
                if self._passes_generation_guard(val_stats) and selection_metric < self.best_val_metric:
                    self.best_val_metric = selection_metric
                    self.patience_counter = 0
                    saved_this_epoch = self._save_checkpoint(epoch, is_best=True) or saved_this_epoch
                else:
                    self.patience_counter += 1
                    patience_limit = self.tc.early_stopping_patience
                    if in_stage1 and self.tc.stage1_early_stopping_patience is not None:
                        patience_limit = self.tc.stage1_early_stopping_patience
                    if self.patience_counter >= patience_limit:
                        logger.warning(
                            "Early stopping at epoch %d (patience=%d)",
                            epoch,
                            patience_limit,
                        )
                        break

            # Periodic checkpoint
            if (epoch + 1) % self.tc.save_every_n_epochs == 0 and not saved_this_epoch:
                self._save_checkpoint(epoch)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f (cls=%.4f, gen=%.4f, w_cls=%.4f, w_gen=%.4f) | val_loss=%s | val(cls=%.4f, gen=%.4f, w_cls=%.4f, w_gen=%.4f) | best=%.4f | patience=%d/%d",
                epoch + 1,
                self.tc.num_epochs,
                epoch_stats["total"],
                epoch_stats["classification"],
                epoch_stats["generation"],
                epoch_stats["weighted_classification"],
                epoch_stats["weighted_generation"],
                f"{val_loss:.4f}" if val_loss is not None else "N/A",
                val_stats["classification"] if val_loss is not None else float("nan"),
                val_stats["generation"] if val_loss is not None else float("nan"),
                val_stats["weighted_classification"] if val_loss is not None else float("nan"),
                val_stats["weighted_generation"] if val_loss is not None else float("nan"),
                self.best_val_metric,
                self.patience_counter,
                self.tc.early_stopping_patience,
            )

        return {
            "best_val_metric": self.best_val_metric,
            "epochs_trained": epoch + 1,
            "history": history,
            "best_checkpoint_path": str(self.last_best_checkpoint_path) if self.last_best_checkpoint_path else None,
        }

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_gen = 0.0
        total_w_cls = 0.0
        total_w_gen = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        teacher_forcing_ratio = self._teacher_forcing_ratio_for_epoch(epoch)
        for batch_idx, batch in enumerate(self.train_loader):
            step_stats = self._train_step(batch, batch_idx, teacher_forcing_ratio)
            loss = step_stats["total"]
            total_loss += step_stats["total"]
            total_cls += step_stats["classification"]
            total_gen += step_stats["generation"]
            total_w_cls += step_stats["weighted_classification"]
            total_w_gen += step_stats["weighted_generation"]
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

        denom = max(num_batches, 1)
        return {
            "total": total_loss / denom,
            "classification": total_cls / denom,
            "generation": total_gen / denom,
            "weighted_classification": total_w_cls / denom,
            "weighted_generation": total_w_gen / denom,
        }

    def _train_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        teacher_forcing_ratio: float,
    ) -> Dict[str, float]:
        """Execute a single training step."""
        # Move data to device
        images = batch["image"].to(self.device)
        labels = batch["labels"].to(self.device)
        target_ids = self._tensor_to_device(batch.get("target_ids"))
        generation_targets = self._tensor_to_device(batch.get("generation_targets"))
        target_ids, generation_targets = self._apply_report_prompt_prefix(target_ids, generation_targets)

        # Guardrail: if stage-1 disabled classification but no generation targets exist,
        # training would collapse to total_loss=0. Restore classification loss in that case.
        if generation_targets is None and self.criterion.classification_weight == 0.0:
            self.criterion.classification_weight = self._base_classification_loss_weight
            if not self._warned_missing_generation_targets:
                logger.warning(
                    "No generation targets in batch; restored classification_weight=%.3f to avoid zero-loss collapse.",
                    self.criterion.classification_weight,
                )
                self._warned_missing_generation_targets = True

        # Build forward kwargs
        forward_kwargs: Dict[str, Any] = {
            "pixel_values": images,
            "graph_x": self._tensor_to_device(batch.get("graph_x")),
            "graph_edge_index": self._tensor_to_device(batch.get("graph_edge_index")),
            "graph_edge_type": self._tensor_to_device(batch.get("graph_edge_type")),
            "graph_batch": self._tensor_to_device(batch.get("graph_batch")),
            "target_ids": target_ids,
            "teacher_forcing_ratio": teacher_forcing_ratio,
        }

        cls_w = float(self.criterion.classification_weight)
        gen_w = float(self.criterion.generation_weight)

        with _autocast_ctx(enabled=self.tc.mixed_precision, device=self.device):
            outputs = self.model(**forward_kwargs)

            losses = self.criterion(
                classification_logits=outputs.get("classification_logits"),
                classification_targets=labels,
                generation_logits=outputs.get("generation_logits"),
                generation_targets=generation_targets,
            )
            loss = losses["total"] / self.tc.accumulation_steps

            cls_loss_item = float(losses["classification"].detach().item())
            gen_loss_item = float(losses["generation"].detach().item())

        # Backward
        self.scaler.scale(loss).backward()

        # Accumulation step
        if (batch_idx + 1) % self.tc.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tc.gradient_clip_norm
            )
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # With AMP overflow, optimizer.step() can be skipped.
            # Avoid stepping LR scheduler in that case.
            scale_after = self.scaler.get_scale()
            if scale_after >= scale_before:
                self.scheduler.step()
            self.global_step += 1

        total_item = float(loss.item() * self.tc.accumulation_steps)
        return {
            "total": total_item,
            "classification": cls_loss_item,
            "generation": gen_loss_item,
            "weighted_classification": cls_w * cls_loss_item,
            "weighted_generation": gen_w * gen_loss_item,
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        total_cls = 0.0
        total_gen = 0.0
        total_w_cls = 0.0
        total_w_gen = 0.0
        total_samples = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["labels"].to(self.device)
            target_ids = self._tensor_to_device(batch.get("target_ids"))
            generation_targets = self._tensor_to_device(batch.get("generation_targets"))
            target_ids, generation_targets = self._apply_report_prompt_prefix(target_ids, generation_targets)

            forward_kwargs: Dict[str, Any] = {
                "pixel_values": images,
                "graph_x": self._tensor_to_device(batch.get("graph_x")),
                "graph_edge_index": self._tensor_to_device(batch.get("graph_edge_index")),
                "graph_edge_type": self._tensor_to_device(batch.get("graph_edge_type")),
                "graph_batch": self._tensor_to_device(batch.get("graph_batch")),
                "target_ids": target_ids,
            }

            outputs = self.model(**forward_kwargs)
            losses = self.criterion(
                classification_logits=outputs.get("classification_logits"),
                classification_targets=labels,
                generation_logits=outputs.get("generation_logits"),
                generation_targets=generation_targets,
            )
            batch_size = images.shape[0]
            total_loss += losses["total"].item() * batch_size
            cls_item = float(losses["classification"].item())
            gen_item = float(losses["generation"].item())
            total_cls += cls_item * batch_size
            total_gen += gen_item * batch_size
            total_w_cls += float(self.criterion.classification_weight) * cls_item * batch_size
            total_w_gen += float(self.criterion.generation_weight) * gen_item * batch_size
            total_samples += batch_size

        self.model.train()
        denom = max(total_samples, 1)
        return {
            "total": total_loss / denom,
            "classification": total_cls / denom,
            "generation": total_gen / denom,
            "weighted_classification": total_w_cls / denom,
            "weighted_generation": total_w_gen / denom,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> bool:
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

        saved_any = False

        path = ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt"
        if _safe_torch_save(checkpoint, path):
            logger.info("Saved checkpoint: %s", path)
            saved_any = True

        if is_best:
            best_path = ckpt_dir / "best_model.pt"
            if _safe_torch_save(checkpoint, best_path):
                logger.info("Saved best model: %s", best_path)
                saved_any = True
                self.last_best_checkpoint_path = best_path

        # Cleanup old checkpoints
        self._cleanup_checkpoints(ckpt_dir)
        return saved_any

    def _cleanup_checkpoints(self, ckpt_dir: Path) -> None:
        """Keep only the last N checkpoints + best."""
        keep = self.tc.keep_last_n_checkpoints
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
        if len(ckpts) > keep:
            for old in ckpts[: len(ckpts) - keep]:
                try:
                    old.unlink()
                    logger.debug("Removed old checkpoint: %s", old)
                except FileNotFoundError:
                    # Another process may have already removed this file.
                    logger.debug("Checkpoint already removed by another worker: %s", old)

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

    def _apply_two_stage_schedule(self, epoch: int) -> None:
        """Apply stage-wise parameter freezing/unfreezing.

        Stage 1 (epoch < stage1_epochs):
            - Freeze visual encoder and classification head (configurable)
            - Disable classification loss so only KG/fusion/decoder paths learn
        Stage 2:
            - Unfreeze all modules and restore full classification loss
        """
        if not self.tc.two_stage_training or self.tc.stage1_epochs <= 0:
            return

        in_stage1 = epoch < self.tc.stage1_epochs
        target_mode = "stage1" if in_stage1 else "stage2"
        if self._stage_mode == target_mode:
            return

        if in_stage1:
            if self.tc.stage1_freeze_visual_encoder:
                self._set_module_trainable("visual_encoder", False)
            if self.tc.stage1_freeze_classification_head:
                self._set_module_trainable("classification_head", False)

            self.criterion.classification_weight = min(
                self._base_classification_loss_weight,
                max(0.0, float(self.tc.stage1_classification_weight)),
            )
            self._stage_mode = "stage1"
            logger.info(
                "Two-stage training: entering stage 1 at epoch %d (classification_weight=%.3f, selective freezing enabled).",
                epoch + 1,
                self.criterion.classification_weight,
            )
            return

        # Stage 2: unfreeze and restore full joint objective.
        self._set_module_trainable("visual_encoder", True)
        self._set_module_trainable("classification_head", True)
        self.criterion.classification_weight = self._base_classification_loss_weight
        # Reset best metric when entering stage 2 so checkpoint selection reflects joint training.
        if self.tc.reset_best_metric_on_stage2:
            self.best_val_metric = float("inf")
            self.patience_counter = 0
        self._stage_mode = "stage2"
        logger.info(
            "Two-stage training: entering stage 2 at epoch %d (all modules trainable, classification_weight=%.3f; reset_best_metric=%s).",
            epoch + 1,
            self.criterion.classification_weight,
            self.tc.reset_best_metric_on_stage2,
        )

    def _set_module_trainable(self, module_name: str, trainable: bool) -> None:
        module = getattr(self.model, module_name, None)
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = trainable

    def _tensor_to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return value

    def _teacher_forcing_ratio_for_epoch(self, epoch: int) -> float:
        rg = self.config.model.report_generation
        if not rg.scheduled_sampling_enabled:
            return 1.0

        start = float(rg.scheduled_sampling_start_ratio)
        end = float(rg.scheduled_sampling_end_ratio)
        decay_epochs = max(1, int(rg.scheduled_sampling_decay_epochs))
        progress = min(max(epoch, 0), decay_epochs) / float(decay_epochs)
        ratio = start + (end - start) * progress
        return max(0.0, min(1.0, ratio))

    def _selection_metric_from_val(self, val_stats: Dict[str, float]) -> float:
        if self.tc.checkpoint_selection_mode == "classification_priority":
            # In stage 1 the classification head may be frozen, so fall back
            # to weighted generation loss to avoid stale metrics.
            if self._stage_mode == "stage1" and self.tc.stage1_freeze_classification_head:
                return float(val_stats.get("weighted_generation", val_stats["total"]))
            return float(val_stats.get("weighted_classification", val_stats.get("classification", val_stats["total"])))
        return float(val_stats["total"])

    def _passes_generation_guard(self, val_stats: Dict[str, float]) -> bool:
        max_gen = self.tc.checkpoint_generation_guard_max_val_gen_loss
        if max_gen is None:
            return True
        # Compare *weighted* generation loss (what actually contributes to
        # the total objective) rather than the raw unweighted value.
        weighted_gen = float(val_stats.get("weighted_generation", 0.0))
        return weighted_gen <= float(max_gen)

    def _write_run_manifest(self) -> None:
        if self._run_manifest_written:
            return

        try:
            log_dir = Path(self.tc.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            manifest_path = log_dir / f"run_manifest_{timestamp}.json"
            payload = {
                "timestamp": datetime.now().isoformat(),
                "git_hash": get_git_hash(),
                "device": str(self.device),
                "config": self.config.to_dict(),
            }
            manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            logger.info("Wrote run manifest: %s", manifest_path)
            self._run_manifest_written = True
        except Exception as exc:
            logger.warning("Failed to write run manifest: %s", exc)

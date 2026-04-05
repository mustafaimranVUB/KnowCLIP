"""Multi-task loss functions for classification + report generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Weighted combination of classification and generation losses.

    Parameters:
        classification_weight: Weight for the classification loss.
        generation_weight: Weight for the generation (language model) loss.
        label_smoothing: Label smoothing for cross-entropy (generation).
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        generation_weight: float = 1.0,
        label_smoothing: float = 0.0,
        classification_loss_type: str = "bce",
        class_pos_weight: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = None,
    ) -> None:
        super().__init__()
        self.classification_weight = classification_weight
        self.generation_weight = generation_weight
        self.classification_loss_type = classification_loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # BCE with logits for multi-label classification
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self._class_pos_weight = class_pos_weight.clone().detach() if class_pos_weight is not None else None

        # Cross-entropy for auto-regressive generation
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
            reduction="mean",
        )

    def forward(
        self,
        classification_logits: torch.Tensor | None = None,
        classification_targets: torch.Tensor | None = None,
        generation_logits: torch.Tensor | None = None,
        generation_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            classification_logits: ``(B, C)`` raw logits.
            classification_targets: ``(B, C)`` float targets in [0, 1].
            generation_logits: ``(B, L, V)`` next-token logits.
            generation_targets: ``(B, L)`` target token IDs.

        Returns:
            Dict with keys: ``total``, ``classification``, ``generation``.
        """
        losses: dict[str, torch.Tensor] = {}
        device = None

        # Classification loss
        if classification_logits is not None and classification_targets is not None:
            device = classification_logits.device
            # Mask missing/uncertain labels; BCE expects binary targets in [0, 1].
            valid_mask = (
                ~torch.isnan(classification_targets)
                & (classification_targets >= 0.0)
                & (classification_targets <= 1.0)
            )
            if valid_mask.any():
                safe_targets = torch.where(
                    valid_mask,
                    classification_targets,
                    torch.zeros_like(classification_targets),
                )
                pos_weight = None
                if self._class_pos_weight is not None:
                    pos_weight = self._class_pos_weight.to(device=device, dtype=classification_logits.dtype)

                base_loss = F.binary_cross_entropy_with_logits(
                    classification_logits,
                    safe_targets,
                    reduction="none",
                    pos_weight=pos_weight,
                )

                if self.classification_loss_type == "focal":
                    p = torch.sigmoid(classification_logits)
                    p_t = torch.where(safe_targets > 0.5, p, 1.0 - p)
                    focal_factor = (1.0 - p_t).pow(self.focal_gamma)

                    if self.focal_alpha is not None:
                        alpha_t = torch.where(
                            safe_targets > 0.5,
                            torch.full_like(safe_targets, float(self.focal_alpha)),
                            torch.full_like(safe_targets, 1.0 - float(self.focal_alpha)),
                        )
                        base_loss = base_loss * alpha_t

                    per_elem = base_loss * focal_factor
                else:
                    per_elem = base_loss

                cls_loss = (per_elem * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
            else:
                cls_loss = torch.tensor(0.0, device=device)
            losses["classification"] = cls_loss
        else:
            losses["classification"] = torch.tensor(0.0)

        # Generation loss (cross-entropy on shifted targets)
        if generation_logits is not None and generation_targets is not None:
            device = generation_logits.device
            B, L, V = generation_logits.shape
            gen_loss = self.ce_loss(
                generation_logits.reshape(-1, V),
                generation_targets.reshape(-1),
            )
            losses["generation"] = gen_loss
        else:
            losses["generation"] = torch.tensor(0.0)

        # Weighted total
        if device is None:
            device = losses["classification"].device

        total = (
            self.classification_weight * losses["classification"].to(device)
            + self.generation_weight * losses["generation"].to(device)
        )
        losses["total"] = total

        return losses

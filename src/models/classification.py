"""Multi-label classification head for CheXpert-14 pathology detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.config import ClassificationHeadConfig


class ClassificationHead(nn.Module):
    """Multi-label classification head.

    Takes a pooled feature vector ``(B, D_in)`` and outputs logits for
    ``num_classes`` independent binary classifiers.

    Parameters:
        config: Classification head configuration.
        input_dim: Dimension of the input features.
    """

    def __init__(
        self,
        config: ClassificationHeadConfig,
        input_dim: int = 768,
    ) -> None:
        super().__init__()
        self.config = config

        self.head = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim) if config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute classification logits.

        Args:
            features: ``(B, D)`` pooled features.

        Returns:
            ``(B, num_classes)`` raw logits (apply sigmoid for probabilities).
        """
        return self.head(features)

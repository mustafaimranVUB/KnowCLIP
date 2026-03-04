"""CLIP-family visual encoder (E_V) implementation.

Supports BioMedCLIP, PubMedCLIP, and generic CLIP ViT backbones.
Outputs **patch** embeddings (CLS token excluded) with an optional
linear projection to a unified output dimension.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from src.core.config import VisualEncoderConfig
from src.models.interfaces import BaseVisualEncoder

logger = logging.getLogger(__name__)


class CLIPVisualEncoder(BaseVisualEncoder):
    """CLIP-based visual encoder with swappable backbone.

    Parameters:
        config: Visual encoder configuration.
    """

    def __init__(self, config: VisualEncoderConfig) -> None:
        # Explicit nn.Module init (ABC+nn.Module MRO)
        nn.Module.__init__(self)
        self.config = config

        # Backbone
        self.backbone = self._load_backbone(config)

        # Projection if backbone hidden_dim != output_dim
        if config.hidden_dim != config.output_dim:
            self.projection = nn.Linear(config.hidden_dim, config.output_dim)
        else:
            self.projection = nn.Identity()

        # Optionally freeze backbone
        if config.freeze_backbone:
            self._freeze_backbone()
        elif config.freeze_layers is not None:
            self._freeze_layers(config.freeze_layers)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to patch embeddings.

        Args:
            pixel_values: ``(B, 3, H, W)`` normalised images.

        Returns:
            ``(B, P, D)`` patch embeddings.
        """
        outputs = self.backbone(pixel_values=pixel_values)

        # ``outputs.last_hidden_state`` contains [CLS, patch_1, ..., patch_P]
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            hidden = outputs["last_hidden_state"]
        else:
            # Some models return a tuple
            hidden = outputs[0]

        # Exclude CLS token (index 0)
        patch_embeddings = hidden[:, 1:, :]  # (B, P, D_hidden)

        # Project to output dim
        patch_embeddings = self.projection(patch_embeddings)

        return patch_embeddings

    def get_output_dim(self) -> int:
        return self.config.output_dim

    def get_num_patches(self) -> int:
        return self.config.num_patches

    # ------------------------------------------------------------------
    # Backbone loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_backbone(config: VisualEncoderConfig) -> nn.Module:
        """Load the CLIP vision backbone from HuggingFace."""
        from transformers import CLIPVisionModel, AutoModel  # type: ignore

        checkpoint = config.checkpoint
        logger.info("Loading visual backbone: %s", checkpoint)

        try:
            # Try CLIPVisionModel first (standard CLIP)
            model = CLIPVisionModel.from_pretrained(checkpoint)
        except (OSError, ValueError):
            # Fall back to AutoModel (e.g. BioMedCLIP uses open_clip)
            try:
                full_model = AutoModel.from_pretrained(
                    checkpoint, trust_remote_code=True
                )
                # Extract vision tower if available
                if hasattr(full_model, "vision_model"):
                    model = full_model.vision_model
                elif hasattr(full_model, "visual"):
                    model = full_model.visual
                else:
                    model = full_model
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load visual backbone '{checkpoint}': {exc}"
                ) from exc

        return model

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Froze all backbone parameters")

    def _freeze_layers(self, layer_indices: List[int]) -> None:
        """Freeze specific encoder layers."""
        if hasattr(self.backbone, "encoder") and hasattr(
            self.backbone.encoder, "layers"
        ):
            layers = self.backbone.encoder.layers
            for idx in layer_indices:
                if 0 <= idx < len(layers):
                    for param in layers[idx].parameters():
                        param.requires_grad = False
            logger.info("Froze layers: %s", layer_indices)
        else:
            logger.warning("Cannot freeze specific layers: encoder.layers not found")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Unfroze all backbone parameters")

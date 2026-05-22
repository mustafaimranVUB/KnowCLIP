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


class _OpenCLIPVisionWrapper(nn.Module):
    """Thin adapter that exposes OpenCLIP's ``visual`` tower with a
    HuggingFace-compatible ``last_hidden_state`` output.

    Handles two OpenCLIP visual architectures:

    * **TimmModel** (BioMedCLIP) — wraps a ``timm`` ViT; patch tokens are
      extracted via ``trunk.forward_features()``.
    * **VisionTransformer** (standard OpenCLIP) — uses ``conv1``,
      ``class_embedding``, ``positional_embedding``, ``ln_pre``,
      ``transformer``, ``ln_post``.
    """

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual
        # TimmModel has a `.trunk` attribute; standard VisionTransformer has `.conv1`
        self._is_timm = hasattr(visual, "trunk")

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Return a namespace with ``last_hidden_state`` (B, P+1, D)."""
        if self._is_timm:
            x = self._forward_timm(pixel_values)
        else:
            x = self._forward_vit(pixel_values)

        class _Output:
            def __init__(self, lhs: torch.Tensor):
                self.last_hidden_state = lhs
        return _Output(x)

    # -- TimmModel path (BioMedCLIP) ----------------------------------

    def _forward_timm(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract [CLS + patch] tokens from a timm-backed OpenCLIP model."""
        trunk = self.visual.trunk
        # timm VisionTransformer.forward_features returns (B, P+1, D)
        x = trunk.forward_features(pixel_values)
        return x

    # -- Standard OpenCLIP VisionTransformer path ----------------------

    def _forward_vit(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract [CLS + patch] tokens from a standard OpenCLIP ViT."""
        v = self.visual
        x = v.conv1(pixel_values)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, P, D)

        cls_token = v.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)

        x = v.patch_dropout(x) if hasattr(v, "patch_dropout") else x
        x = v.ln_pre(x)

        x = v.transformer(x)

        x = v.ln_post(x)
        return x


def _torch_supports_safe_pickle_load() -> bool:
    """Return True if torch version is new enough for transformers pickle guard."""
    v = torch.__version__.split("+")[0]
    parts = v.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return False
    return (major, minor) >= (2, 6)


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

        # Keep patch tokens regardless of whether CLS is present.
        if hidden.dim() != 3:
            raise RuntimeError(f"Unexpected visual hidden shape: {tuple(hidden.shape)}")

        if hidden.shape[1] == self.config.num_patches + 1:
            patch_embeddings = hidden[:, 1:, :]  # [CLS] + patches
        elif hidden.shape[1] == self.config.num_patches:
            patch_embeddings = hidden  # patches only
        else:
            logger.debug(
                "Unexpected token count %d for %s; using all tokens as patch embeddings.",
                hidden.shape[1],
                self.config.backbone_type,
            )
            patch_embeddings = hidden

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
        """Load the CLIP vision backbone from HuggingFace or OpenCLIP.

        BioMedCLIP is distributed in OpenCLIP format and requires the
        ``open_clip_torch`` package.  We try that first (native format),
        then fall back to HuggingFace loaders, then to geometry-compatible
        CLIP checkpoints.

        **Important**: a fallback to ``openai/clip-vit-base-patch16`` means
        the model is NOT using biomedically fine-tuned weights.  This is
        logged as a WARNING so it cannot be silently missed.
        """
        from transformers import AutoModel, CLIPModel, CLIPVisionModel  # type: ignore

        errors: List[str] = []

        # --- OpenCLIP path (preferred for BioMedCLIP / PubMedCLIP) ---
        if config.backbone_type in ("biomedclip", "pubmedclip"):
            try:
                import open_clip  # type: ignore

                if config.backbone_type == "biomedclip":
                    model, _, _ = open_clip.create_model_and_transforms(
                        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                    )
                else:
                    model, _, _ = open_clip.create_model_and_transforms(
                        "hf-hub:flaviagiammarino/pubmed-clip-vit-base-patch32",
                    )
                # open_clip wraps the vision tower in model.visual
                if hasattr(model, "visual"):
                    logger.info(
                        "Loaded %s via open_clip (native format).",
                        config.backbone_type,
                    )
                    return _OpenCLIPVisionWrapper(model.visual)
                errors.append(f"open_clip[{config.backbone_type}]: model.visual not found")
            except ImportError:
                errors.append(
                    f"open_clip not installed — cannot load {config.backbone_type} natively. "
                    "Install with: pip install open-clip-torch"
                )
            except Exception as exc:
                errors.append(f"open_clip[{config.backbone_type}]: {exc}")

        # --- HuggingFace paths ---
        checkpoint_candidates = [config.checkpoint]
        if config.backbone_type == "biomedclip":
            checkpoint_candidates.extend(
                [
                    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
                    "openai/clip-vit-base-patch16",
                ]
            )
        elif config.backbone_type == "pubmedclip":
            checkpoint_candidates.extend(
                [
                    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
                    "openai/clip-vit-base-patch32",
                    "openai/clip-vit-base-patch16",
                ]
            )

        for checkpoint in checkpoint_candidates:
            logger.info("Loading visual backbone: %s", checkpoint)

            # Track whether this is a fallback (not the primary checkpoint)
            is_fallback = checkpoint != config.checkpoint

            try:
                vision_model = CLIPVisionModel.from_pretrained(checkpoint, use_safetensors=True)
                if is_fallback:
                    logger.warning(
                        "FALLBACK: Loaded '%s' instead of '%s'. "
                        "The model may NOT have biomedical fine-tuning. "
                        "Install open-clip-torch for native BioMedCLIP support.",
                        checkpoint,
                        config.checkpoint,
                    )
                return vision_model
            except Exception as exc:
                errors.append(f"CLIPVisionModel[{checkpoint}]: {exc}")

            try:
                clip_model = CLIPModel.from_pretrained(checkpoint, use_safetensors=True)
                if is_fallback:
                    logger.warning(
                        "FALLBACK: Loaded '%s' instead of '%s'. "
                        "The model may NOT have biomedical fine-tuning.",
                        checkpoint,
                        config.checkpoint,
                    )
                return clip_model.vision_model
            except Exception as exc:
                errors.append(f"CLIPModel[{checkpoint}]: {exc}")

            try:
                full_model = AutoModel.from_pretrained(
                    checkpoint,
                    trust_remote_code=True,
                    use_safetensors=True,
                )
                if hasattr(full_model, "vision_model"):
                    if is_fallback:
                        logger.warning(
                            "FALLBACK: Loaded '%s' instead of '%s'.",
                            checkpoint,
                            config.checkpoint,
                        )
                    return full_model.vision_model
                if hasattr(full_model, "visual"):
                    return full_model.visual
                return full_model
            except Exception as exc:
                errors.append(f"AutoModel[{checkpoint}]: {exc}")

            # Last resort for older repositories that only ship pickle weights.
            if _torch_supports_safe_pickle_load():
                try:
                    return CLIPVisionModel.from_pretrained(checkpoint, use_safetensors=False)
                except Exception as exc:
                    errors.append(f"CLIPVisionModel[pickle:{checkpoint}]: {exc}")

        details = "\n".join(f"  - {e}" for e in errors[-6:])
        raise RuntimeError(
            "Failed to load visual backbone. Tried multiple loaders/checkpoints.\n"
            f"Requested backbone_type='{config.backbone_type}', checkpoint='{config.checkpoint}'.\n"
            "Recent errors:\n"
            f"{details}\n"
            "If you need BioMedCLIP specifically, install open-clip support in the"
            " existing environment: `pip install open-clip-torch` and retry."
        )

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

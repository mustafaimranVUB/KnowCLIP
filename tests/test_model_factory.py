"""Tests for model construction and forward passes.

Uses a mock visual encoder to avoid downloading BioMedCLIP from HuggingFace.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.core.config import ModelConfig, VisualEncoderConfig, get_baseline_config, get_neurosymbolic_config
from src.models.interfaces import BaseVisualEncoder
from src.models.model_factory import MedicalVLM, build_model


# ---------------------------------------------------------------------------
# Mock visual encoder (no HF download required)
# ---------------------------------------------------------------------------

class MockVisualEncoder(BaseVisualEncoder, nn.Module):
    """Lightweight visual encoder for testing (no HuggingFace download)."""

    def __init__(self, config: VisualEncoderConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        # Small conv to produce patch embeddings without huge memory
        self._conv = nn.Conv2d(3, config.output_dim, kernel_size=16, stride=16)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]
        # Conv: (B, 3, 224, 224) → (B, D, 14, 14) → (B, D, 196) → (B, 196, D)
        h = self._conv(pixel_values)  # (B, D, H', W')
        h = h.flatten(2).transpose(1, 2)  # (B, P, D)
        return h

    def get_output_dim(self) -> int:
        return self.config.output_dim

    def get_num_patches(self) -> int:
        return self.config.num_patches


@pytest.fixture(autouse=True)
def _patch_visual_encoder():
    """Patch CLIPVisualEncoder with MockVisualEncoder for all tests in this module."""
    with patch("src.models.visual_encoder.CLIPVisualEncoder", MockVisualEncoder):
        yield


class TestBuildModel:
    def test_build_baseline(self, baseline_config):
        model = build_model(baseline_config)
        assert isinstance(model, MedicalVLM)
        assert model.knowledge_encoder is None
        assert model.fusion_module is None
        assert model.classification_head is not None
        assert model.decoder is None

    def test_build_neurosymbolic(self, neurosymbolic_config):
        model = build_model(neurosymbolic_config)
        assert isinstance(model, MedicalVLM)
        assert model.knowledge_encoder is not None
        assert model.fusion_module is not None
        assert model.classification_head is not None
        assert model.decoder is not None

    def test_param_count_positive(self, baseline_config):
        model = build_model(baseline_config)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_neurosymbolic_more_params(self, baseline_config, neurosymbolic_config):
        base = build_model(baseline_config)
        ns = build_model(neurosymbolic_config)
        base_params = sum(p.numel() for p in base.parameters())
        ns_params = sum(p.numel() for p in ns.parameters())
        assert ns_params > base_params


class TestBaselineForward:
    def test_forward_shape(self, baseline_config, dummy_images):
        model = build_model(baseline_config)
        model.eval()
        with torch.no_grad():
            outputs = model(pixel_values=dummy_images)

        assert "classification_logits" in outputs
        assert outputs["classification_logits"].shape == (2, 14)

    def test_no_graph_keys(self, baseline_config, dummy_images):
        model = build_model(baseline_config)
        model.eval()
        with torch.no_grad():
            outputs = model(pixel_values=dummy_images)

        # Fused features should be None for baseline
        assert outputs.get("fused_features") is None

    def test_visual_features_present(self, baseline_config, dummy_images):
        model = build_model(baseline_config)
        model.eval()
        with torch.no_grad():
            outputs = model(pixel_values=dummy_images)

        assert "visual_features" in outputs
        Z_v = outputs["visual_features"]
        assert Z_v.shape[0] == 2  # batch
        assert Z_v.shape[2] == 768  # hidden dim


class TestNeurosymbolicForward:
    def test_forward_with_graph(self, neurosymbolic_config, dummy_images, dummy_graph):
        model = build_model(neurosymbolic_config)
        model.eval()
        with torch.no_grad():
            outputs = model(
                pixel_values=dummy_images,
                **dummy_graph,
            )

        assert "classification_logits" in outputs
        assert outputs["classification_logits"].shape == (2, 14)

    def test_fused_features_present(self, neurosymbolic_config, dummy_images, dummy_graph):
        model = build_model(neurosymbolic_config)
        model.eval()
        with torch.no_grad():
            outputs = model(
                pixel_values=dummy_images,
                **dummy_graph,
            )

        assert outputs.get("fused_features") is not None
        Z_fused = outputs["fused_features"]
        assert Z_fused.shape[0] == 2  # batch

    def test_generation_logits_with_targets(self, neurosymbolic_config, dummy_images, dummy_graph):
        model = build_model(neurosymbolic_config)
        model.eval()
        target_ids = torch.randint(0, 50257, (2, 16))
        with torch.no_grad():
            outputs = model(
                pixel_values=dummy_images,
                target_ids=target_ids,
                **dummy_graph,
            )

        assert "generation_logits" in outputs
        gen = outputs["generation_logits"]
        assert gen.shape[0] == 2
        assert gen.shape[1] == 16
        assert gen.shape[2] == 50257

    def test_attention_weights(self, neurosymbolic_config, dummy_images, dummy_graph):
        model = build_model(neurosymbolic_config)
        model.eval()
        with torch.no_grad():
            outputs = model(
                pixel_values=dummy_images,
                return_attention=True,
                **dummy_graph,
            )

        assert "attention_weights" in outputs
        assert "explainability" in outputs
        explainability = outputs["explainability"]
        assert "graph" in explainability
        assert "fusion" in explainability
        assert "pooling" in explainability
        assert len(explainability["graph"]["edge_attention_layers"]) == neurosymbolic_config.knowledge_encoder.num_gat_layers
        assert len(explainability["fusion"]["per_layer"]) == neurosymbolic_config.fusion_module.num_fusion_layers
        assert explainability["pooling"]["weights"].shape[0] == 2


class TestGradientFlow:
    def test_baseline_gradients(self, baseline_config, dummy_images, dummy_labels):
        model = build_model(baseline_config)
        model.train()

        outputs = model(pixel_values=dummy_images)
        logits = outputs["classification_logits"]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, dummy_labels)
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients flowing through baseline model"

    def test_neurosymbolic_gradients(self, neurosymbolic_config, dummy_images, dummy_graph, dummy_labels):
        model = build_model(neurosymbolic_config)
        model.train()

        outputs = model(pixel_values=dummy_images, **dummy_graph)
        logits = outputs["classification_logits"]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, dummy_labels)
        loss.backward()

        # Check knowledge encoder has gradients
        ke_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.knowledge_encoder.parameters()
            if p.requires_grad
        )
        assert ke_has_grad, "No gradients flowing through knowledge encoder"

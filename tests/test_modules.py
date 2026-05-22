"""Tests for individual model components.

Visual encoder tests are marked as requiring network access (HuggingFace).
All other component tests run offline.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.core.config import (
    ClassificationHeadConfig,
    FusionModuleConfig,
    KnowledgeEncoderConfig,
    ReportGenerationConfig,
    VisualEncoderConfig,
)

requires_network = pytest.mark.skipif(
    True,
    reason="Requires HuggingFace model download (run with --run-network to enable)",
)


class TestVisualEncoder:
    @requires_network
    def test_forward_shape(self):
        from src.models.visual_encoder import CLIPVisualEncoder

        cfg = VisualEncoderConfig(backbone_type="biomedclip")
        encoder = CLIPVisualEncoder(cfg)
        encoder.eval()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = encoder(dummy)

        assert out.shape == (2, cfg.num_patches, cfg.output_dim)

    @requires_network
    def test_output_dim(self):
        from src.models.visual_encoder import CLIPVisualEncoder

        cfg = VisualEncoderConfig(backbone_type="biomedclip")
        encoder = CLIPVisualEncoder(cfg)
        assert encoder.get_output_dim() == 768

    @requires_network
    def test_num_patches(self):
        from src.models.visual_encoder import CLIPVisualEncoder

        cfg = VisualEncoderConfig(backbone_type="biomedclip")
        encoder = CLIPVisualEncoder(cfg)
        assert encoder.get_num_patches() == 196


class TestKnowledgeEncoder:
    def test_forward_shape(self):
        from src.models.knowledge_encoder import GATv2KnowledgeEncoder

        cfg = KnowledgeEncoderConfig()
        encoder = GATv2KnowledgeEncoder(cfg)
        encoder.eval()

        x = torch.randn(6, 768)
        edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])
        edge_type = torch.tensor([0, 1, 2, 3])
        batch = torch.tensor([0, 0, 0, 1, 1, 1])

        with torch.no_grad():
            out = encoder(x, edge_index, edge_type, batch)

        assert out.shape == (6, 768)

    def test_output_normalized(self):
        from src.models.knowledge_encoder import GATv2KnowledgeEncoder

        cfg = KnowledgeEncoderConfig(normalize_outputs=True)
        encoder = GATv2KnowledgeEncoder(cfg)
        encoder.eval()

        x = torch.randn(4, 768)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_type = torch.tensor([0, 1])
        batch = torch.tensor([0, 0, 0, 0])

        with torch.no_grad():
            out = encoder(x, edge_index, edge_type, batch)

        # Check L2 norm ≈ 1
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)

    def test_returns_attention_trace(self):
        from src.models.knowledge_encoder import GATv2KnowledgeEncoder

        cfg = KnowledgeEncoderConfig()
        encoder = GATv2KnowledgeEncoder(cfg)
        encoder.eval()

        x = torch.randn(4, 768)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_type = torch.tensor([0, 1])

        with torch.no_grad():
            out, trace = encoder(x, edge_index, edge_type, return_attention=True)

        assert out.shape == (4, 768)
        assert len(trace["edge_attention_layers"]) == cfg.num_gat_layers
        assert "attention_vector" in trace["edge_attention_layers"][0]


class TestFusionModule:
    def test_cross_attention_shape(self):
        from src.models.fusion import CrossAttentionFusion

        cfg = FusionModuleConfig(num_fusion_layers=2, num_heads=8, hidden_dim=768)
        fusion = CrossAttentionFusion(cfg)
        fusion.eval()

        Z_k = torch.randn(2, 3, 768)  # knowledge
        Z_v = torch.randn(2, 196, 768)  # visual

        with torch.no_grad():
            out = fusion(Z_k, Z_v)

        assert out.shape == (2, 3, 768)

    def test_attention_weights_returned(self):
        from src.models.fusion import CrossAttentionFusion

        cfg = FusionModuleConfig()
        fusion = CrossAttentionFusion(cfg)
        fusion.eval()

        Z_k = torch.randn(2, 3, 768)
        Z_v = torch.randn(2, 196, 768)

        with torch.no_grad():
            out, attn = fusion(Z_k, Z_v, return_attention=True)

        assert attn is not None
        assert len(attn["per_layer"]) == cfg.num_fusion_layers
        assert attn["last_layer"].shape == (2, cfg.num_heads, 3, 196)


class TestClassificationHead:
    def test_forward_shape(self):
        from src.models.classification import ClassificationHead

        cfg = ClassificationHeadConfig(num_classes=14, hidden_dim=512)
        head = ClassificationHead(cfg, input_dim=768)
        head.eval()

        x = torch.randn(4, 768)
        with torch.no_grad():
            out = head(x)

        assert out.shape == (4, 14)


class TestReportDecoder:
    def test_forward_shape(self):
        from src.models.decoder import TransformerReportDecoder

        cfg = ReportGenerationConfig(
            vocab_size=1000,
            max_report_length=32,
            num_decoder_layers=2,
            decoder_dim=768,
            num_decoder_heads=8,
            decoder_ffn_dim=2048,
        )
        decoder = TransformerReportDecoder(cfg)
        decoder.eval()

        encoder_output = torch.randn(2, 10, 768)
        target_ids = torch.randint(0, 1000, (2, 16))

        with torch.no_grad():
            out = decoder(encoder_output, target_ids)

        assert out.shape == (2, 16, 1000)

    def test_get_vocab_size(self):
        from src.models.decoder import TransformerReportDecoder

        cfg = ReportGenerationConfig(vocab_size=50257)
        decoder = TransformerReportDecoder(cfg)
        assert decoder.get_vocab_size() == 50257

    def test_transformer_return_attention_is_backward_compatible(self):
        from src.models.decoder import TransformerReportDecoder

        cfg = ReportGenerationConfig(
            vocab_size=1000,
            max_report_length=32,
            num_decoder_layers=2,
            decoder_dim=768,
            num_decoder_heads=8,
            decoder_ffn_dim=2048,
        )
        decoder = TransformerReportDecoder(cfg)
        decoder.eval()

        encoder_output = torch.randn(2, 10, 768)
        target_ids = torch.randint(0, 1000, (2, 16))

        with torch.no_grad():
            logits, trace = decoder(encoder_output, target_ids, return_attention=True)

        assert logits.shape == (2, 16, 1000)
        assert trace is None


class TestSelfAttentionPooling:
    def test_output_shape(self):
        from src.models.fusion import SelfAttentionPooling

        pool = SelfAttentionPooling(hidden_dim=768)
        pool.eval()

        x = torch.randn(2, 196, 768)
        with torch.no_grad():
            out = pool(x)

        assert out.shape == (2, 768)

    def test_masked_attention_weights_zero_out_padding(self):
        from src.models.fusion import SelfAttentionPooling

        pool = SelfAttentionPooling(hidden_dim=768)
        pool.eval()

        x = torch.randn(2, 4, 768)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

        with torch.no_grad():
            out, weights = pool(x, mask=mask, return_attention=True)

        assert out.shape == (2, 768)
        assert weights.shape == (2, 4)
        assert torch.allclose(weights[0, 2:], torch.zeros(2), atol=1e-6)
        assert torch.allclose(weights[1, 1:], torch.zeros(3), atol=1e-6)

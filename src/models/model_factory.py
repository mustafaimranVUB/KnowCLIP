"""
Model Factory for Phase II: Medical Vision-Language Model.

This module provides the main MedicalVLM wrapper class that orchestrates
the entire architecture and allows easy comparison between:
- Neuro-Symbolic (with KG) vs. Pure Vision (without KG)
- Different visual backbones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Optional, Dict, Tuple, Any

from .config import ModelConfig
from .modules import VisualEncoder, KnowledgeEncoder, FusionModule, SelfAttentionPooling


class MedicalVLM(nn.Module):
    """
    Medical Vision-Language Model with optional Knowledge Graph integration.

    Architecture modes:
    1. Neuro-Symbolic (use_kg=True):
       Image → VisualEncoder → FusionModule ← KnowledgeEncoder ← KG
                                    ↓
                            [Classification Head]
                            [Report Generator]

    2. Pure Vision Baseline (use_kg=False):
       Image → VisualEncoder → Pooling → Linear Probe
                                    ↓
                            [Classification Head]

    Args:
        config: ModelConfig with all hyperparameters and architecture choices
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Initialize Visual Encoder (always present)
        self.visual_encoder = VisualEncoder(config.visual_encoder)

        # Initialize Knowledge components (only if use_kg=True)
        if config.use_kg:
            self.knowledge_encoder = KnowledgeEncoder(config.knowledge_encoder)
            self.fusion_module = FusionModule(config.fusion_module)
        else:
            self.knowledge_encoder = None
            self.fusion_module = None

            # For baseline: simple pooling instead of fusion
            self.visual_pooling = SelfAttentionPooling(config.visual_encoder.hidden_dim)

        # Task-specific heads
        if config.enable_classification:
            self.classification_head = ClassificationHead(
                input_dim=config.visual_encoder.hidden_dim,
                config=config.classification_head,
            )
        else:
            self.classification_head = None

        if config.enable_report_generation:
            self.report_generator = ReportGenerator(
                encoder_dim=config.visual_encoder.hidden_dim,
                config=config.report_generation,
            )
        else:
            self.report_generator = None

    def forward(
        self,
        images: torch.Tensor,
        graphs: Optional[Batch] = None,
        global_graph: Optional[Data] = None,
        return_attention: bool = False,
        generate_report: bool = False,
        report_target_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            images: (B, 3, H, W) - input images
            graphs: Batched PyTorch Geometric Data - report-specific KGs
            global_graph: PyTorch Geometric Data - global medical KG
            return_attention: Whether to return attention maps
            generate_report: Whether to generate reports
            report_target_ids: (B, L) - target token IDs for training

        Returns:
            Dictionary with:
            - classification_logits: (B, num_classes) if classification enabled
            - report_logits: (B, L, vocab_size) if generation enabled
            - attention_maps: List of attention tensors if return_attention=True
            - generated_text: List of strings if generate_report=True
        """
        outputs = {}

        # Step 1: Visual Encoding
        Z_v = self.visual_encoder(images)  # (B, P, D_v)

        # Step 2a: Knowledge Graph Processing (if use_kg=True)
        if self.config.use_kg:
            if graphs is None or graphs.x.shape[0] == 0:
                # Fallback: if no graph provided, use visual-only mode
                Z_fused = Z_v.mean(dim=1)  # (B, D_v)
                attention_maps = None
            else:
                # Encode knowledge graph
                Z_k = self.knowledge_encoder(
                    x=graphs.x,
                    edge_index=graphs.edge_index,
                    edge_type=graphs.edge_type,
                    batch=graphs.batch,
                )  # (total_nodes, D_k)

                # Reshape Z_k to (B, K, D_k) for fusion
                # Note: This requires knowing how many nodes per graph
                # For now, use simple batching - in practice, handle variable sizes
                batch_size = images.shape[0]
                avg_nodes_per_graph = Z_k.shape[0] // batch_size

                # Simplified: assume equal nodes per graph (handle properly in production)
                Z_k_batched = Z_k.view(batch_size, avg_nodes_per_graph, -1)

                # Fusion: Cross-attention between knowledge and vision
                Z_fused, attention_maps = self.fusion_module(
                    Z_k=Z_k_batched, Z_v=Z_v, return_attention=return_attention
                )  # (B, D)

        # Step 2b: Visual-only baseline (if use_kg=False)
        else:
            Z_fused = self.visual_pooling(Z_v)  # (B, D_v)
            attention_maps = None

        # Step 3: Classification Head
        if self.config.enable_classification and self.classification_head:
            classification_logits = self.classification_head(
                Z_fused
            )  # (B, num_classes)
            outputs["classification_logits"] = classification_logits

        # Step 4: Report Generation
        if self.config.enable_report_generation and self.report_generator:
            if generate_report:
                # Inference mode: generate text
                generated_text = self.report_generator.generate(
                    encoder_output=Z_fused,
                    max_length=self.config.report_generation.max_report_length,
                )
                outputs["generated_text"] = generated_text
            elif report_target_ids is not None:
                # Training mode: compute loss
                report_logits = self.report_generator(
                    encoder_output=Z_fused, target_ids=report_target_ids
                )
                outputs["report_logits"] = report_logits

        # Optional attention maps for explainability
        if return_attention and attention_maps is not None:
            outputs["attention_maps"] = attention_maps

        return outputs


class ClassificationHead(nn.Module):
    """
    Multi-label classification head for pathology prediction.

    Architecture:
        Input → Linear → BatchNorm → ReLU → Dropout → Linear → Sigmoid
    """

    def __init__(self, input_dim: int, config):
        super().__init__()

        layers = []

        # Hidden layer
        layers.append(nn.Linear(input_dim, config.hidden_dim))
        if config.use_batch_norm:
            layers.append(nn.BatchNorm1d(config.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))

        # Output layer
        layers.append(nn.Linear(config.hidden_dim, config.num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) - fused representation

        Returns:
            logits: (B, num_classes) - classification logits
        """
        return self.classifier(x)


class ReportGenerator(nn.Module):
    """
    Transformer-based radiology report generator.

    Uses encoder-decoder architecture where:
    - Encoder output: Fused visual-knowledge representation
    - Decoder: Autoregressive transformer for text generation
    """

    def __init__(self, encoder_dim: int, config):
        super().__init__()
        self.config = config

        # Project encoder output to decoder dimension
        self.encoder_projection = nn.Linear(encoder_dim, config.decoder_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=config.decoder_dim, max_len=config.max_report_length
        )

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.decoder_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.num_decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.decoder_dropout,
            batch_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.decoder_dim, config.vocab_size)

    def forward(
        self, encoder_output: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.

        Args:
            encoder_output: (B, D) - fused representation from encoder
            target_ids: (B, L) - target token IDs

        Returns:
            logits: (B, L, vocab_size) - token prediction logits
        """
        batch_size, seq_len = target_ids.shape

        # Expand encoder output for cross-attention
        memory = self.encoder_projection(encoder_output).unsqueeze(1)  # (B, 1, D)

        # Embed target tokens
        tgt_embed = self.token_embedding(target_ids)  # (B, L, D)
        tgt_embed = self.pos_encoding(tgt_embed)

        # Create causal mask for autoregressive decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=target_ids.device
        )

        # Transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask
        )  # (B, L, D)

        # Project to vocabulary
        logits = self.output_projection(decoder_output)  # (B, L, vocab_size)

        return logits

    def generate(
        self,
        encoder_output: torch.Tensor,
        max_length: int = 128,
        start_token_id: int = 1,
        end_token_id: int = 2,
    ) -> list:
        """
        Autoregressive report generation.

        Args:
            encoder_output: (B, D) - fused representation
            max_length: Maximum report length
            start_token_id: BOS token ID
            end_token_id: EOS token ID

        Returns:
            List of generated token sequences (one per batch item)
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        # Expand encoder output
        memory = self.encoder_projection(encoder_output).unsqueeze(1)  # (B, 1, D)

        # Initialize with start token
        generated = torch.full(
            (batch_size, 1), start_token_id, dtype=torch.long, device=device
        )

        # Greedy decoding (can be replaced with beam search)
        for _ in range(max_length - 1):
            # Embed current sequence
            tgt_embed = self.token_embedding(generated)
            tgt_embed = self.pos_encoding(tgt_embed)

            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.shape[1], device=device
            )

            # Decode
            decoder_output = self.transformer_decoder(
                tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask
            )

            # Get next token prediction
            logits = self.output_projection(decoder_output[:, -1, :])  # (B, vocab_size)
            next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check if all sequences have ended
            if (next_token == end_token_id).all():
                break

        return generated.tolist()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, : x.size(1), :]


# Factory functions for easy model instantiation


def build_model(config: ModelConfig) -> MedicalVLM:
    """
    Build MedicalVLM from configuration.

    Args:
        config: ModelConfig with all settings

    Returns:
        Initialized MedicalVLM model
    """
    model = MedicalVLM(config)
    return model


def build_baseline_model(backbone: str = "biomedclip") -> MedicalVLM:
    """
    Build baseline (Pure Vision) model.

    Args:
        backbone: Visual backbone name

    Returns:
        MedicalVLM with use_kg=False
    """
    from .config import get_baseline_config

    config = get_baseline_config()
    config.visual_encoder.backbone_type = backbone

    model = MedicalVLM(config)
    return model


def build_neurosymbolic_model(backbone: str = "biomedclip") -> MedicalVLM:
    """
    Build Neuro-Symbolic model with full KG integration.

    Args:
        backbone: Visual backbone name

    Returns:
        MedicalVLM with use_kg=True
    """
    from .config import get_neurosymbolic_config

    config = get_neurosymbolic_config()
    config.visual_encoder.backbone_type = backbone

    model = MedicalVLM(config)
    return model


# Example usage
if __name__ == "__main__":
    from .config import ModelConfig

    print("=" * 60)
    print("Building Baseline Model (Pure Vision)")
    print("=" * 60)

    baseline = build_baseline_model(backbone="biomedclip")
    print(f"Baseline parameters: {sum(p.numel() for p in baseline.parameters()):,}")

    # Test forward pass
    dummy_images = torch.randn(2, 3, 224, 224)
    outputs_baseline = baseline(dummy_images)
    print(f"Baseline output keys: {outputs_baseline.keys()}")
    if "classification_logits" in outputs_baseline:
        print(
            f"Classification logits shape: {outputs_baseline['classification_logits'].shape}"
        )

    print("\n" + "=" * 60)
    print("Building Neuro-Symbolic Model (with KG)")
    print("=" * 60)

    neurosymbolic = build_neurosymbolic_model(backbone="biomedclip")
    print(
        f"Neuro-Symbolic parameters: {sum(p.numel() for p in neurosymbolic.parameters()):,}"
    )

    print("\nModel architecture ready for Phase II experiments!")

"""
Core neural network modules for Phase II: Neuro-Symbolic Architecture.

This module implements:
1. VisualEncoder (E_V): Swappable vision transformer backbones
2. KnowledgeEncoder (E_K): Graph Attention Network for knowledge graphs
3. FusionModule: Cross-attention for knowledge-visual fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from transformers import CLIPVisionModel, CLIPImageProcessor
from typing import Optional, Tuple, Dict
import math

from .config import VisualEncoderConfig, KnowledgeEncoderConfig, FusionModuleConfig


class VisualEncoder(nn.Module):
    """
    Visual Encoder (E_V) with swappable CLIP-based backbones.

    Supports:
    - BioMedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
    - PubMedCLIP (flaviagiammarino/pubmed-clip-vit-base-patch32)
    - Standard CLIP ViT-B/16 and ViT-L/14

    Args:
        config: VisualEncoderConfig with backbone selection

    Input:
        x: Tensor of shape (B, 3, H, W) - batch of RGB images

    Output:
        Z_v: Tensor of shape (B, P, D_v) - visual patch embeddings
             where P = num_patches, D_v = hidden_dim
    """

    def __init__(self, config: VisualEncoderConfig):
        super().__init__()
        self.config = config

        # Load pretrained CLIP vision model
        model_checkpoint = config.model_checkpoints[config.backbone_type]
        self.vision_model = CLIPVisionModel.from_pretrained(model_checkpoint)
        self.processor = CLIPImageProcessor.from_pretrained(model_checkpoint)

        # Freeze backbone if specified
        if config.freeze_backbone:
            self._freeze_backbone()

        # Projection to standardize output dimension
        self.output_projection = nn.Linear(
            self.vision_model.config.hidden_size, config.hidden_dim
        )

        # Store dimensions
        self.hidden_dim = config.hidden_dim
        self.num_patches = config.num_patches

    def _freeze_backbone(self):
        """Freeze vision transformer parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Optionally freeze specific layers
        if self.config.freeze_layers is not None:
            for layer_idx in self.config.freeze_layers:
                for param in self.vision_model.vision_model.encoder.layers[
                    layer_idx
                ].parameters():
                    param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: (B, 3, H, W) normalized image tensor

        Returns:
            Z_v: (B, P, D_v) visual patch embeddings
        """
        # CLIP vision model forward
        outputs = self.vision_model(pixel_values, output_hidden_states=True)

        # Get patch embeddings (exclude CLS token)
        # outputs.last_hidden_state shape: (B, P+1, hidden_size)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # (B, P, hidden_size)

        # Project to standard dimension
        Z_v = self.output_projection(patch_embeddings)  # (B, P, D_v)

        return Z_v

    def preprocess_image(self, image):
        """
        Preprocess PIL image or numpy array for CLIP model.

        Args:
            image: PIL.Image or numpy array

        Returns:
            Preprocessed tensor ready for forward pass
        """
        return self.processor(images=image, return_tensors="pt")["pixel_values"]


class KnowledgeEncoder(nn.Module):
    """
    Knowledge Encoder (E_K) using Graph Attention Networks (GAT).

    Implements the equation:
        h_i^{(l+1)} = σ(Σ_{j∈N_i} α_{ij} W h_j^{(l)})

    where α_{ij} are dynamic attention weights computed for each edge type.

    Args:
        config: KnowledgeEncoderConfig

    Input:
        graph: PyTorch Geometric Data or Batch object with:
            - x: Node features (N, F_in)
            - edge_index: Graph connectivity (2, E)
            - edge_attr: Edge type encodings (E, edge_dim)

    Output:
        Z_k: Refined concept embeddings (K, D_k) where K = num_concepts
    """

    def __init__(self, config: KnowledgeEncoderConfig):
        super().__init__()
        self.config = config

        # Edge type embedding
        self.edge_type_embedding = nn.Embedding(config.num_edge_types, config.edge_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # Input layer
        self.gat_layers.append(
            GATv2Conv(
                in_channels=config.concept_embedding_dim,
                out_channels=config.hidden_channels,
                heads=config.num_attention_heads,
                dropout=config.dropout,
                edge_dim=config.edge_dim,
                concat=True,
            )
        )
        self.layer_norms.append(
            nn.LayerNorm(config.hidden_channels * config.num_attention_heads)
        )

        # Hidden layers
        for _ in range(config.num_gat_layers - 1):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=config.hidden_channels * config.num_attention_heads,
                    out_channels=config.hidden_channels,
                    heads=config.num_attention_heads,
                    dropout=config.dropout,
                    edge_dim=config.edge_dim,
                    concat=False,  # Average attention heads in final layers
                )
            )
            self.layer_norms.append(nn.LayerNorm(config.hidden_channels))

        # Output projection to match visual dimension
        self.output_projection = nn.Linear(
            config.hidden_channels, config.concept_embedding_dim
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GAT layers.

        Args:
            x: Node features (N, F_in)
            edge_index: Graph connectivity (2, E)
            edge_type: Edge types (E,) - integers 0-3
            batch: Batch assignment (N,) - which graph each node belongs to

        Returns:
            Z_k: Refined concept embeddings (N, D_k)
        """
        # Encode edge types
        edge_attr = self.edge_type_embedding(edge_type)  # (E, edge_dim)

        h = x
        for i, (gat_layer, layer_norm) in enumerate(
            zip(self.gat_layers, self.layer_norms)
        ):
            # GAT layer with residual connection
            h_new = gat_layer(h, edge_index, edge_attr=edge_attr)
            h_new = layer_norm(h_new)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)

            # Residual connection (if dimensions match)
            if i > 0:
                h = h + h_new
            else:
                h = h_new

        # Project to output dimension
        Z_k = self.output_projection(h)

        # Normalize if specified
        if self.config.normalize_outputs:
            Z_k = F.normalize(Z_k, p=2, dim=-1)

        return Z_k


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention module.

    Implements: Attention(Q, K, V) = softmax(QK^T / √d) V

    where:
    - Q comes from knowledge concepts (queries)
    - K, V come from visual patches
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Query projection (from knowledge)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Key and Value projections (from visual)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Cross-attention forward pass.

        Args:
            query: (B, K, D) - knowledge concept embeddings
            key_value: (B, P, D) - visual patch embeddings
            return_attention: Whether to return attention weights

        Returns:
            output: (B, K, D) - attended features
            attention_weights: (B, num_heads, K, P) - optional attention map
        """
        B, K, D = query.shape
        _, P, _ = key_value.shape

        # Project to Q, K, V
        Q = self.q_proj(query)  # (B, K, D)
        K = self.k_proj(key_value)  # (B, P, D)
        V = self.v_proj(key_value)  # (B, P, D)

        # Reshape for multi-head attention
        Q = Q.reshape(B, K, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, K, d)
        K = K.reshape(B, P, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, P, d)
        V = V.reshape(B, P, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, P, d)

        # Compute attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, K, P)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = attn @ V  # (B, H, K, d)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, K, D)  # (B, K, D)
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        if return_attention:
            return out, attn
        return out, None


class FusionModule(nn.Module):
    """
    Knowledge-Visual Fusion Module using Cross-Attention.

    Medical concepts (Z_k) act as queries to attend over visual patches (Z_v),
    highlighting diagnostically relevant image regions.

    Architecture:
        Z_k (queries) → Cross-Attention → Z_v (keys/values)
        → Bottleneck → Fused representation

    Args:
        config: FusionModuleConfig

    Input:
        Z_k: (B, K, D) - knowledge concept embeddings
        Z_v: (B, P, D) - visual patch embeddings

    Output:
        Z_fused: (B, D) - fused multimodal representation
    """

    def __init__(self, config: FusionModuleConfig):
        super().__init__()
        self.config = config

        # Multi-layer cross-attention
        self.fusion_layers = nn.ModuleList(
            [
                MultiHeadCrossAttention(
                    dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    qkv_bias=config.qkv_bias,
                    attn_drop=config.attention_dropout,
                    proj_drop=config.projection_dropout,
                )
                for _ in range(config.num_fusion_layers)
            ]
        )

        # Layer normalization
        self.layer_norms_q = nn.ModuleList(
            [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_fusion_layers)]
        )

        # Feed-forward networks
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(config.projection_dropout),
                    nn.Linear(config.hidden_dim * 4, config.hidden_dim),
                    nn.Dropout(config.projection_dropout),
                )
                for _ in range(config.num_fusion_layers)
            ]
        )

        self.layer_norms_ffn = nn.ModuleList(
            [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_fusion_layers)]
        )

        # Attention bottleneck for final fusion
        if config.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(config.hidden_dim, config.bottleneck_dim),
                nn.GELU(),
                nn.Dropout(config.projection_dropout),
                nn.Linear(config.bottleneck_dim, config.hidden_dim),
            )

        # Final projection
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self, Z_k: torch.Tensor, Z_v: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse knowledge and visual representations.

        Args:
            Z_k: (B, K, D_k) - knowledge concept embeddings
            Z_v: (B, P, D_v) - visual patch embeddings
            return_attention: Whether to return attention maps

        Returns:
            Z_fused: (B, D) - fused representation (pooled over K concepts)
            attention_maps: List of attention weights from each layer (optional)
        """
        attention_maps = [] if return_attention else None

        # Multi-layer cross-attention with residual connections
        h_knowledge = Z_k
        for i, (attn_layer, ln_q, ffn, ln_ffn) in enumerate(
            zip(self.fusion_layers, self.layer_norms_q, self.ffns, self.layer_norms_ffn)
        ):
            # Cross-attention: knowledge queries attend to visual patches
            attn_out, attn_weights = attn_layer(
                h_knowledge, Z_v, return_attention=return_attention
            )

            if return_attention and attn_weights is not None:
                attention_maps.append(attn_weights)

            # Residual connection
            if self.config.use_residual:
                h_knowledge = ln_q(h_knowledge + attn_out)
            else:
                h_knowledge = ln_q(attn_out)

            # Feed-forward network with residual
            ffn_out = ffn(h_knowledge)
            h_knowledge = ln_ffn(h_knowledge + ffn_out)

        # Global pooling over concepts: (B, K, D) → (B, D)
        Z_fused = h_knowledge.mean(dim=1)  # Average pooling over K concepts

        # Apply bottleneck
        if self.config.use_bottleneck:
            Z_fused = self.bottleneck(Z_fused)

        # Final projection
        Z_fused = self.output_projection(Z_fused)

        return Z_fused, attention_maps


class SelfAttentionPooling(nn.Module):
    """
    Self-attention based pooling for aggregating patch embeddings.

    Computes: α_i = softmax(w^T tanh(W h_i))
    Output: Σ α_i h_i
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - sequence of embeddings

        Returns:
            pooled: (B, D) - weighted average
        """
        # Compute attention scores
        attn_scores = self.attention_weights(x)  # (B, N, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, N, 1)

        # Weighted sum
        pooled = (attn_weights * x).sum(dim=1)  # (B, D)

        return pooled

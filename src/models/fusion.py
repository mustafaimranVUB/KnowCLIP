"""Cross-Attention Fusion Module.

Fuses knowledge embeddings (Z_k) with visual patch embeddings (Z_v)
via multi-head cross-attention.  Knowledge concepts act as **queries**
attending over visual patches, highlighting diagnostically relevant
image regions.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import FusionModuleConfig
from src.models.interfaces import BaseFusionModule

logger = logging.getLogger(__name__)


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with feed-forward network.

    Implements: LayerNorm → MultiHeadCrossAttention → Residual → LayerNorm → FFN → Residual
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        ffn_dim = ffn_dim or hidden_dim * 4

        # Cross-attention: Q from Z_k, K/V from Z_v
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Layer norms (post-norm style)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        Z_k: torch.Tensor,
        Z_v: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Cross-attention: Z_k queries Z_v.

        Args:
            Z_k: ``(B, K, D)`` knowledge embeddings (queries).
            Z_v: ``(B, P, D)`` visual patch embeddings (keys/values).
            return_attention: If True, also return attention weights.

        Returns:
            ``(B, K, D)`` updated knowledge embeddings.
        """
        B, K, D = Z_k.shape
        _, P, _ = Z_v.shape

        # Multi-head cross-attention
        residual = Z_k
        Q = self.q_proj(Z_k).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, K, d)
        K_proj = self.k_proj(Z_v).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, P, d)
        V = self.v_proj(Z_v).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, P, d)

        attn_weights = torch.matmul(Q, K_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, K, P)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights_dropped, V)  # (B, H, K, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, K, D)  # (B, K, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        # Residual + LayerNorm
        h = self.norm1(residual + attn_output)

        # FFN
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)

        if return_attention:
            return h, attn_weights
        return h


class CrossAttentionFusion(BaseFusionModule):
    """Multi-layer cross-attention fusion module.

    Parameters:
        config: Fusion module configuration.
    """

    def __init__(self, config: FusionModuleConfig) -> None:
        nn.Module.__init__(self)
        self.config = config

        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.attention_dropout,
                qkv_bias=config.qkv_bias,
            )
            for _ in range(config.num_fusion_layers)
        ])

        # Optional bottleneck
        if config.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(config.hidden_dim, config.bottleneck_dim),
                nn.GELU(),
                nn.Linear(config.bottleneck_dim, config.hidden_dim),
            )
        else:
            self.bottleneck = nn.Identity()

    def forward(
        self,
        Z_k: torch.Tensor,
        Z_v: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict[str, object]]:
        """Fuse knowledge and visual features.

        Args:
            Z_k: ``(B, K, D)`` knowledge embeddings.
            Z_v: ``(B, P, D)`` visual patch embeddings.
            return_attention: Whether to return last-layer attention weights.

        Returns:
            ``(B, K, D)`` fused features.
        """
        h = Z_k
        attn_weights = None
        attention_per_layer: list[torch.Tensor] = []

        for layer in self.layers:
            if return_attention:
                h, attn_weights = layer(h, Z_v, return_attention=True)
                attention_per_layer.append(attn_weights)
            else:
                h = layer(h, Z_v, return_attention=False)

        h = self.bottleneck(h)

        if return_attention and attn_weights is not None:
            return h, {
                "per_layer": attention_per_layer,
                "last_layer": attn_weights,
            }
        return h

    def get_output_dim(self) -> int:
        return self.config.hidden_dim


class SelfAttentionPooling(nn.Module):
    """Self-attention pooling for the baseline model (no KG).

    Pools patch embeddings ``(B, P, D)`` → ``(B, D)`` using learned
    attention weights.
    """

    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pool patch embeddings.

        Args:
            x: ``(B, P, D)`` patch embeddings.
            mask: Optional ``(B, P)`` bool tensor where ``True`` marks valid
                positions.
            return_attention: If ``True``, also return the pooling weights.

        Returns:
            ``(B, D)`` pooled representation.
        """
        logits = self.attn(x)  # (B, P, 1)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(logits, dim=1)
        pooled = (weights * x).sum(dim=1)  # (B, D)
        pooled = self.norm(pooled)
        if return_attention:
            return pooled, weights.squeeze(-1)
        return pooled

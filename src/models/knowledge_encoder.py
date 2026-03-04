"""GATv2-based Knowledge Encoder (E_K).

Encodes the hybrid knowledge graph using Graph Attention Networks with
edge-type-aware embeddings, producing node representations in the same
space as the visual encoder output for cross-attention fusion.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import KnowledgeEncoderConfig
from src.knowledge.graph_builder import NUM_EDGE_TYPES
from src.models.interfaces import BaseKnowledgeEncoder

logger = logging.getLogger(__name__)


class GATv2KnowledgeEncoder(BaseKnowledgeEncoder):
    """Graph Attention Network v2 knowledge encoder.

    Architecture:
        - *L* GATv2Conv layers with edge-type embeddings.
        - First layer: ``concat=True`` (output = hidden_channels × heads).
        - Subsequent layers: ``concat=False`` (output = hidden_channels).
        - Residual connections from layer 2 onward.
        - LayerNorm after each layer.
        - Output projection to ``output_dim``.
        - L2 normalisation of output.

    Parameters:
        config: Knowledge encoder configuration.
    """

    def __init__(self, config: KnowledgeEncoderConfig) -> None:
        nn.Module.__init__(self)
        self.config = config

        # Edge-type embedding
        self.edge_embedding = nn.Embedding(
            num_embeddings=config.num_edge_types or NUM_EDGE_TYPES,
            embedding_dim=config.edge_dim,
        )

        # Build GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        from torch_geometric.nn import GATv2Conv  # type: ignore

        for i in range(config.num_gat_layers):
            if i == 0:
                in_channels = config.input_dim
                concat = True
                out_channels = config.hidden_channels
                total_out = out_channels * config.num_attention_heads
            else:
                in_channels = config.hidden_channels * config.num_attention_heads if i == 1 else config.hidden_channels
                concat = False
                out_channels = config.hidden_channels
                total_out = out_channels

            gat = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=config.num_attention_heads,
                concat=concat,
                dropout=config.dropout,
                edge_dim=config.edge_dim,
                add_self_loops=False,  # We add them in the graph builder
            )
            self.gat_layers.append(gat)
            self.norms.append(nn.LayerNorm(total_out))

        # Output projection
        final_dim = config.hidden_channels
        self.output_proj = nn.Linear(final_dim, config.output_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode graph nodes.

        Args:
            x: ``(N, D_in)`` node features.
            edge_index: ``(2, E)`` edge indices.
            edge_type: ``(E,)`` integer edge types.
            batch: ``(N,)`` batch vector (unused here but part of interface).

        Returns:
            ``(N, output_dim)`` encoded node embeddings.
        """
        # Compute edge-type embeddings
        edge_attr = self.edge_embedding(edge_type)  # (E, edge_dim)

        h = x
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            h_in = h
            h = gat(h, edge_index, edge_attr=edge_attr)  # GATv2Conv
            h = F.elu(h)
            h = norm(h)
            h = self.dropout(h)

            # Residual connection from layer 2 onward (if dims match)
            if i >= 1 and h.shape == h_in.shape:
                h = h + h_in

        # Output projection + L2 normalisation
        h = self.output_proj(h)

        if self.config.normalize_outputs:
            h = F.normalize(h, p=2, dim=-1)

        return h

    def get_output_dim(self) -> int:
        return self.config.output_dim

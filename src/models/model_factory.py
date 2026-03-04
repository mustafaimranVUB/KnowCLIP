"""Model factory — assembles the complete MedicalVLM from config.

Follows the **Factory** and **Composition** patterns: the orchestrator
(:class:`MedicalVLM`) *contains* (not *is-a*) encoders, fusion, and
heads.  Construction is driven entirely by :class:`ModelConfig`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import ModelConfig
from src.models.interfaces import (
    BaseDecoder,
    BaseFusionModule,
    BaseKnowledgeEncoder,
    BaseVisualEncoder,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MedicalVLM(nn.Module):
    """Knowledge-graph-infused Vision-Language Model (KnoCLIP-XAI).

    The model orchestrates four components via their ABCs:

    1. **Visual Encoder** (E_V):  image → patch embeddings
    2. **Knowledge Encoder** (E_K):  graph → node embeddings
    3. **Fusion Module**:  cross-attention Z_k × Z_v → Z_fused
    4. **Task heads**:  classification and/or report generation

    Parameters:
        config: Model configuration.
        visual_encoder: Concrete visual encoder.
        knowledge_encoder: Concrete knowledge encoder (None if baseline).
        fusion_module: Concrete fusion module (None if baseline).
        classification_head: Classification head module (optional).
        decoder: Report decoder module (optional).
    """

    def __init__(
        self,
        config: ModelConfig,
        visual_encoder: BaseVisualEncoder,
        knowledge_encoder: Optional[BaseKnowledgeEncoder] = None,
        fusion_module: Optional[BaseFusionModule] = None,
        classification_head: Optional[nn.Module] = None,
        decoder: Optional[BaseDecoder] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.visual_encoder = visual_encoder
        self.knowledge_encoder = knowledge_encoder
        self.fusion_module = fusion_module
        self.classification_head = classification_head
        self.decoder = decoder

        # Baseline pooling (used when knowledge encoder is absent)
        if not config.use_kg:
            from src.models.fusion import SelfAttentionPooling

            self.baseline_pooling = SelfAttentionPooling(
                hidden_dim=visual_encoder.get_output_dim()
            )
        else:
            self.baseline_pooling = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_edge_type: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
        graph_num_nodes_per_sample: Optional[List[int]] = None,
        target_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """Full forward pass.

        Args:
            pixel_values: ``(B, 3, H, W)`` images.
            graph_x: ``(N_total, D)`` batched node features.
            graph_edge_index: ``(2, E_total)`` batched edge indices.
            graph_edge_type: ``(E_total,)`` edge types.
            graph_batch: ``(N_total,)`` batch vector for graph nodes.
            graph_num_nodes_per_sample: Number of nodes per sample
                (for re-batching).
            target_ids: ``(B, L)`` target token IDs for report generation.
            return_attention: Return attention maps for explainability.

        Returns:
            Dict with keys: ``classification_logits``, ``generation_logits``,
            ``fused_features``, ``visual_features``, ``attention_weights``.
        """
        outputs: Dict[str, Any] = {}

        # ----- Visual Encoding -----
        Z_v = self.visual_encoder(pixel_values)  # (B, P, D)
        outputs["visual_features"] = Z_v

        if self.config.use_kg and self.knowledge_encoder is not None and graph_x is not None:
            # ----- Knowledge Encoding -----
            Z_k_flat = self.knowledge_encoder(
                graph_x, graph_edge_index, graph_edge_type, graph_batch
            )  # (N_total, D)

            # Re-batch: (N_total, D) → (B, K_max, D) with padding
            Z_k = self._rebatch_graph(
                Z_k_flat, graph_batch, pixel_values.shape[0]
            )  # (B, K_max, D)

            # ----- Fusion -----
            if self.fusion_module is not None:
                if return_attention:
                    Z_fused, attn_weights = self.fusion_module(
                        Z_k, Z_v, return_attention=True
                    )
                    outputs["attention_weights"] = attn_weights
                else:
                    Z_fused = self.fusion_module(Z_k, Z_v)
            else:
                Z_fused = Z_k

            outputs["fused_features"] = Z_fused

            # Pool fused features for classification: mean over K dim
            pooled = Z_fused.mean(dim=1)  # (B, D)

        else:
            # ----- Baseline (no KG) -----
            pooled = self.baseline_pooling(Z_v)  # (B, D)
            Z_fused = None
            outputs["fused_features"] = None

        # ----- Classification -----
        if self.classification_head is not None and self.config.enable_classification:
            classification_logits = self.classification_head(pooled)  # (B, C)
            outputs["classification_logits"] = classification_logits

        # ----- Report Generation -----
        if self.decoder is not None and self.config.enable_report_generation and target_ids is not None:
            # Decoder uses fused features (or just visual if baseline)
            decoder_context = Z_fused if Z_fused is not None else Z_v
            generation_logits = self.decoder(decoder_context, target_ids)
            outputs["generation_logits"] = generation_logits

        return outputs

    def generate_report(
        self,
        pixel_values: torch.Tensor,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_edge_type: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
        max_length: int = 128,
    ) -> List[List[int]]:
        """Generate reports auto-regressively.

        Args:
            pixel_values: ``(B, 3, H, W)`` images.
            graph_*: Optional graph tensors.
            max_length: Max tokens.

        Returns:
            List of generated token ID sequences.
        """
        if self.decoder is None:
            raise RuntimeError("Decoder not configured for report generation")

        Z_v = self.visual_encoder(pixel_values)

        if self.config.use_kg and self.knowledge_encoder is not None and graph_x is not None:
            Z_k_flat = self.knowledge_encoder(
                graph_x, graph_edge_index, graph_edge_type, graph_batch
            )
            Z_k = self._rebatch_graph(Z_k_flat, graph_batch, pixel_values.shape[0])

            if self.fusion_module is not None:
                Z_fused = self.fusion_module(Z_k, Z_v)
            else:
                Z_fused = Z_k

            context = Z_fused
        else:
            context = Z_v

        return self.decoder.generate(context, max_length=max_length)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rebatch_graph(
        node_embeddings: torch.Tensor,
        batch_vector: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Convert flat node embeddings → padded batch tensor.

        Args:
            node_embeddings: ``(N_total, D)``
            batch_vector: ``(N_total,)`` integer batch assignment.
            batch_size: B.

        Returns:
            ``(B, K_max, D)`` zero-padded.
        """
        D = node_embeddings.shape[1]
        device = node_embeddings.device

        # Find max nodes per sample
        counts = torch.bincount(batch_vector, minlength=batch_size)
        K_max = int(counts.max().item())

        result = torch.zeros(batch_size, K_max, D, device=device)
        idx = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(node_embeddings.shape[0]):
            b = batch_vector[i].item()
            pos = idx[b].item()
            result[b, pos] = node_embeddings[i]
            idx[b] += 1

        return result


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(config: ModelConfig) -> MedicalVLM:
    """Construct a :class:`MedicalVLM` from a :class:`ModelConfig`.

    This is the **only** entry point for model construction — callers
    should never instantiate encoders, fusion modules, or heads directly.

    Args:
        config: Model configuration specifying all components.

    Returns:
        Fully assembled :class:`MedicalVLM`.
    """
    from src.models.visual_encoder import CLIPVisualEncoder
    from src.models.knowledge_encoder import GATv2KnowledgeEncoder
    from src.models.fusion import CrossAttentionFusion
    from src.models.classification import ClassificationHead
    from src.models.decoder import TransformerReportDecoder

    # Visual encoder (always required)
    visual_encoder = CLIPVisualEncoder(config.visual_encoder)

    # Knowledge encoder (only if KG enabled)
    knowledge_encoder = None
    fusion_module = None
    if config.use_kg:
        knowledge_encoder = GATv2KnowledgeEncoder(config.knowledge_encoder)
        fusion_module = CrossAttentionFusion(config.fusion_module)

    # Classification head
    classification_head = None
    if config.enable_classification:
        classification_head = ClassificationHead(
            config.classification_head,
            input_dim=config.visual_encoder.output_dim,
        )

    # Report decoder
    decoder = None
    if config.enable_report_generation:
        decoder = TransformerReportDecoder(config.report_generation)

    model = MedicalVLM(
        config=config,
        visual_encoder=visual_encoder,
        knowledge_encoder=knowledge_encoder,
        fusion_module=fusion_module,
        classification_head=classification_head,
        decoder=decoder,
    )

    # Log param count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built MedicalVLM: %.2fM params (%.2fM trainable) | use_kg=%s | classify=%s | generate=%s",
        total_params / 1e6,
        trainable_params / 1e6,
        config.use_kg,
        config.enable_classification,
        config.enable_report_generation,
    )

    return model

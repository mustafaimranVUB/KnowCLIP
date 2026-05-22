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

        # Baseline pooling (always initialized as fallback)
        # Used when: (1) use_kg=False, or (2) KG data missing at inference
        from src.models.fusion import SelfAttentionPooling

        self.concept_pooling = SelfAttentionPooling(
            hidden_dim=visual_encoder.get_output_dim()
        )
        self.baseline_pooling = SelfAttentionPooling(
            hidden_dim=visual_encoder.get_output_dim()
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_edge_type: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
        graph_num_nodes_per_sample: Optional[List[int]] = None,
        target_ids: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
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
        explainability: Dict[str, Any] = {}

        # ----- Visual Encoding -----
        Z_v = self.visual_encoder(pixel_values)  # (B, P, D)
        outputs["visual_features"] = Z_v

        if self.config.use_kg and self.knowledge_encoder is not None and graph_x is not None:
            # ----- Knowledge Encoding -----
            if return_attention:
                Z_k_flat, graph_trace = self.knowledge_encoder(
                    graph_x,
                    graph_edge_index,
                    graph_edge_type,
                    graph_batch,
                    return_attention=True,
                )
                explainability["graph"] = graph_trace
            else:
                Z_k_flat = self.knowledge_encoder(
                    graph_x, graph_edge_index, graph_edge_type, graph_batch
                )  # (N_total, D)

            # Re-batch: (N_total, D) → (B, K_max, D) with padding
            Z_k = self._rebatch_graph(
                Z_k_flat, graph_batch, pixel_values.shape[0]
            )  # (B, K_max, D)
            outputs["knowledge_features"] = Z_k

            counts = torch.bincount(graph_batch, minlength=pixel_values.shape[0])
            mask = torch.arange(Z_k.shape[1], device=Z_k.device).unsqueeze(0) < counts.unsqueeze(1)  # (B, K_max)

            # ----- Fusion -----
            if self.fusion_module is not None:
                if return_attention:
                    Z_fused, fusion_trace = self.fusion_module(
                        Z_k, Z_v, return_attention=True
                    )
                    outputs["attention_weights"] = fusion_trace["last_layer"]
                    explainability["fusion"] = fusion_trace
                else:
                    Z_fused = self.fusion_module(Z_k, Z_v)
            else:
                Z_fused = Z_k

            outputs["fused_features"] = Z_fused

            # Methodology-aligned concept pooling with explicit attention weights.
            if return_attention:
                pooled, pooling_weights = self.concept_pooling(
                    Z_fused,
                    mask=mask,
                    return_attention=True,
                )
                explainability["pooling"] = {
                    "weights": pooling_weights,
                    "mask": mask,
                }
            else:
                pooled = self.concept_pooling(Z_fused, mask=mask)

        else:
            # ----- Baseline (no KG) -----
            if return_attention:
                pooled, pooling_weights = self.baseline_pooling(
                    Z_v,
                    return_attention=True,
                )
                explainability["pooling"] = {
                    "weights": pooling_weights,
                    "mask": torch.ones(
                        Z_v.shape[0],
                        Z_v.shape[1],
                        dtype=torch.bool,
                        device=Z_v.device,
                    ),
                }
            else:
                pooled = self.baseline_pooling(Z_v)  # (B, D)
            Z_fused = None
            outputs["fused_features"] = None
            outputs["knowledge_features"] = None

        # ----- Classification -----
        if self.classification_head is not None and self.config.enable_classification:
            classification_logits = self.classification_head(pooled)  # (B, C)
            outputs["classification_logits"] = classification_logits

        # ----- Report Generation -----
        if self.decoder is not None and self.config.enable_report_generation and target_ids is not None:
            # Decoder cross-attends to both KG-fused features AND visual
            # patches so it has rich context for text generation.  Using
            # only Z_fused (~5-20 KG nodes) is an information bottleneck
            # that produces incoherent output.
            encoder_padding_mask: Optional[torch.Tensor] = None
            if Z_fused is not None:
                decoder_context = torch.cat([Z_fused, Z_v], dim=1)  # (B, K+P, D)
                # Mask: True for zero-padded KG positions the decoder should ignore
                kg_pad = torch.arange(Z_fused.shape[1], device=Z_fused.device).unsqueeze(0) >= counts.unsqueeze(1)  # (B, K_max)
                vis_pad = torch.zeros(Z_v.shape[0], Z_v.shape[1], dtype=torch.bool, device=Z_v.device)  # (B, P)
                encoder_padding_mask = torch.cat([kg_pad, vis_pad], dim=1)  # (B, K+P)
            else:
                decoder_context = Z_v  # baseline: visual patches only
            if return_attention:
                generation_result = self.decoder(
                    decoder_context,
                    target_ids,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    encoder_padding_mask=encoder_padding_mask,
                    return_attention=True,
                )
                if isinstance(generation_result, tuple):
                    generation_logits, decoder_trace = generation_result
                else:
                    generation_logits, decoder_trace = generation_result, None
                if decoder_trace is not None:
                    explainability["decoder"] = decoder_trace
            else:
                generation_logits = self.decoder(
                    decoder_context,
                    target_ids,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    encoder_padding_mask=encoder_padding_mask,
                )
            outputs["generation_logits"] = generation_logits

        if return_attention:
            outputs["explainability"] = explainability

        return outputs

    def generate_report(
        self,
        pixel_values: torch.Tensor,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_edge_type: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """Generate reports auto-regressively.

        Args:
            pixel_values: ``(B, 3, H, W)`` images.
            graph_*: Optional graph tensors.
            max_length: Max tokens. If None, uses config ``max_report_length``.

        Returns:
            List of generated token ID sequences.
        """
        if self.decoder is None:
            raise RuntimeError("Decoder not configured for report generation")

        Z_v = self.visual_encoder(pixel_values)

        encoder_padding_mask: Optional[torch.Tensor] = None
        if self.config.use_kg and self.knowledge_encoder is not None and graph_x is not None:
            Z_k_flat = self.knowledge_encoder(
                graph_x, graph_edge_index, graph_edge_type, graph_batch
            )
            Z_k = self._rebatch_graph(Z_k_flat, graph_batch, pixel_values.shape[0])

            if self.fusion_module is not None:
                Z_fused = self.fusion_module(Z_k, Z_v)
            else:
                Z_fused = Z_k

            # Concatenate KG-fused features with visual patches so the
            # decoder has both semantic KG guidance and fine-grained
            # visual detail for report generation.
            context = torch.cat([Z_fused, Z_v], dim=1)  # (B, K+P, D)
            # Mask: True for zero-padded KG positions
            counts = torch.bincount(graph_batch, minlength=pixel_values.shape[0])
            kg_pad = torch.arange(Z_fused.shape[1], device=Z_fused.device).unsqueeze(0) >= counts.unsqueeze(1)
            vis_pad = torch.zeros(Z_v.shape[0], Z_v.shape[1], dtype=torch.bool, device=Z_v.device)
            encoder_padding_mask = torch.cat([kg_pad, vis_pad], dim=1)
        else:
            context = Z_v

        decode_len = int(max_length) if max_length is not None else int(self.config.report_generation.max_report_length)
        return self.decoder.generate(context, max_length=decode_len, encoder_padding_mask=encoder_padding_mask)

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

        counts = torch.bincount(batch_vector, minlength=batch_size)
        K_max = int(counts.max().item())
        result = torch.zeros(batch_size, K_max, D, device=device)

        # Vectorised per-node position within its sample (replaces Python loop).
        # For each node, compute its index within the batch-group using
        # cumulative counting: ones-per-sample → cumsum → subtract base offset.
        ones = torch.ones(node_embeddings.shape[0], dtype=torch.long, device=device)
        cumsum = torch.cumsum(ones, dim=0)  # 1-based running count
        batch_starts = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        batch_starts[1:] = torch.cumsum(counts, dim=0)
        offsets = cumsum - 1 - batch_starts[batch_vector]

        result[batch_vector, offsets] = node_embeddings

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
    from src.models.decoder import TransformerReportDecoder, GPT2ReportDecoder

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
        decoder_type = getattr(config.report_generation, "decoder_type", "transformer")
        if decoder_type == "gpt2":
            decoder = GPT2ReportDecoder(config.report_generation)
        else:
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

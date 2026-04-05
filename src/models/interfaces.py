"""Abstract Base Classes (ABCs) defining the component interfaces.

Every concrete encoder, fusion module, and decoder **must** inherit from
the corresponding ABC.  The orchestrator (:class:`MedicalVLM` in
``model_factory.py``) programs to these interfaces, not to
implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn


class BaseVisualEncoder(ABC, nn.Module):
    """Interface for all visual encoders (E_V).

    Contract:
        ``forward(pixel_values)`` → ``(B, P, D)`` patch embeddings.
    """

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into patch-level embeddings.

        Args:
            pixel_values: ``(B, 3, H, W)`` images.

        Returns:
            ``(B, P, D)`` patch embeddings (CLS token excluded).
        """
        ...

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the output embedding dimension."""
        ...

    @abstractmethod
    def get_num_patches(self) -> int:
        """Return the number of output patches."""
        ...


class BaseKnowledgeEncoder(ABC, nn.Module):
    """Interface for all knowledge-graph encoders (E_K).

    Contract:
        ``forward(x, edge_index, edge_type, batch)`` → ``(N, D)``
    """

    @abstractmethod
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
            edge_index: ``(2, E)`` COO edge indices.
            edge_type: ``(E,)`` integer edge types.
            batch: ``(N,)`` batch assignment (for batched graphs).

        Returns:
            ``(N, D_out)`` encoded node embeddings.
        """
        ...

    @abstractmethod
    def get_output_dim(self) -> int:
        ...


class BaseFusionModule(ABC, nn.Module):
    """Interface for knowledge–visual fusion.

    Contract:
        ``forward(Z_k, Z_v)`` → ``(B, K, D)`` fused representations.
    """

    @abstractmethod
    def forward(
        self,
        Z_k: torch.Tensor,
        Z_v: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Fuse knowledge and visual representations.

        Args:
            Z_k: ``(B, K, D)`` knowledge embeddings.
            Z_v: ``(B, P, D)`` visual patch embeddings.
            return_attention: If ``True``, also return attention weights.

        Returns:
            ``(B, K, D)`` fused features (and optionally attention weights).
        """
        ...

    @abstractmethod
    def get_output_dim(self) -> int:
        ...


class BaseDecoder(ABC, nn.Module):
    """Interface for report-generation decoders.

    Contract:
        ``forward(encoder_output, target_ids)`` → logits
        ``generate(encoder_output, max_length)`` → token sequences
    """

    @abstractmethod
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute next-token logits given encoder context and target IDs.

        Args:
            encoder_output: ``(B, K, D)`` fused features.
            target_ids: ``(B, L)`` target token IDs (teacher-forced).
            teacher_forcing_ratio: Probability of using ground-truth tokens.
            encoder_padding_mask: ``(B, K)`` bool mask — ``True`` for
                positions to **ignore** (zero-padded KG nodes).

        Returns:
            ``(B, L, vocab_size)`` logits.
        """
        ...

    @abstractmethod
    def generate(
        self,
        encoder_output: torch.Tensor,
        max_length: int = 128,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Auto-regressively generate token sequences.

        Args:
            encoder_output: ``(B, K, D)`` fused features.
            max_length: Max tokens to generate.
            encoder_padding_mask: ``(B, K)`` bool mask — ``True`` for
                positions to **ignore**.

        Returns:
            List of token ID sequences, one per batch item.
        """
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        ...

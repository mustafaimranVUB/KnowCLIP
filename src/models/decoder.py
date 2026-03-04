"""Transformer decoder for radiology report generation.

A vanilla Transformer decoder with cross-attention over fused
knowledge–visual features.  Uses sinusoidal positional encoding.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.config import ReportGenerationConfig
from src.models.interfaces import BaseDecoder

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: ``(B, L, D)`` token embeddings.

        Returns:
            ``(B, L, D)`` with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerReportDecoder(BaseDecoder):
    """Transformer decoder for auto-regressive report generation.

    Parameters:
        config: Report generation configuration.
    """

    def __init__(self, config: ReportGenerationConfig) -> None:
        nn.Module.__init__(self)
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.decoder_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=config.decoder_dim,
            max_len=config.max_report_length + 10,
            dropout=config.decoder_dropout,
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.num_decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
        )

        # Output head
        self.output_proj = nn.Linear(config.decoder_dim, config.vocab_size, bias=False)

        # Tie weights
        self.output_proj.weight = self.token_embedding.weight

        # Causal mask cache
        self._causal_mask_cache: Optional[torch.Tensor] = None

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute next-token logits (teacher-forced).

        Args:
            encoder_output: ``(B, K, D)`` fused features as memory.
            target_ids: ``(B, L)`` target token IDs.

        Returns:
            ``(B, L, vocab_size)`` logits.
        """
        B, L = target_ids.shape

        # Token + positional embedding
        tgt = self.token_embedding(target_ids)  # (B, L, D)
        tgt = self.pos_encoding(tgt)

        # Causal mask
        causal_mask = self._get_causal_mask(L, target_ids.device)

        # Decode
        decoded = self.decoder(
            tgt=tgt,
            memory=encoder_output,
            tgt_mask=causal_mask,
        )  # (B, L, D)

        logits = self.output_proj(decoded)  # (B, L, vocab_size)
        return logits

    def generate(
        self,
        encoder_output: torch.Tensor,
        max_length: int = 128,
    ) -> List[List[int]]:
        """Auto-regressive beam-search generation.

        For simplicity, this implements **greedy** decoding.
        Full beam search is a future enhancement.

        Args:
            encoder_output: ``(B, K, D)`` fused features.
            max_length: Maximum tokens to generate.

        Returns:
            List of token ID sequences.
        """
        B = encoder_output.shape[0]
        device = encoder_output.device

        # Start with BOS token (assume token_id = 50256 for GPT-2)
        bos_token_id = self.config.vocab_size - 1  # GPT-2 EOS/BOS token
        input_ids = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        generated: List[List[int]] = [[] for _ in range(B)]
        finished = [False] * B

        for step in range(max_length):
            logits = self.forward(encoder_output, input_ids)  # (B, L, V)
            next_logits = logits[:, -1, :]  # (B, V)

            # Greedy selection
            next_token = next_logits.argmax(dim=-1)  # (B,)

            for b in range(B):
                if not finished[b]:
                    tok = next_token[b].item()
                    if tok == bos_token_id:  # EOS
                        finished[b] = True
                    else:
                        generated[b].append(tok)

            if all(finished):
                break

            input_ids = torch.cat(
                [input_ids, next_token.unsqueeze(1)], dim=1
            )

        return generated

    def get_vocab_size(self) -> int:
        return self.config.vocab_size

    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Get or create a causal attention mask."""
        if self._causal_mask_cache is None or self._causal_mask_cache.shape[0] < size:
            mask = nn.Transformer.generate_square_subsequent_mask(size, device=device)
            self._causal_mask_cache = mask
            return mask
        return self._causal_mask_cache[:size, :size].to(device)

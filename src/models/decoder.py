"""Transformer decoder for radiology report generation.

Two implementations:
- ``TransformerReportDecoder``: Vanilla Transformer decoder (random init
  apart from GPT-2 embeddings).  Preserved for backward-compatibility.
- ``GPT2ReportDecoder``: Pretrained GPT-2 backbone with injected
  cross-attention layers.  Much stronger language prior for small
  datasets (<5 k samples).
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


# ---------------------------------------------------------------------------
# Shared generation utilities
# ---------------------------------------------------------------------------

def _apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits for already generated tokens."""
    adjusted = logits.clone()
    for b in range(input_ids.shape[0]):
        for tok in set(int(t) for t in input_ids[b].tolist()):
            score = adjusted[b, tok]
            adjusted[b, tok] = score / penalty if score > 0 else score * penalty
    return adjusted


def _banned_next_tokens(seq: torch.Tensor, ngram_size: int) -> set[int]:
    """Return banned next-token IDs to enforce no-repeat n-grams."""
    if seq.numel() < ngram_size - 1:
        return set()
    tokens = seq.tolist()
    prefix = tuple(tokens[-(ngram_size - 1):])
    banned: set[int] = set()
    for i in range(len(tokens) - ngram_size + 1):
        ng = tokens[i : i + ngram_size]
        if tuple(ng[:-1]) == prefix:
            banned.add(int(ng[-1]))
    return banned


def _apply_decoding_constraints(
    next_logits: torch.Tensor,
    input_ids: torch.Tensor,
    config: ReportGenerationConfig,
) -> torch.Tensor:
    """Apply repetition penalty, n-gram blocking, and immediate-repeat suppression."""
    B = next_logits.shape[0]

    penalty = float(getattr(config, "repetition_penalty", 1.0))
    if penalty > 1.0 and input_ids.shape[1] > 1:
        next_logits = _apply_repetition_penalty(next_logits, input_ids, penalty)

    ngram_size = max(0, int(config.no_repeat_ngram_size))
    if ngram_size > 1 and input_ids.shape[1] >= ngram_size - 1:
        for b in range(B):
            banned = _banned_next_tokens(input_ids[b], ngram_size)
            if banned:
                next_logits[b, list(banned)] = -1e9

    if input_ids.shape[1] > 1:
        last_token = input_ids[:, -1]
        next_logits.scatter_(1, last_token.unsqueeze(1), -1e9)

    return next_logits


# =====================================================================
# 1) TransformerReportDecoder — vanilla (backward-compatible)
# =====================================================================

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
        self._init_pretrained_embeddings(config)

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
        teacher_forcing_ratio: float = 1.0,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = target_ids.shape

        decode_ids = target_ids
        if (
            self.training
            and self.config.scheduled_sampling_enabled
            and 0.0 <= teacher_forcing_ratio < 1.0
            and L > 1
        ):
            decode_ids = self._scheduled_sampling_mix_ids(
                encoder_output=encoder_output,
                target_ids=target_ids,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

        tgt = self.token_embedding(decode_ids)
        tgt = self.pos_encoding(tgt)
        causal_mask = self._get_causal_mask(L, target_ids.device)

        decoded = self.decoder(
            tgt=tgt,
            memory=encoder_output,
            tgt_mask=causal_mask,
            memory_key_padding_mask=encoder_padding_mask,
        )
        logits = self.output_proj(decoded)
        return logits

    def _scheduled_sampling_mix_ids(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        with torch.no_grad():
            warm_logits = self._decode_from_ids(encoder_output, target_ids)
            pred_ids = warm_logits.argmax(dim=-1)

        keep_mask = torch.rand_like(target_ids.float()) < float(teacher_forcing_ratio)
        keep_mask[:, 0] = True
        mixed = torch.where(keep_mask, target_ids, pred_ids)
        return mixed

    def _decode_from_ids(
        self,
        encoder_output: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, L = token_ids.shape
        tgt = self.token_embedding(token_ids)
        tgt = self.pos_encoding(tgt)
        causal_mask = self._get_causal_mask(L, token_ids.device)
        decoded = self.decoder(
            tgt=tgt,
            memory=encoder_output,
            tgt_mask=causal_mask,
        )
        return self.output_proj(decoded)

    def generate(
        self,
        encoder_output: torch.Tensor,
        max_length: int = 128,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        B = encoder_output.shape[0]
        device = encoder_output.device
        eos_token_id = self.config.vocab_size - 1
        input_ids = torch.full((B, 1), eos_token_id, dtype=torch.long, device=device)
        generated: List[List[int]] = [[] for _ in range(B)]
        finished = [False] * B

        full_causal_mask = nn.Transformer.generate_square_subsequent_mask(
            max_length + 1, device=device
        )

        for step in range(max_length):
            seq_len = input_ids.shape[1]
            causal_mask = full_causal_mask[:seq_len, :seq_len]

            tgt = self.token_embedding(input_ids)
            tgt = self.pos_encoding(tgt)
            decoded = self.decoder(
                tgt=tgt,
                memory=encoder_output,
                tgt_mask=causal_mask,
                memory_key_padding_mask=encoder_padding_mask,
            )
            logits = self.output_proj(decoded)
            next_logits = logits[:, -1, :]

            next_logits = _apply_decoding_constraints(next_logits, input_ids, self.config)
            next_token = next_logits.argmax(dim=-1)

            for b in range(B):
                if not finished[b]:
                    tok = next_token[b].item()
                    if tok == eos_token_id:
                        finished[b] = True
                    else:
                        generated[b].append(tok)

            if all(finished):
                break
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        return generated

    def get_vocab_size(self) -> int:
        return self.config.vocab_size

    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask_cache is None or self._causal_mask_cache.shape[0] < size:
            mask = nn.Transformer.generate_square_subsequent_mask(size, device=device)
            self._causal_mask_cache = mask
            return mask
        return self._causal_mask_cache[:size, :size].to(device)

    def _init_pretrained_embeddings(self, config: ReportGenerationConfig) -> None:
        if config.vocab_size != 50257 or config.decoder_dim != 768:
            return
        try:
            from transformers import GPT2Model
            gpt2 = GPT2Model.from_pretrained("gpt2")
            self.token_embedding.weight.data.copy_(gpt2.wte.weight.data)
            del gpt2
            logger.info("Initialized decoder token embeddings from GPT-2 pretrained weights.")
        except Exception as exc:
            logger.warning("Could not load GPT-2 pretrained embeddings: %s", exc)


# =====================================================================
# 2) GPT2ReportDecoder — pretrained GPT-2 with cross-attention injection
# =====================================================================

class _CrossAttentionBlock(nn.Module):
    """Single cross-attention bridge inserted after a GPT-2 block.

    Pre-norm → MultiHeadAttention(Q=hidden, K/V=encoder) → residual.
    Only this block's parameters are randomly initialised; everything
    else comes from pretrained GPT-2.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden
        h = self.ln(hidden)
        h, _ = self.cross_attn(
            query=h, key=encoder_output, value=encoder_output,
            key_padding_mask=encoder_padding_mask,
        )
        return residual + self.dropout(h)


class GPT2ReportDecoder(BaseDecoder):
    """GPT-2 backbone with injected cross-attention for encoder-decoder generation.

    Loads **all** pretrained GPT-2 weights (embeddings + 12 transformer
    blocks + final LayerNorm), then inserts a lightweight cross-attention
    block after every GPT-2 layer.  Only the cross-attention parameters
    are randomly initialised — the rest of the 124 M params already know
    English language and provide a strong prior even on ~1 k training
    samples.

    Parameters:
        config: Report generation configuration (``decoder_type="gpt2"``).
    """

    def __init__(self, config: ReportGenerationConfig) -> None:
        nn.Module.__init__(self)
        self.config = config

        from transformers import GPT2Model, GPT2Config  # type: ignore

        gpt2_config = GPT2Config.from_pretrained("gpt2")
        # Override dropout to match our config
        gpt2_config.resid_pdrop = config.decoder_dropout
        gpt2_config.attn_pdrop = config.decoder_dropout
        gpt2_config.embd_pdrop = config.decoder_dropout

        self.gpt2 = GPT2Model.from_pretrained("gpt2", config=gpt2_config)

        num_layers = gpt2_config.n_layer  # 12
        d_model = gpt2_config.n_embd  # 768
        n_heads = config.num_decoder_heads  # from our config

        # Inject cross-attention blocks (randomly initialised)
        self.cross_attention_blocks = nn.ModuleList([
            _CrossAttentionBlock(d_model, n_heads, dropout=config.decoder_dropout)
            for _ in range(num_layers)
        ])

        # Output head — tied to GPT-2 token embeddings
        self.output_proj = nn.Linear(d_model, config.vocab_size, bias=False)
        self.output_proj.weight = self.gpt2.wte.weight

        # Freeze pretrained GPT-2 backbone — only train cross-attention
        # bridges and the final LayerNorm.  This prevents overfitting on
        # small datasets while preserving the language prior.
        for param in self.gpt2.parameters():
            param.requires_grad = False
        # Unfreeze final layer norm so it can adapt to the new cross-attn residuals
        for param in self.gpt2.ln_f.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "GPT2ReportDecoder: loaded pretrained GPT-2 (%d layers, %d dim) "
            "+ %d cross-attention bridges.  Trainable: %s / %s params.",
            num_layers, d_model, num_layers,
            f"{trainable:,}", f"{total:,}",
        )

    # ---- forward (teacher-forced training) ----------------------------

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute next-token logits with teacher forcing.

        Args:
            encoder_output: ``(B, M, D)`` encoder memory (visual + KG).
            target_ids: ``(B, L)`` target token IDs.
            encoder_padding_mask: ``(B, M)`` bool — ``True`` for positions
                to ignore (zero-padded KG nodes).

        Returns:
            ``(B, L, vocab_size)`` logits.
        """
        B, L = target_ids.shape

        decode_ids = target_ids
        if (
            self.training
            and self.config.scheduled_sampling_enabled
            and 0.0 <= teacher_forcing_ratio < 1.0
            and L > 1
        ):
            with torch.no_grad():
                warm_logits = self._gpt2_forward(decode_ids, encoder_output, encoder_padding_mask)
                pred_ids = warm_logits.argmax(dim=-1)
            keep_mask = torch.rand_like(target_ids.float()) < float(teacher_forcing_ratio)
            keep_mask[:, 0] = True
            decode_ids = torch.where(keep_mask, target_ids, pred_ids)

        return self._gpt2_forward(decode_ids, encoder_output, encoder_padding_mask)

    def _gpt2_forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run GPT-2 blocks interleaved with cross-attention."""
        # GPT-2 embeddings (token + position)
        inputs_embeds = self.gpt2.wte(input_ids)
        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)
        hidden = inputs_embeds + self.gpt2.wpe(position_ids)
        hidden = self.gpt2.drop(hidden)

        # Run through each GPT-2 block + cross-attention
        for gpt2_block, xattn_block in zip(self.gpt2.h, self.cross_attention_blocks):
            hidden = gpt2_block(hidden)[0]  # GPT-2 self-attention + FFN
            hidden = xattn_block(hidden, encoder_output, encoder_padding_mask)  # cross-attention to encoder

        hidden = self.gpt2.ln_f(hidden)
        return self.output_proj(hidden)

    # ---- beam search generation ---------------------------------------

    def generate(
        self,
        encoder_output: torch.Tensor,
        max_length: int = 128,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Auto-regressive beam search decoding.

        Falls back to greedy if ``beam_size <= 1``.
        """
        self._gen_encoder_padding_mask = encoder_padding_mask
        beam_size = max(1, int(getattr(self.config, "beam_size", 1)))
        if beam_size <= 1:
            return self._greedy_generate(encoder_output, max_length)
        return self._beam_search_generate(encoder_output, max_length, beam_size)

    def _greedy_generate(
        self,
        encoder_output: torch.Tensor,
        max_length: int,
    ) -> List[List[int]]:
        B = encoder_output.shape[0]
        device = encoder_output.device
        eos_token_id = self.config.vocab_size - 1
        input_ids = torch.full((B, 1), eos_token_id, dtype=torch.long, device=device)
        generated: List[List[int]] = [[] for _ in range(B)]
        finished = [False] * B

        mask = getattr(self, '_gen_encoder_padding_mask', None)
        for _ in range(max_length):
            logits = self._gpt2_forward(input_ids, encoder_output, mask)
            next_logits = logits[:, -1, :]
            next_logits = _apply_decoding_constraints(next_logits, input_ids, self.config)
            next_token = next_logits.argmax(dim=-1)

            for b in range(B):
                if not finished[b]:
                    tok = next_token[b].item()
                    if tok == eos_token_id:
                        finished[b] = True
                    else:
                        generated[b].append(tok)

            if all(finished):
                break
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        return generated

    def _beam_search_generate(
        self,
        encoder_output: torch.Tensor,
        max_length: int,
        beam_size: int,
    ) -> List[List[int]]:
        """Beam search — processes each sample independently."""
        B = encoder_output.shape[0]
        device = encoder_output.device
        eos_token_id = self.config.vocab_size - 1
        length_penalty = float(getattr(self.config, "length_penalty", 1.0))

        all_generated: List[List[int]] = []

        mask = getattr(self, '_gen_encoder_padding_mask', None)
        for b in range(B):
            enc = encoder_output[b:b+1]  # (1, M, D)
            enc_beam = enc.expand(beam_size, -1, -1)  # (beam, M, D)
            b_mask = mask[b:b+1] if mask is not None else None

            # Each beam: (log_prob, token_ids_tensor)
            beams = [(0.0, torch.full((1, 1), eos_token_id, dtype=torch.long, device=device))]
            completed: list[tuple[float, list[int]]] = []

            for _ in range(max_length):
                candidates: list[tuple[float, torch.Tensor]] = []

                for score, seq in beams:
                    logits = self._gpt2_forward(seq, enc, b_mask)
                    next_logits = logits[:, -1, :]
                    next_logits = _apply_decoding_constraints(next_logits, seq, self.config)
                    log_probs = F.log_softmax(next_logits, dim=-1).squeeze(0)

                    topk_scores, topk_ids = log_probs.topk(beam_size)
                    for k in range(beam_size):
                        tok = topk_ids[k].item()
                        new_score = score + topk_scores[k].item()
                        new_seq = torch.cat(
                            [seq, topk_ids[k].view(1, 1)], dim=1
                        )
                        candidates.append((new_score, new_seq))

                # Rank and prune
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = []
                for cand_score, cand_seq in candidates:
                    if cand_seq[0, -1].item() == eos_token_id:
                        seq_len = max(cand_seq.shape[1] - 2, 1)  # exclude BOS/EOS
                        normalised = cand_score / (seq_len ** length_penalty)
                        tokens = cand_seq[0, 1:-1].tolist()  # exclude BOS and EOS
                        completed.append((normalised, tokens))
                    else:
                        beams.append((cand_score, cand_seq))
                    if len(beams) >= beam_size:
                        break

                if not beams:
                    break

            # Fallback: if no completed beams, take the best active beam
            if not completed:
                for score, seq in beams:
                    tokens = seq[0, 1:].tolist()  # exclude BOS
                    completed.append((score / max(len(tokens), 1), tokens))

            best_tokens = max(completed, key=lambda x: x[0])[1]
            # Filter out EOS tokens
            best_tokens = [t for t in best_tokens if t != eos_token_id]
            all_generated.append(best_tokens)

        return all_generated

    def get_vocab_size(self) -> int:
        return self.config.vocab_size

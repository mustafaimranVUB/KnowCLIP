"""Tests for generation target collation in MIMIC batch collation."""

from __future__ import annotations

import torch

from src.data import dataset as dataset_mod
from src.data.dataset import collate_mimic


def test_collate_mimic_emits_target_ids_with_fallback_tokenizer(monkeypatch):
    class FakeTokenizer:
        def __call__(self, texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt"):
            ids = torch.full((len(texts), max_length), 7, dtype=torch.long)
            mask = torch.ones((len(texts), max_length), dtype=torch.long)
            mask[:, -2:] = 0
            return {"input_ids": ids, "attention_mask": mask}

    monkeypatch.setattr(dataset_mod, "_REPORT_TOKENIZER", FakeTokenizer())

    batch = [
        {
            "image": torch.randn(3, 224, 224),
            "labels": torch.zeros(14),
            "report_text": "small right pleural effusion",
            "subject_id": 1,
            "study_id": 11,
            "study_key": "1_11",
        },
        {
            "image": torch.randn(3, 224, 224),
            "labels": torch.ones(14),
            "report_text": "no focal consolidation",
            "subject_id": 2,
            "study_id": 22,
            "study_key": "2_22",
        },
    ]

    out = collate_mimic(batch)

    assert "target_ids" in out
    assert "generation_targets" in out
    assert out["target_ids"].shape == (2, 127)
    assert out["target_ids"].dtype == torch.long
    assert out["generation_targets"].shape == (2, 127)
    assert out["generation_targets"].dtype == torch.long
    assert torch.all(out["generation_targets"][:, -2:] == -100)


def test_collate_bos_eos_with_real_tokenizer(monkeypatch):
    """Verify BOS is prepended and EOS is a valid target (not masked)."""
    try:
        from transformers import GPT2TokenizerFast
    except ImportError:
        import pytest
        pytest.skip("transformers not installed")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    monkeypatch.setattr(dataset_mod, "_REPORT_TOKENIZER", tok)

    batch = [
        {
            "image": torch.randn(3, 224, 224),
            "labels": torch.zeros(14),
            "report_text": "Heart size is normal.",
            "subject_id": 1,
            "study_id": 11,
            "study_key": "1_11",
        },
    ]

    out = collate_mimic(batch)
    target_ids = out["target_ids"]   # (1, 127)
    gen_targets = out["generation_targets"]  # (1, 127)

    eos_id = tok.eos_token_id  # 50256

    # target_ids must start with BOS (same as EOS for GPT-2)
    assert target_ids[0, 0].item() == eos_id

    # The first generation target must be a real content token, not -100
    assert gen_targets[0, 0].item() != -100

    # There must be at least one EOS in gen_targets (model learns to stop)
    real_targets = gen_targets[0][gen_targets[0] != -100]
    assert eos_id in real_targets.tolist()

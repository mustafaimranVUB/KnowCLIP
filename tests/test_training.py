"""Tests for loss functions and training utilities."""

from __future__ import annotations

import pytest
import torch

from src.training.losses import MultiTaskLoss
from src.training.scheduler import get_linear_warmup_cosine_decay


class TestMultiTaskLoss:
    def test_classification_only(self):
        loss_fn = MultiTaskLoss(classification_weight=1.0, generation_weight=0.0)
        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        losses = loss_fn(
            classification_logits=logits,
            classification_targets=targets,
        )

        assert "total" in losses
        assert "classification" in losses
        assert "generation" in losses
        assert torch.isfinite(losses["total"])
        assert losses["generation"].item() == 0.0

    def test_generation_only(self):
        loss_fn = MultiTaskLoss(classification_weight=0.0, generation_weight=1.0)
        gen_logits = torch.randn(4, 32, 1000)
        gen_targets = torch.randint(0, 1000, (4, 32))

        losses = loss_fn(
            generation_logits=gen_logits,
            generation_targets=gen_targets,
        )

        assert torch.isfinite(losses["total"])
        assert losses["classification"].item() == 0.0

    def test_combined_loss(self):
        loss_fn = MultiTaskLoss(classification_weight=1.0, generation_weight=1.0)

        cls_logits = torch.randn(4, 14)
        cls_targets = torch.randint(0, 2, (4, 14)).float()
        gen_logits = torch.randn(4, 32, 1000)
        gen_targets = torch.randint(0, 1000, (4, 32))

        losses = loss_fn(
            classification_logits=cls_logits,
            classification_targets=cls_targets,
            generation_logits=gen_logits,
            generation_targets=gen_targets,
        )

        expected_total = losses["classification"] + losses["generation"]
        assert torch.isclose(losses["total"], expected_total, atol=1e-5)

    def test_nan_label_masking(self):
        loss_fn = MultiTaskLoss()
        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()
        targets[0, :] = float("nan")  # First sample all NaN

        losses = loss_fn(
            classification_logits=logits,
            classification_targets=targets,
        )

        # Should not produce NaN
        assert torch.isfinite(losses["total"])

    def test_all_nan_labels(self):
        loss_fn = MultiTaskLoss()
        logits = torch.randn(4, 14)
        targets = torch.full((4, 14), float("nan"))

        losses = loss_fn(
            classification_logits=logits,
            classification_targets=targets,
        )

        # Should return 0 loss when all labels are NaN
        assert losses["classification"].item() == 0.0

    def test_loss_weights(self):
        loss_fn = MultiTaskLoss(classification_weight=2.0, generation_weight=0.5)

        cls_logits = torch.randn(4, 14)
        cls_targets = torch.randint(0, 2, (4, 14)).float()
        gen_logits = torch.randn(4, 32, 1000)
        gen_targets = torch.randint(0, 1000, (4, 32))

        losses = loss_fn(
            classification_logits=cls_logits,
            classification_targets=cls_targets,
            generation_logits=gen_logits,
            generation_targets=gen_targets,
        )

        expected = 2.0 * losses["classification"] + 0.5 * losses["generation"]
        assert torch.isclose(losses["total"], expected, atol=1e-5)


class TestScheduler:
    def test_warmup_lr_increases(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = get_linear_warmup_cosine_decay(
            optimizer, warmup_steps=10, total_steps=100,
        )

        lrs = []
        for _ in range(15):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # During warmup, LR should be increasing
        assert lrs[5] > lrs[0]
        # After warmup, LR should start decreasing (cosine)
        assert lrs[14] <= lrs[10]

    def test_zero_warmup(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = get_linear_warmup_cosine_decay(
            optimizer, warmup_steps=0, total_steps=100,
        )

        # Should not crash with 0 warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()

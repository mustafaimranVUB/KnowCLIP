"""Tests for loss functions and training utilities."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.core.config import ProjectConfig
from src.training.losses import MultiTaskLoss
from src.training.scheduler import get_linear_warmup_cosine_decay
from src.training.trainer import Trainer, _safe_torch_save


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

    def test_uncertain_minus_one_labels_masked(self):
        loss_fn = MultiTaskLoss()
        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()
        targets[0, :3] = -1.0

        losses = loss_fn(
            classification_logits=logits,
            classification_targets=targets,
        )

        assert torch.isfinite(losses["total"])

    def test_focal_loss_path(self):
        loss_fn = MultiTaskLoss(
            classification_loss_type="focal",
            focal_gamma=2.0,
            focal_alpha=0.25,
        )
        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        losses = loss_fn(
            classification_logits=logits,
            classification_targets=targets,
        )
        assert torch.isfinite(losses["classification"])

    def test_pos_weight_path(self):
        pos_weight = torch.linspace(1.0, 5.0, steps=14)
        loss_fn = MultiTaskLoss(class_pos_weight=pos_weight)
        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        losses = loss_fn(
            classification_logits=logits,
            classification_targets=targets,
        )
        assert torch.isfinite(losses["classification"])

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


class TestCheckpointSave:
    def test_safe_torch_save_success(self, tmp_path: Path):
        payload = {"a": torch.tensor([1, 2, 3])}
        out_path = tmp_path / "ckpt.pt"

        ok = _safe_torch_save(payload, out_path)

        assert ok is True
        assert out_path.exists()

    def test_safe_torch_save_handles_io_failure(self, monkeypatch, tmp_path: Path):
        class _FailingIOError(RuntimeError):
            errno = errno.ENOSPC

        def _boom(*args, **kwargs):
            raise _FailingIOError("file write failed")

        monkeypatch.setattr(torch, "save", _boom)
        out_path = tmp_path / "ckpt.pt"

        ok = _safe_torch_save({"x": 1}, out_path)

        assert ok is False
        assert not out_path.exists()


class _DummyDataset(Dataset):
    def __init__(self, n: int = 4) -> None:
        self.samples = [
            {"labels": torch.randint(0, 2, (14,), dtype=torch.float32)}
            for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return {
            "image": torch.zeros(3, 224, 224),
            "labels": self.samples[idx]["labels"],
            "target_ids": torch.randint(0, 10, (8,), dtype=torch.long),
            "generation_targets": torch.randint(0, 10, (8,), dtype=torch.long),
        }


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.tensor(0.0))

    def forward(self, **kwargs):
        pixel_values = kwargs["pixel_values"]
        target_ids = kwargs.get("target_ids")
        b = pixel_values.shape[0]
        cls = self.param + torch.zeros(b, 14, device=pixel_values.device)
        out = {"classification_logits": cls}
        if target_ids is not None:
            l = target_ids.shape[1]
            gen = self.param + torch.zeros(b, l, 10, device=pixel_values.device)
            out["generation_logits"] = gen
        return out


class TestTrainerNewControls:
    def _build_trainer(self) -> Trainer:
        cfg = ProjectConfig()
        cfg.training.use_pos_weight = False
        cfg.training.mixed_precision = False
        cfg.training.log_dir = Path("outputs/test_logs")
        cfg.training.checkpoint_dir = Path("outputs/test_ckpts")
        cfg.training.num_epochs = 2
        cfg.training.warmup_steps = 0
        cfg.training.accumulation_steps = 1
        cfg.model.report_generation.scheduled_sampling_enabled = True
        cfg.model.report_generation.scheduled_sampling_start_ratio = 1.0
        cfg.model.report_generation.scheduled_sampling_end_ratio = 0.5
        cfg.model.report_generation.scheduled_sampling_decay_epochs = 10

        ds = _DummyDataset()
        loader = DataLoader(ds, batch_size=2)
        model = _DummyModel()
        return Trainer(model=model, config=cfg, train_loader=loader, val_loader=loader)

    def test_teacher_forcing_schedule_decreases(self):
        tr = self._build_trainer()
        r0 = tr._teacher_forcing_ratio_for_epoch(0)
        r5 = tr._teacher_forcing_ratio_for_epoch(5)
        r10 = tr._teacher_forcing_ratio_for_epoch(10)
        assert r0 == pytest.approx(1.0)
        assert r5 < r0
        assert r10 == pytest.approx(0.5)

    def test_classification_priority_selection_metric(self):
        tr = self._build_trainer()
        tr.tc.checkpoint_selection_mode = "classification_priority"
        val_stats = {
            "total": 10.0,
            "classification": 2.0,
            "weighted_classification": 1.25,
            "generation": 8.0,
        }
        assert tr._selection_metric_from_val(val_stats) == pytest.approx(1.25)

    def test_generation_guard(self):
        tr = self._build_trainer()
        tr.tc.checkpoint_generation_guard_max_val_gen_loss = 3.0
        assert tr._passes_generation_guard({"weighted_generation": 2.5}) is True
        assert tr._passes_generation_guard({"weighted_generation": 3.5}) is False

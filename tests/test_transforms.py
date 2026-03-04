"""Tests for data transforms and utilities."""

from __future__ import annotations

import pytest
import torch

from src.data.transforms import get_eval_transforms, get_train_transforms, denormalise


class TestTransforms:
    def test_train_transforms_exist(self):
        t = get_train_transforms()
        assert t is not None

    def test_eval_transforms_exist(self):
        t = get_eval_transforms()
        assert t is not None

    def test_eval_transforms_deterministic(self):
        t = get_eval_transforms()
        # Create a fake PIL image
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        out1 = t(img)
        out2 = t(img)
        assert torch.allclose(out1, out2)

    def test_eval_output_shape(self):
        t = get_eval_transforms()
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 224, 224)


class TestDenormalise:
    def test_roundtrip(self):
        """Denormalise should approximately invert normalization."""
        from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
        # Create a tensor in [0, 1]
        t = torch.rand(3, 32, 32)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        normalized = (t - mean) / std

        recovered = denormalise(normalized)
        assert torch.allclose(recovered, t, atol=1e-5)

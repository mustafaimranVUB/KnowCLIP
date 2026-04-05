"""Image transforms / augmentations for training and evaluation.

Uses ``torchvision.transforms.v2`` where available, with a
``torchvision.transforms`` fall-back.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision import transforms  # type: ignore

# CLIP-family normalisation (ImageNet stats)
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


def get_train_transforms(
    image_size: int = 224,
    *,
    random_crop: bool = True,
    horizontal_flip: bool = False,
    color_jitter: bool = False,
) -> transforms.Compose:
    """Build training-time image transforms.

    The default configuration is conservative for medical images:
    no horizontal flip (left-right matters clinically) and no colour
    jitter.

    Args:
        image_size: Target spatial resolution.
        random_crop: Use random resized crop (else centre crop).
        horizontal_flip: Enable random horizontal flip.
        color_jitter: Enable random colour jitter.

    Returns:
        ``torchvision.transforms.Compose`` pipeline.
    """
    tfms: list = []

    if random_crop:
        tfms.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
    else:
        tfms.append(transforms.Resize(int(image_size * 256 / 224)))
        tfms.append(transforms.CenterCrop(image_size))

    if horizontal_flip:
        tfms.append(transforms.RandomHorizontalFlip(p=0.5))

    if color_jitter:
        tfms.append(
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        )

    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(tfms)


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Build evaluation-time (deterministic) image transforms.

    Args:
        image_size: Target spatial resolution.

    Returns:
        ``torchvision.transforms.Compose`` pipeline.
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalise(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
) -> torch.Tensor:
    """Reverse ImageNet normalisation for visualisation.

    Args:
        tensor: (C, H, W) or (B, C, H, W) normalised tensor.
        mean: Per-channel mean used during normalisation.
        std: Per-channel std used during normalisation.

    Returns:
        Tensor with pixel values approximately in [0, 1].
    """
    m = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
    s = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)

    if tensor.ndim == 4:
        m = m.view(1, 3, 1, 1)
        s = s.view(1, 3, 1, 1)
    elif tensor.ndim == 3:
        m = m.view(3, 1, 1)
        s = s.view(3, 1, 1)

    return tensor * s + m

"""Learning rate schedulers."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_warmup_cosine_decay(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Linear warm-up followed by cosine decay to ``min_lr_ratio * base_lr``.

    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Steps for linear warm-up.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as a fraction of base LR.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)

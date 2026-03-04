"""Core utility functions for reproducibility, device management, and logging."""

from __future__ import annotations

import logging
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seed for full reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device.

    Args:
        prefer_cuda: If True, prefer CUDA over CPU.

    Returns:
        torch.device for computation.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    experiment_name: str = "knoclip",
) -> logging.Logger:
    """Configure structured logging to file and console.

    Args:
        log_dir: Directory for log files. If None, console only.
        level: Logging level.
        experiment_name: Name for the logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"{experiment_name}_{timestamp}.log")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

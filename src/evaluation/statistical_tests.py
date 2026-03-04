"""Statistical significance testing for model comparison.

Implements:
- McNemar's test (paired error-rate comparison)
- Bootstrap confidence intervals (any metric)
- Wilcoxon signed-rank test (paired per-sample scores)
- Bonferroni correction
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> Dict[str, float]:
    """McNemar's test comparing two classifiers on the same test set.

    Tests whether models A and B have significantly different error rates.

    Args:
        y_true: ``(N,)`` ground-truth labels.
        preds_a: ``(N,)`` predictions from model A.
        preds_b: ``(N,)`` predictions from model B.

    Returns:
        Dict with ``statistic``, ``p_value``, ``n_01``, ``n_10``.
    """
    errors_a = (preds_a != y_true).astype(int)
    errors_b = (preds_b != y_true).astype(int)

    # Contingency: A wrong & B right (n_10), A right & B wrong (n_01)
    n_10 = np.sum((errors_a == 1) & (errors_b == 0))
    n_01 = np.sum((errors_a == 0) & (errors_b == 1))

    # Continuity-corrected McNemar
    if n_10 + n_01 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_01": int(n_01), "n_10": int(n_10)}

    statistic = (abs(n_10 - n_01) - 1) ** 2 / max(n_10 + n_01, 1)

    from scipy.stats import chi2  # type: ignore

    p_value = float(1 - chi2.cdf(statistic, df=1))

    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "n_01": int(n_01),
        "n_10": int(n_10),
    }


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for any metric.

    Args:
        metric_fn: Function ``(y_true, y_score) → scalar``.
        y_true: Ground-truth values.
        y_score: Predictions / scores.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with ``point_estimate``, ``ci_lower``, ``ci_upper``.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point_estimate = float(metric_fn(y_true, y_score))

    bootstrap_scores = []
    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        try:
            score = metric_fn(y_true[indices], y_score[indices])
            bootstrap_scores.append(score)
        except (ValueError, ZeroDivisionError):
            continue

    if not bootstrap_scores:
        return {
            "point_estimate": point_estimate,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def wilcoxon_signed_rank(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Dict[str, float]:
    """Wilcoxon signed-rank test for paired per-sample scores.

    Args:
        scores_a: Per-sample metric scores from model A.
        scores_b: Per-sample metric scores from model B.

    Returns:
        Dict with ``statistic`` and ``p_value``.
    """
    from scipy.stats import wilcoxon  # type: ignore

    diff = scores_a - scores_b
    # Remove zeros (ties)
    nonzero = diff[diff != 0]

    if len(nonzero) < 10:
        logger.warning("Too few non-tied pairs (%d) for Wilcoxon test", len(nonzero))
        return {"statistic": 0.0, "p_value": 1.0}

    stat, p_value = wilcoxon(nonzero)
    return {"statistic": float(stat), "p_value": float(p_value)}


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Apply Bonferroni correction for multiple hypothesis testing.

    Args:
        p_values: List of p-values from individual tests.
        alpha: Family-wise error rate.

    Returns:
        List of dicts with ``original_p``, ``corrected_p``, ``significant``.
    """
    m = len(p_values)
    corrected_alpha = alpha / m

    results = []
    for p in p_values:
        corrected_p = min(p * m, 1.0)
        results.append({
            "original_p": p,
            "corrected_p": corrected_p,
            "significant": corrected_p < alpha,
            "corrected_alpha": corrected_alpha,
        })

    return results

"""Report-generation evaluation metrics: BLEU, ROUGE, BERTScore, F1-RadGraph."""

from __future__ import annotations

import logging
import numbers
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class GenerationEvaluator:
    """Evaluate generated radiology reports against references.

    Metrics:
        - BLEU (1–4)
        - ROUGE (1, 2, L)
        - BERTScore
        - F1-RadGraph (radiograph-specific clinical accuracy)
    """

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """Compute all generation metrics.

        Args:
            predictions: Generated report texts.
            references: Ground-truth report texts.

        Returns:
            Dict with metric names as keys.
        """
        results: Dict[str, Any] = {}

        results.update(self._compute_bleu(predictions, references))
        results.update(self._compute_rouge(predictions, references))

        # BERTScore (optional, may be slow)
        try:
            results.update(self._compute_bertscore(predictions, references))
        except ImportError:
            logger.warning("bert_score not installed, skipping BERTScore")

        # F1-RadGraph (optional)
        try:
            results.update(self._compute_f1_radgraph(predictions, references))
        except (ImportError, Exception) as exc:
            logger.warning("F1-RadGraph skipped: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bleu(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute corpus BLEU-1, 2, 4."""
        import nltk  # type: ignore
        from nltk.translate.bleu_score import (  # type: ignore
            corpus_bleu,
            SmoothingFunction,
        )

        # Tokenise
        refs_tok = [[ref.split()] for ref in references]
        hyps_tok = [pred.split() for pred in predictions]

        smooth = SmoothingFunction().method1

        bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu2 = corpus_bleu(refs_tok, hyps_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

        return {"bleu_1": bleu1, "bleu_2": bleu2, "bleu_4": bleu4}

    @staticmethod
    def _compute_rouge(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L (F-measure)."""
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        scores = {"rouge_1": [], "rouge_2": [], "rouge_l": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            scores["rouge_1"].append(s["rouge1"].fmeasure)
            scores["rouge_2"].append(s["rouge2"].fmeasure)
            scores["rouge_l"].append(s["rougeL"].fmeasure)

        return {
            k: sum(v) / max(len(v), 1)
            for k, v in scores.items()
        }

    @staticmethod
    def _compute_bertscore(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute BERTScore (precision, recall, F1)."""
        from bert_score import score as bert_score_fn  # type: ignore

        P, R, F1 = bert_score_fn(
            predictions,
            references,
            lang="en",
            verbose=False,
        )
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }

    @staticmethod
    def _compute_f1_radgraph(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute F1-RadGraph (radiology-specific entity/relation overlap).

        F1RadGraph.forward() returns::

            (mean_reward, reward_list, hyp_annotations, ref_annotations)

        With ``reward_level="all"``:
        - ``mean_reward`` is ``(simple, partial, complete)``
        - ``reward_list`` is ``([simple_per_sample], [partial_per_sample],
          [complete_per_sample])``

        We report the **partial** score as the primary ``f1_radgraph``
        metric (most commonly used in the CXR report-generation
        literature) and all three sub-scores for completeness.
        """
        from radgraph import F1RadGraph  # type: ignore

        scorer = F1RadGraph(reward_level="all")
        mean_reward, reward_list, _, _ = scorer(
            hyps=predictions,
            refs=references,
        )

        # reward_level="all" → mean_reward is (simple, partial, complete)
        if isinstance(mean_reward, (tuple, list)) and len(mean_reward) == 3:
            return {
                "f1_radgraph": float(mean_reward[1]),           # partial (primary)
                "f1_radgraph_simple": float(mean_reward[0]),
                "f1_radgraph_partial": float(mean_reward[1]),
                "f1_radgraph_complete": float(mean_reward[2]),
            }

        # Fallback for non-"all" reward levels (scalar mean_reward)
        return {"f1_radgraph": float(mean_reward)}

    @staticmethod
    def _flatten_numeric_dict(obj: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        """Flatten nested RadGraph metric dicts to numeric leaf values."""
        out: Dict[str, float] = {}
        for key, value in obj.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, numbers.Real):
                out[full_key] = float(value)
            elif isinstance(value, dict):
                out.update(GenerationEvaluator._flatten_numeric_dict(value, prefix=full_key))
        return out

    @staticmethod
    def _is_score_key(key: str) -> bool:
        lower = key.lower()
        if lower.endswith("start_ix") or lower.endswith("end_ix"):
            return False
        if "start_ix" in lower or "end_ix" in lower:
            return False
        return any(tok in lower for tok in ("f1", "precision", "recall", "score"))

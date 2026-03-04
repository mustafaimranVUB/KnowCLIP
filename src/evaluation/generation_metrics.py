"""Report-generation evaluation metrics: BLEU, ROUGE, BERTScore, F1-RadGraph."""

from __future__ import annotations

import logging
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
        """Compute F1-RadGraph (radiology-specific entity/relation overlap)."""
        from radgraph import F1RadGraph  # type: ignore

        scorer = F1RadGraph(reward_level="all")
        _, _, f1_scores, _ = scorer(
            hyps=predictions,
            refs=references,
        )
        mean_f1 = sum(f1_scores) / max(len(f1_scores), 1) if f1_scores else 0.0
        return {"f1_radgraph": mean_f1}

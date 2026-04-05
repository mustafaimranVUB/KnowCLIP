"""Tests for generation metric aggregation, including RadGraph dict outputs."""

from __future__ import annotations

import sys
import types

from src.evaluation.generation_metrics import GenerationEvaluator


class _FakeF1RadGraph:
    """Mimic the real F1RadGraph(reward_level="all") return signature.

    Real API returns::

        (mean_reward, reward_list, hyp_annotations, ref_annotations)

    With reward_level="all":
    - mean_reward = (simple_mean, partial_mean, complete_mean)
    - reward_list = ([simple_per_sample], [partial_per_sample], [complete_per_sample])
    """

    def __init__(self, reward_level: str = "all") -> None:
        self.reward_level = reward_level

    def __call__(self, hyps, refs):
        n = len(hyps)
        # Per-sample scores for each reward level
        simple_scores = [0.5] * n
        partial_scores = [0.7] * n
        complete_scores = [0.4] * n

        mean_reward = (0.5, 0.7, 0.4)
        reward_list = (simple_scores, partial_scores, complete_scores)
        hyp_annotations = [{"entities": {"1": {"start_ix": 0, "end_ix": 5}}}] * n
        ref_annotations = [{"entities": {"1": {"start_ix": 0, "end_ix": 5}}}] * n
        return mean_reward, reward_list, hyp_annotations, ref_annotations


def test_f1_radgraph_dict_outputs_are_aggregated(monkeypatch):
    fake_module = types.ModuleType("radgraph")
    fake_module.F1RadGraph = _FakeF1RadGraph
    monkeypatch.setitem(sys.modules, "radgraph", fake_module)

    evaluator = GenerationEvaluator()
    results = evaluator._compute_f1_radgraph(
        predictions=["a", "b"],
        references=["c", "d"],
    )

    assert "f1_radgraph" in results
    # Primary metric is the partial score (position 1 of mean_reward tuple)
    assert abs(results["f1_radgraph"] - 0.7) < 1e-8
    assert "f1_radgraph_simple" in results
    assert abs(results["f1_radgraph_simple"] - 0.5) < 1e-8
    assert "f1_radgraph_partial" in results
    assert abs(results["f1_radgraph_partial"] - 0.7) < 1e-8
    assert "f1_radgraph_complete" in results
    assert abs(results["f1_radgraph_complete"] - 0.4) < 1e-8

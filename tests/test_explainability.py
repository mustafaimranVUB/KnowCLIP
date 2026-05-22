"""Tests for explainability artefact export helpers."""

from __future__ import annotations

import torch

from src.evaluation.explainability import ExplainabilityExporter, ExplainabilitySample


def test_export_sample_creates_expected_files(tmp_path):
    exporter = ExplainabilityExporter(
        tmp_path,
        top_k_pathologies=3,
        top_k_concepts=2,
        top_k_edges=2,
        top_k_decoder_tokens=2,
    )

    explainability = {
        "pooling": {
            "weights": torch.tensor([[0.7, 0.3]], dtype=torch.float32),
            "mask": torch.tensor([[True, True]]),
        },
        "fusion": {
            "last_layer": torch.ones(1, 2, 2, 16, dtype=torch.float32),
            "per_layer": [torch.ones(1, 2, 2, 16, dtype=torch.float32)],
        },
        "graph": {
            "edge_attention_layers": [
                {
                    "layer_index": 0,
                    "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                    "alpha": torch.tensor([[0.8, 0.6], [0.4, 0.2]], dtype=torch.float32),
                    "edge_type": torch.tensor([0, 3], dtype=torch.long),
                    "attention_vector": torch.randn(1, 2, 8),
                }
            ]
        },
        "decoder": {
            "per_layer": [
                torch.rand(1, 2, 3, 18, dtype=torch.float32),
                torch.rand(1, 2, 3, 18, dtype=torch.float32),
            ]
        },
    }

    sample = ExplainabilitySample(
        study_key="100_200",
        image=torch.rand(3, 224, 224),
        explainability=explainability,
        class_names=["atelectasis", "cardiomegaly", "effusion", "edema"],
        classification_probs=torch.tensor([0.2, 0.9, 0.6, 0.1], dtype=torch.float32),
        reference_report="reference report",
        generated_report="generated report",
        generated_token_ids=[10, 11, 12],
        generated_token_labels=["mild", "pleural", "effusion"],
        graph_node_texts=["heart", "effusion"],
        graph_node_cuis=["C0018787", "C0014180"],
        graph_node_types=["anatomy", "observation"],
        graph_node_certainties=["definitely_present", "definitely_present"],
    )

    manifest = exporter.export_sample(sample)
    sample_dir = tmp_path / "100_200"

    assert manifest["study_key"] == "100_200"
    assert manifest["top_pathologies"][0]["label"] == "cardiomegaly"
    assert manifest["top_concepts"][0]["label"].startswith("heart")

    expected_files = [
        "input_image.png",
        "classification_scores.png",
        "concept_importance.png",
        "cross_attention_overlays.png",
        "graph_attention.png",
        "gat_attention_vector.png",
        "decoder_concepts.png",
        "decoder_visual_overlays.png",
        "summary.json",
        "summary.md",
        "trace_bundle.pt",
    ]
    for file_name in expected_files:
        assert (sample_dir / file_name).exists(), file_name
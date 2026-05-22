"""Test quantitative XAI metrics (concept ablation, deletion/insertion curves)."""

import torch
import pytest
from unittest.mock import MagicMock, patch

from src.evaluation.quantitative_xai_metrics import QuantitativeXAIEvaluator, QuantitativeXAIResults


class TestQuantitativeXAIMetrics:
    """Tests for quantitative explainability metrics."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        
        def mock_forward(pixel_values, graph_x=None, graph_edge_index=None, 
                        graph_edge_type=None, graph_batch=None, return_attention=False):
            batch_size = pixel_values.shape[0]
            output = {
                "classification_logits": torch.randn(batch_size, 14),
            }
            if return_attention:
                output["explainability"] = {
                    "pooling": {
                        "weights": torch.ones(batch_size, 10) / 10 if graph_x is not None else torch.ones(batch_size, 196) / 196,
                    }
                }
            return output
        
        model.side_effect = mock_forward
        return mock_forward

    @pytest.fixture
    def evaluator(self, mock_model):
        """Create quantitative XAI evaluator."""
        model_obj = MagicMock()
        model_obj.side_effect = mock_model
        
        return QuantitativeXAIEvaluator(
            model=model_obj,
            device=torch.device("cpu"),
            num_visual_ablation_steps=3,
        )

    def test_evaluate_baseline_only(self, evaluator, mock_model):
        """Test evaluation with visual-only (baseline) model."""
        batch_images = torch.randn(2, 3, 224, 224)
        batch_labels = torch.randint(0, 2, (2, 14), dtype=torch.float32)
        
        # Create mock model that returns proper outputs
        evaluator.model = MagicMock()
        evaluator.model.side_effect = mock_model
        
        explanations = {
            "pooling": {
                "weights": torch.ones(2, 196) / 196,
            }
        }

        result = evaluator.evaluate(
            batch_images=batch_images,
            batch_graphs=None,
            batch_graph_edge_index=None,
            batch_graph_edge_type=None,
            batch_graph_batch=None,
            batch_labels=batch_labels,
            label_index=0,
            explanations=explanations,
        )

        assert isinstance(result, QuantitativeXAIResults)
        assert isinstance(result.visual_ablation_auc_drop, float)
        assert isinstance(result.visual_deletion_auc, float)

    def test_evaluate_neuro_symbolic(self, evaluator, mock_model):
        """Test evaluation with KG-augmented model."""
        batch_images = torch.randn(2, 3, 224, 224)
        batch_graphs = torch.randn(10, 768)  # 10 total nodes
        batch_graph_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        batch_graph_edge_type = torch.zeros(3, dtype=torch.long)
        batch_graph_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        batch_labels = torch.randint(0, 2, (2, 14), dtype=torch.float32)

        # Create mock model
        evaluator.model = MagicMock()
        evaluator.model.side_effect = mock_model

        explanations = {
            "pooling": {
                "weights": torch.ones(2, 196) / 196,
            }
        }

        result = evaluator.evaluate(
            batch_images=batch_images,
            batch_graphs=batch_graphs,
            batch_graph_edge_index=batch_graph_edge_index,
            batch_graph_edge_type=batch_graph_edge_type,
            batch_graph_batch=batch_graph_batch,
            batch_labels=batch_labels,
            label_index=0,
            explanations=explanations,
        )

        assert isinstance(result, QuantitativeXAIResults)
        assert isinstance(result.visual_ablation_auc_drop, float)
        assert isinstance(result.visual_deletion_auc, float)
        assert len(result.visual_ablation_ranks) > 0

    def test_deletion_curve_properties(self, evaluator, mock_model):
        """Test that deletion curves have expected properties."""
        batch_images = torch.randn(1, 3, 224, 224)
        batch_labels = torch.tensor([[0.5] * 14], dtype=torch.float32)

        evaluator.model = MagicMock()
        evaluator.model.side_effect = mock_model

        explanations = {
            "pooling": {
                "weights": torch.ones(1, 196) / 196,
            }
        }

        result = evaluator.evaluate(
            batch_images=batch_images,
            batch_graphs=None,
            batch_graph_edge_index=None,
            batch_graph_edge_type=None,
            batch_graph_batch=None,
            batch_labels=batch_labels,
            label_index=0,
            explanations=explanations,
        )

        # Deletion curve should decrease as more patches are removed
        assert len(result.visual_deletion_curve_x) > 0
        assert len(result.visual_deletion_curve_y) == len(result.visual_deletion_curve_x)
        
        # X values should go from 0 to 1
        assert result.visual_deletion_curve_x[0] == 0.0
        assert result.visual_deletion_curve_x[-1] == 1.0

    def test_multiple_labels(self, evaluator, mock_model):
        """Test evaluation across multiple labels."""
        batch_images = torch.randn(2, 3, 224, 224)
        batch_labels = torch.randint(0, 2, (2, 14), dtype=torch.float32)

        evaluator.model = MagicMock()
        evaluator.model.side_effect = mock_model

        explanations = {
            "pooling": {
                "weights": torch.ones(2, 196) / 196,
            }
        }

        results = []
        for label_idx in range(3):  # Test 3 labels
            result = evaluator.evaluate(
                batch_images=batch_images,
                batch_graphs=None,
                batch_graph_edge_index=None,
                batch_graph_edge_type=None,
                batch_graph_batch=None,
                batch_labels=batch_labels,
                label_index=label_idx,
                explanations=explanations,
            )
            results.append(result)

        assert len(results) == 3
        assert all(isinstance(r, QuantitativeXAIResults) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

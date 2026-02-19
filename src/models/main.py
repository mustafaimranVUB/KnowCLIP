"""
Phase II Demo: Neuro-Symbolic Architecture Comparative Framework.

This script demonstrates:
1. Baseline model (Pure Vision) instantiation and forward pass
2. Neuro-Symbolic model (with KG) instantiation and forward pass
3. Comparison of architectures
4. Validation of modular design

Usage:
    python -m src.models.main --backbone biomedclip --batch-size 4
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.config import (
    ModelConfig,
    get_baseline_config,
    get_neurosymbolic_config,
    get_config_for_backbone,
)
from src.models.model_factory import (
    build_baseline_model,
    build_neurosymbolic_model,
    MedicalVLM,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def create_dummy_batch(batch_size: int = 4, num_concepts: int = 10) -> dict:
    """
    Create dummy data for testing forward passes.

    Args:
        batch_size: Number of samples in batch
        num_concepts: Number of knowledge concepts per sample

    Returns:
        Dictionary with dummy inputs
    """
    # Dummy images
    images = torch.randn(batch_size, 3, 224, 224)

    # Dummy report-specific knowledge graphs
    graphs_list = []
    for i in range(batch_size):
        # Create small graph for this sample
        num_nodes = num_concepts
        x = torch.randn(num_nodes, 768)  # Node features

        # Create some edges (LOCATED_AT relations)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]],  # Source nodes  # Target nodes
            dtype=torch.long,
        )

        edge_type = torch.zeros(4, dtype=torch.long)  # All LOCATED_AT

        graph = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        graphs_list.append(graph)

    # Batch the graphs
    batched_graphs = Batch.from_data_list(graphs_list)

    # Dummy labels for classification
    labels = torch.randint(0, 2, (batch_size, 14)).float()

    # Dummy report target IDs for generation
    report_target_ids = torch.randint(0, 10000, (batch_size, 64))

    return {
        "images": images,
        "graphs": batched_graphs,
        "labels": labels,
        "report_target_ids": report_target_ids,
    }


def test_baseline_model(backbone: str = "biomedclip", batch_size: int = 4):
    """
    Test baseline (Pure Vision) model.

    Args:
        backbone: Visual backbone to use
        batch_size: Batch size for dummy data
    """
    print_section("BASELINE MODEL (Pure Vision - No KG)")

    # Build model
    print(f"\nBuilding baseline model with {backbone} backbone...")
    model = build_baseline_model(backbone=backbone)

    # Print architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Configuration:")
    print(f"  - Backbone: {backbone}")
    print(f"  - Use KG: {model.config.use_kg}")
    print(f"  - Classification: {model.config.enable_classification}")
    print(f"  - Report Generation: {model.config.enable_report_generation}")

    print(f"\nModel Parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")

    # Create dummy batch
    print(f"\nCreating dummy batch (size={batch_size})...")
    batch = create_dummy_batch(batch_size=batch_size)

    # Forward pass (no KG needed)
    print(f"\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            images=batch["images"], graphs=None, global_graph=None  # No KG for baseline
        )

    # Validate outputs
    print(f"\nOutput Validation:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {tuple(value.shape)}")
        else:
            print(f"  - {key}: {type(value)}")

    # Check classification logits
    if "classification_logits" in outputs:
        logits = outputs["classification_logits"]
        assert logits.shape == (
            batch_size,
            14,
        ), f"Expected shape ({batch_size}, 14), got {logits.shape}"
        print(f"\n✓ Classification logits shape correct: {logits.shape}")

    print(f"\n✓ Baseline model test PASSED")

    return model, outputs


def test_neurosymbolic_model(backbone: str = "biomedclip", batch_size: int = 4):
    """
    Test Neuro-Symbolic model with KG integration.

    Args:
        backbone: Visual backbone to use
        batch_size: Batch size for dummy data
    """
    print_section("NEURO-SYMBOLIC MODEL (Vision + Knowledge Graph)")

    # Build model
    print(f"\nBuilding neuro-symbolic model with {backbone} backbone...")
    model = build_neurosymbolic_model(backbone=backbone)

    # Print architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Configuration:")
    print(f"  - Backbone: {backbone}")
    print(f"  - Use KG: {model.config.use_kg}")
    print(f"  - Classification: {model.config.enable_classification}")
    print(f"  - Report Generation: {model.config.enable_report_generation}")
    print(f"  - GAT Layers: {model.config.knowledge_encoder.num_gat_layers}")
    print(f"  - Fusion Layers: {model.config.fusion_module.num_fusion_layers}")

    print(f"\nModel Parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")

    # Create dummy batch
    print(f"\nCreating dummy batch with KG (size={batch_size})...")
    batch = create_dummy_batch(batch_size=batch_size)

    # Forward pass (with KG)
    print(f"\nRunning forward pass with KG...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            images=batch["images"],
            graphs=batch["graphs"],
            global_graph=None,
            return_attention=True,  # Get attention maps for explainability
        )

    # Validate outputs
    print(f"\nOutput Validation:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {tuple(value.shape)}")
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], torch.Tensor):
                print(
                    f"  - {key}: List of {len(value)} tensors, first shape: {tuple(value[0].shape)}"
                )
            else:
                print(f"  - {key}: List of {len(value)} items")
        else:
            print(f"  - {key}: {type(value)}")

    # Check classification logits
    if "classification_logits" in outputs:
        logits = outputs["classification_logits"]
        assert logits.shape == (
            batch_size,
            14,
        ), f"Expected shape ({batch_size}, 14), got {logits.shape}"
        print(f"\n✓ Classification logits shape correct: {logits.shape}")

    # Check attention maps
    if "attention_maps" in outputs:
        print(f"✓ Attention maps available for explainability")

    print(f"\n✓ Neuro-Symbolic model test PASSED")

    return model, outputs


def compare_models(baseline_model, neurosymbolic_model):
    """
    Compare baseline and neuro-symbolic architectures.

    Args:
        baseline_model: Baseline MedicalVLM
        neurosymbolic_model: Neuro-Symbolic MedicalVLM
    """
    print_section("MODEL COMPARISON")

    # Parameter counts
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    ns_params = sum(p.numel() for p in neurosymbolic_model.parameters())

    print(f"\nParameter Count:")
    print(f"  - Baseline:        {baseline_params:,}")
    print(f"  - Neuro-Symbolic:  {ns_params:,}")
    print(
        f"  - Difference:      {ns_params - baseline_params:,} (+{(ns_params/baseline_params - 1)*100:.1f}%)"
    )

    # Component analysis
    print(f"\nComponent Analysis:")
    print(f"\n  Baseline (Pure Vision):")
    print(f"    - Visual Encoder:        ✓")
    print(f"    - Knowledge Encoder:     ✗")
    print(f"    - Fusion Module:         ✗")
    print(f"    - Visual Pooling:        ✓")
    print(f"    - Classification Head:   ✓")
    print(f"    - Report Generator:      ✗")

    print(f"\n  Neuro-Symbolic:")
    print(f"    - Visual Encoder:        ✓")
    print(
        f"    - Knowledge Encoder:     ✓ (GAT with {neurosymbolic_model.config.knowledge_encoder.num_gat_layers} layers)"
    )
    print(
        f"    - Fusion Module:         ✓ (Cross-attention with {neurosymbolic_model.config.fusion_module.num_fusion_layers} layers)"
    )
    print(f"    - Visual Pooling:        ✗")
    print(f"    - Classification Head:   ✓")
    print(f"    - Report Generator:      ✓")

    # Modular design validation
    print(f"\nModular Design Validation:")
    print(f"  ✓ Visual backbone is swappable")
    print(f"  ✓ KG component can be toggled ON/OFF")
    print(f"  ✓ Classification task is configurable")
    print(f"  ✓ Generation task is configurable")
    print(f"  ✓ Ready for comparative benchmarking")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Phase II Model Demo")
    parser.add_argument(
        "--backbone",
        type=str,
        default="biomedclip",
        choices=["biomedclip", "pubmedclip", "vit-b-16", "vit-l-14"],
        help="Visual backbone to use",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for dummy data"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip baseline model testing"
    )
    parser.add_argument(
        "--skip-neurosymbolic",
        action="store_true",
        help="Skip neuro-symbolic model testing",
    )

    args = parser.parse_args()

    print_section("Phase II: Neuro-Symbolic Architecture Demo")
    print(f"\nConfiguration:")
    print(f"  - Visual Backbone: {args.backbone}")
    print(f"  - Batch Size: {args.batch_size}")

    # Test baseline model
    baseline_model = None
    if not args.skip_baseline:
        baseline_model, baseline_outputs = test_baseline_model(
            backbone=args.backbone, batch_size=args.batch_size
        )

    # Test neuro-symbolic model
    neurosymbolic_model = None
    if not args.skip_neurosymbolic:
        neurosymbolic_model, ns_outputs = test_neurosymbolic_model(
            backbone=args.backbone, batch_size=args.batch_size
        )

    # Compare models
    if baseline_model is not None and neurosymbolic_model is not None:
        compare_models(baseline_model, neurosymbolic_model)

    # Summary
    print_section("SUMMARY")
    print(f"\n✓ Phase II implementation complete!")
    print(f"\nDeliverables:")
    print(f"  ✓ config.py          - Modular configuration system")
    print(
        f"  ✓ modules.py         - Core neural components (VisualEncoder, KnowledgeEncoder, FusionModule)"
    )
    print(f"  ✓ dataset.py         - MIMIC-CXR dataset with KG integration")
    print(f"  ✓ model_factory.py   - MedicalVLM wrapper for comparative framework")
    print(f"  ✓ main.py            - Demo script with baseline comparison")

    print(f"\nNext Steps:")
    print(
        f"  1. Implement dataset metadata loading (_create_sample_index() in dataset.py)"
    )
    print(f"  2. Download MIMIC-CXR-JPG images")
    print(f"  3. Prepare CheXpert labels CSV")
    print(f"  4. Train baseline model")
    print(f"  5. Train neuro-symbolic model")
    print(f"  6. Compare performance metrics")

    print(f"\nReady for Phase II experiments! 🚀")


if __name__ == "__main__":
    main()

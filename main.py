#!/usr/bin/env python3
"""KnoCLIP-XAI — Main CLI entry point.

Usage
-----
.. code-block:: bash

    # Phase I — Build knowledge graph
    python main.py phase1 --config configs/phase1_kg.yaml

    # Phase II — Train baseline
    python main.py train --config configs/phase2_baseline.yaml

    # Phase II — Train neuro-symbolic
    python main.py train --config configs/phase2_neurosymbolic.yaml

    # Evaluate a checkpoint
    python main.py evaluate --config configs/phase2_neurosymbolic.yaml \\
        --checkpoint outputs/checkpoints/best_model.pt

    # Validate installation (no GPU / data needed)
    python main.py validate
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Redirect HuggingFace cache away from $HOME (limited quota on HPC).
# Falls back to $VSC_SCRATCH/hf_cache if available, otherwise outputs/hf_cache.
_vsc_scratch = os.environ.get("VSC_SCRATCH", "")
_hf_cache_default = os.path.join(_vsc_scratch, "hf_cache") if _vsc_scratch else "outputs/hf_cache"
_hf_cache = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or _hf_cache_default
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _hf_cache)
# Avoid deprecated cache env var warnings in transformers>=4.57.
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
Path(_hf_cache).mkdir(parents=True, exist_ok=True)

import torch

from src.core.config import ProjectConfig, load_config
from src.core.utils import get_device, set_seed, setup_logging

logger = logging.getLogger("knoclip")


# ======================================================================
# CLI
# ======================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="KnoCLIP-XAI",
        description="Knowledge-Grounded CLIP for Explainable Chest X-Ray Analysis",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: use built-in defaults).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g. 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    sub = parser.add_subparsers(dest="command", help="Pipeline to run.")

    # ---- phase1 ----
    p1 = sub.add_parser("phase1", help="Phase I: Knowledge graph construction.")
    p1.add_argument("--batch-size", type=int, default=32)
    p1.add_argument("--skip-extraction", action="store_true")
    p1.add_argument("--skip-grounding", action="store_true")
    p1.add_argument("--skip-embedding", action="store_true")
    p1.add_argument(
        "--max-studies",
        type=int,
        default=None,
        help="Cap the number of studies to process (dev / smoke-test).",
    )

    # ---- train ----
    tr = sub.add_parser("train", help="Phase II: Model training.")
    tr.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from.")
    tr.add_argument("--no-eval", action="store_true", help="Skip evaluation after training.")

    # ---- evaluate ----
    ev = sub.add_parser("evaluate", help="Evaluate a trained model.")
    ev.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")

    # ---- validate ----
    sub.add_parser("validate", help="Validate installation on CPU with dummy data.")

    return parser.parse_args(argv)


# ======================================================================
# Commands
# ======================================================================


def cmd_phase1(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Execute Phase I — Knowledge Graph Construction."""
    from src.pipelines.hybrid_kg_pipeline import HybridKGPipeline

    pipeline = HybridKGPipeline(config=config, device=args.device)
    max_studies = getattr(args, "max_studies", None)
    summary = pipeline.run(
        batch_size=args.batch_size,
        skip_extraction=args.skip_extraction,
        skip_grounding=args.skip_grounding,
        skip_embedding=args.skip_embedding,
        max_studies=max_studies,
    )
    logger.info("Phase I complete. Summary:\n%s", summary)


def cmd_train(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Execute Phase II — Model Training."""
    from src.pipelines.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline(config=config, device=args.device)
    results = pipeline.run(
        resume_from=args.resume,
        evaluate_after=not args.no_eval,
    )
    logger.info("Training complete. Results:\n%s", results)


def cmd_evaluate(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Evaluate a trained model on the test set."""
    from src.pipelines.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline(config=config, device=args.device)
    pipeline.setup()

    train_loader, val_loader, test_loader = pipeline.build_dataloaders()
    results = pipeline.evaluate(test_loader, checkpoint_path=args.checkpoint)
    logger.info("Evaluation results:\n%s", results)


def cmd_validate(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Validate installation with dummy data on CPU.

    Uses a lightweight mock visual encoder to avoid HuggingFace downloads.

    Checks:
    - All imports succeed
    - Model builds and forward-passes with random tensors
    - Graph builder produces valid PyG Data
    - Loss computes without error
    - Evaluation metrics run on toy data
    """
    import torch.nn as nn
    from unittest.mock import patch

    device = torch.device("cpu")
    logger.info("=" * 60)
    logger.info("VALIDATION MODE — running on CPU with dummy data")
    logger.info("=" * 60)

    errors: list[str] = []

    # Lightweight mock visual encoder (avoids HuggingFace downloads)
    class _MockVisualEncoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self._conv = nn.Conv2d(3, cfg.output_dim, kernel_size=16, stride=16)

        def forward(self, pixel_values):
            h = self._conv(pixel_values)
            return h.flatten(2).transpose(1, 2)

        def get_output_dim(self):
            return self.config.output_dim

        def get_num_patches(self):
            return self.config.num_patches

    # 1. Import check
    logger.info("[1/7] Checking imports …")
    try:
        from src.core.config import get_baseline_config, get_neurosymbolic_config  # noqa: F401
        from src.data.transforms import get_eval_transforms, get_train_transforms  # noqa: F401
        from src.knowledge.extraction import EntityExtractor, EntityType  # noqa: F401
        from src.knowledge.graph_builder import KnowledgeGraphBuilder, validate_graph  # noqa: F401
        from src.models.model_factory import build_model  # noqa: F401
        from src.training.losses import MultiTaskLoss  # noqa: F401
        from src.evaluation.classification_metrics import ClassificationEvaluator  # noqa: F401
        from src.evaluation.generation_metrics import GenerationEvaluator  # noqa: F401
        logger.info("  ✓ All imports succeeded.")
    except ImportError as e:
        msg = f"Import failed: {e}"
        errors.append(msg)
        logger.error("  ✗ %s", msg)

    # 2. Baseline model forward pass
    logger.info("[2/7] Baseline model forward pass …")
    try:
        from src.core.config import get_baseline_config
        from src.models.model_factory import build_model

        with patch("src.models.visual_encoder.CLIPVisualEncoder", _MockVisualEncoder):
            baseline_cfg = get_baseline_config()
            baseline = build_model(baseline_cfg).to(device)

            dummy_img = torch.randn(2, 3, 224, 224, device=device)
            with torch.no_grad():
                out = baseline(pixel_values=dummy_img)

        assert "classification_logits" in out, "Missing classification_logits"
        assert out["classification_logits"].shape == (2, 14), (
            f"Wrong shape: {out['classification_logits'].shape}"
        )
        logger.info("  ✓ Baseline forward pass OK. Output shape: %s", out["classification_logits"].shape)
    except Exception as e:
        msg = f"Baseline forward pass failed: {e}"
        errors.append(msg)
        logger.error("  ✗ %s", msg)

    # 3. Neuro-symbolic model forward pass
    logger.info("[3/7] Neuro-symbolic model forward pass …")
    try:
        from src.core.config import get_neurosymbolic_config
        from src.models.model_factory import build_model

        with patch("src.models.visual_encoder.CLIPVisualEncoder", _MockVisualEncoder):
            ns_cfg = get_neurosymbolic_config()
            ns_model = build_model(ns_cfg).to(device)

        dummy_img = torch.randn(2, 3, 224, 224, device=device)
        # Dummy graph: 6 nodes, some edges
        graph_x = torch.randn(6, 768, device=device)
        graph_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device)
        graph_edge_type = torch.tensor([0, 1, 2, 3], device=device)
        graph_batch = torch.tensor([0, 0, 0, 1, 1, 1], device=device)

        with torch.no_grad():
            out = ns_model(
                pixel_values=dummy_img,
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,
                graph_edge_type=graph_edge_type,
                graph_batch=graph_batch,
            )

        assert "classification_logits" in out
        assert out["classification_logits"].shape == (2, 14)
        logger.info(
            "  ✓ Neuro-symbolic forward pass OK. Keys: %s",
            list(out.keys()),
        )
    except Exception as e:
        msg = f"Neuro-symbolic forward pass failed: {e}"
        errors.append(msg)
        logger.error("  ✗ %s", msg)

    # 4. Graph builder
    logger.info("[4/7] Graph builder …")
    try:
        from src.knowledge.extraction import (
            ExtractedEntity,
            ExtractedTriple,
            EntityType,
            Certainty,
            RelationType,
        )
        from src.knowledge.graph_builder import KnowledgeGraphBuilder, validate_graph

        builder = KnowledgeGraphBuilder(node_dim=768, add_self_loops=True)

        ents = [
            ExtractedEntity(text="heart", entity_type=EntityType.ANATOMY, certainty=Certainty.DEFINITELY_PRESENT, start_ix=0, end_ix=5),
            ExtractedEntity(text="cardiomegaly", entity_type=EntityType.OBSERVATION, certainty=Certainty.DEFINITELY_PRESENT, start_ix=6, end_ix=18),
            ExtractedEntity(text="lungs", entity_type=EntityType.ANATOMY, certainty=Certainty.DEFINITELY_PRESENT, start_ix=20, end_ix=25),
            ExtractedEntity(text="opacity", entity_type=EntityType.OBSERVATION, certainty=Certainty.UNCERTAIN, start_ix=26, end_ix=33),
        ]
        triples = [
            ExtractedTriple(head_text="cardiomegaly", relation=RelationType.LOCATED_AT, tail_text="heart"),
            ExtractedTriple(head_text="opacity", relation=RelationType.LOCATED_AT, tail_text="lungs"),
            ExtractedTriple(head_text="cardiomegaly", relation=RelationType.ASSOCIATED_WITH, tail_text="opacity"),
        ]

        graph = builder.build_from_extraction(entities=ents, triples=triples)
        val_report = validate_graph(graph)

        assert val_report.get("valid", False), f"Graph invalid: {val_report}"
        logger.info("  ✓ Graph builder OK. Nodes: %d, Edges: %d", graph.x.shape[0], graph.edge_index.shape[1])
    except Exception as e:
        msg = f"Graph builder failed: {e}"
        errors.append(msg)
        logger.error("  ✗ %s", msg)

    # 5. Loss computation
    logger.info("[5/7] Loss computation …")
    try:
        from src.training.losses import MultiTaskLoss

        loss_fn = MultiTaskLoss(classification_weight=1.0, generation_weight=1.0)

        cls_logits = torch.randn(4, 14)
        cls_labels = torch.randint(0, 2, (4, 14)).float()
        gen_logits = torch.randn(4, 32, 50257)
        gen_targets = torch.randint(0, 50257, (4, 32))

        losses = loss_fn(
            classification_logits=cls_logits,
            classification_targets=cls_labels,
            generation_logits=gen_logits,
            generation_targets=gen_targets,
        )

        assert torch.isfinite(losses["total"]), f"Loss is not finite: {losses['total']}"
        logger.info("  ✓ Loss OK: %.4f (cls=%.4f, gen=%.4f)", losses["total"].item(), losses["classification"].item(), losses["generation"].item())
    except Exception as e:
        msg = f"Loss computation failed: {e}"
        errors.append(msg)
        logger.error("  ✗ %s", msg)

    # 6. Classification metrics
    logger.info("[6/7] Classification metrics …")
    try:
        import numpy as np
        from src.evaluation.classification_metrics import ClassificationEvaluator

        evaluator = ClassificationEvaluator()
        preds = np.random.rand(50, 14)
        labels = np.random.randint(0, 2, (50, 14))

        metrics = evaluator.evaluate(y_true=labels, y_score=preds)
        assert "macro" in metrics, f"Missing 'macro' in {metrics.keys()}"
        logger.info("  ✓ Classification metrics OK. macro_auc=%.4f", metrics["macro"]["auc_roc"])
    except Exception as e:
        msg = f"Classification metrics failed: {e}"
        errors.append(msg)
        logger.error("  ✗ %s", msg)

    # 7. Generation metrics (may fail gracefully if packages not installed)
    logger.info("[7/7] Generation metrics …")
    try:
        from src.evaluation.generation_metrics import GenerationEvaluator

        gen_eval = GenerationEvaluator()
        refs = ["The heart is normal in size.", "No acute cardiopulmonary process."]
        hyps = ["Heart size is normal.", "No acute findings."]

        gen_metrics = gen_eval.evaluate(predictions=hyps, references=refs)
        logger.info("  ✓ Generation metrics OK. %s", gen_metrics)
    except Exception as e:
        msg = f"Generation metrics failed: {e}"
        errors.append(msg)
        logger.error("  ✗ %s", msg)

    # Summary
    logger.info("=" * 60)
    if errors:
        logger.error("VALIDATION FAILED — %d error(s):", len(errors))
        for i, err in enumerate(errors, 1):
            logger.error("  %d. %s", i, err)
        sys.exit(1)
    else:
        logger.info("ALL VALIDATIONS PASSED ✓")
        logger.info("=" * 60)


# ======================================================================
# Main
# ======================================================================


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # Load config
    config = load_config(Path(args.config) if args.config else None)

    # Override seed if specified
    if args.seed is not None:
        config.training.seed = args.seed

    # Setup logging
    setup_logging(
        log_dir=config.training.log_dir,
        level=getattr(logging, args.log_level),
        experiment_name="knoclip",
    )

    set_seed(config.training.seed)

    # Dispatch
    if args.command == "phase1":
        cmd_phase1(config, args)
    elif args.command == "train":
        cmd_train(config, args)
    elif args.command == "evaluate":
        cmd_evaluate(config, args)
    elif args.command == "validate":
        cmd_validate(config, args)
    else:
        logger.error("No command specified. Use --help for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()

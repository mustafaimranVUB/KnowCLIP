"""Shared pytest fixtures for KnoCLIP-XAI tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.core.config import (
    ModelConfig,
    ProjectConfig,
    get_baseline_config,
    get_neurosymbolic_config,
)
from src.knowledge.extraction import (
    Certainty,
    EntityType,
    ExtractedEntity,
    ExtractedTriple,
    RelationType,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@pytest.fixture
def project_config() -> ProjectConfig:
    """Default project configuration for testing."""
    return ProjectConfig()


@pytest.fixture
def baseline_config() -> ModelConfig:
    """Baseline model config (no KG)."""
    return get_baseline_config()


@pytest.fixture
def neurosymbolic_config() -> ModelConfig:
    """Full neuro-symbolic model config (with KG)."""
    return get_neurosymbolic_config()


# ---------------------------------------------------------------------------
# Dummy data
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_images() -> torch.Tensor:
    """Batch of 2 random 224×224 RGB images."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def dummy_graph():
    """Small graph fixture: 6 nodes, 4 edges, batched for 2 samples."""
    graph_x = torch.randn(6, 768)
    graph_edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
    graph_edge_type = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    graph_batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    return {
        "graph_x": graph_x,
        "graph_edge_index": graph_edge_index,
        "graph_edge_type": graph_edge_type,
        "graph_batch": graph_batch,
    }


@pytest.fixture
def sample_entities() -> list[ExtractedEntity]:
    """Sample entities for graph building tests."""
    return [
        ExtractedEntity(
            text="heart",
            entity_type=EntityType.ANATOMY,
            certainty=Certainty.DEFINITELY_PRESENT,
            start_ix=0,
            end_ix=5,
        ),
        ExtractedEntity(
            text="cardiomegaly",
            entity_type=EntityType.OBSERVATION,
            certainty=Certainty.DEFINITELY_PRESENT,
            start_ix=6,
            end_ix=18,
        ),
        ExtractedEntity(
            text="lungs",
            entity_type=EntityType.ANATOMY,
            certainty=Certainty.DEFINITELY_PRESENT,
            start_ix=20,
            end_ix=25,
        ),
        ExtractedEntity(
            text="opacity",
            entity_type=EntityType.OBSERVATION,
            certainty=Certainty.UNCERTAIN,
            start_ix=26,
            end_ix=33,
        ),
    ]


@pytest.fixture
def sample_triples() -> list[ExtractedTriple]:
    """Sample triples for graph building tests."""
    return [
        ExtractedTriple(
            head_text="cardiomegaly",
            relation=RelationType.LOCATED_AT,
            tail_text="heart",
        ),
        ExtractedTriple(
            head_text="opacity",
            relation=RelationType.LOCATED_AT,
            tail_text="lungs",
        ),
    ]


@pytest.fixture
def dummy_labels() -> torch.Tensor:
    """Random binary labels for 14-class classification, batch=2."""
    return torch.randint(0, 2, (2, 14)).float()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed_all():
    """Set deterministic seed for all tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield

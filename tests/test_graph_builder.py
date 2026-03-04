"""Tests for src.knowledge.graph_builder module."""

from __future__ import annotations

import pytest
import torch

from src.knowledge.extraction import (
    Certainty,
    EntityType,
    ExtractedEntity,
    ExtractedTriple,
    RelationType,
)
from src.knowledge.graph_builder import (
    EDGE_TYPE_MAP,
    NUM_EDGE_TYPES,
    KnowledgeGraphBuilder,
    validate_graph,
)


class TestEdgeTypeMap:
    def test_expected_types(self):
        assert EDGE_TYPE_MAP["located_at"] == 0
        assert EDGE_TYPE_MAP["modify"] == 1
        assert EDGE_TYPE_MAP["suggestive_of"] == 2
        assert EDGE_TYPE_MAP["associated_with"] == 3
        assert EDGE_TYPE_MAP["self_loop"] == 4

    def test_num_edge_types(self):
        assert NUM_EDGE_TYPES == 5


class TestKnowledgeGraphBuilder:
    def test_build_from_extraction(self, sample_entities, sample_triples):
        builder = KnowledgeGraphBuilder(node_dim=768, add_self_loops=True)
        graph = builder.build_from_extraction(
            entities=sample_entities,
            triples=sample_triples,
        )

        # Node features
        assert graph.x is not None
        assert graph.x.shape[0] == 4  # 4 unique entities
        assert graph.x.shape[1] == 768

        # Edges: 2 triples + 4 self-loops = 6
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] == 6

        # Edge types
        assert graph.edge_type is not None
        assert graph.edge_type.shape[0] == 6

    def test_build_without_self_loops(self, sample_entities, sample_triples):
        builder = KnowledgeGraphBuilder(node_dim=768, add_self_loops=False)
        graph = builder.build_from_extraction(
            entities=sample_entities,
            triples=sample_triples,
        )
        # Only 2 triples
        assert graph.edge_index.shape[1] == 2

    def test_build_bidirectional(self, sample_entities, sample_triples):
        builder = KnowledgeGraphBuilder(node_dim=768, add_self_loops=False, bidirectional=True)
        graph = builder.build_from_extraction(
            entities=sample_entities,
            triples=sample_triples,
        )
        # 2 triples × 2 directions = 4
        assert graph.edge_index.shape[1] == 4

    def test_deduplication(self):
        """Duplicate entity texts map to the same node."""
        builder = KnowledgeGraphBuilder(node_dim=768, add_self_loops=False)
        entities = [
            ExtractedEntity(text="heart", entity_type=EntityType.ANATOMY, certainty=Certainty.DEFINITELY_PRESENT),
            ExtractedEntity(text="Heart", entity_type=EntityType.ANATOMY, certainty=Certainty.DEFINITELY_PRESENT),
            ExtractedEntity(text="opacity", entity_type=EntityType.OBSERVATION, certainty=Certainty.UNCERTAIN),
        ]
        triples = [
            ExtractedTriple(head_text="opacity", relation=RelationType.LOCATED_AT, tail_text="heart"),
        ]
        graph = builder.build_from_extraction(entities=entities, triples=triples)
        # "heart" and "Heart" normalise to same → 2 unique nodes
        assert graph.x.shape[0] == 2

    def test_build_global_graph(self, sample_entities, sample_triples):
        builder = KnowledgeGraphBuilder(node_dim=768, add_self_loops=True)
        graph = builder.build_global_graph(
            all_entities=[sample_entities, sample_entities],
            all_triples=[sample_triples, sample_triples],
        )
        # Entities are deduplicated across reports
        assert graph.x.shape[0] == 4

    def test_empty_entities(self):
        builder = KnowledgeGraphBuilder(node_dim=768, add_self_loops=True)
        graph = builder.build_from_extraction(entities=[], triples=[])
        assert graph.x.shape[0] == 0
        assert graph.edge_index.shape[1] == 0

    def test_no_nan_features(self, sample_entities, sample_triples):
        builder = KnowledgeGraphBuilder(node_dim=768, add_self_loops=True)
        graph = builder.build_from_extraction(
            entities=sample_entities,
            triples=sample_triples,
        )
        assert not torch.isnan(graph.x).any()


class TestValidateGraph:
    def test_valid_graph(self, sample_entities, sample_triples):
        # Add a cross-component triple to ensure connectivity >= 80%
        from src.knowledge.extraction import ExtractedTriple, RelationType
        connected_triples = sample_triples + [
            ExtractedTriple(
                head_text="cardiomegaly",
                relation=RelationType.ASSOCIATED_WITH,
                tail_text="opacity",
            ),
        ]
        builder = KnowledgeGraphBuilder(node_dim=768, add_self_loops=True)
        graph = builder.build_from_extraction(
            entities=sample_entities,
            triples=connected_triples,
        )
        result = validate_graph(graph)
        assert result["valid"] is True
        assert result["node_feature_dim"]["passed"] is True
        assert result["no_nan_features"]["passed"] is True
        assert result["edge_type_range"]["passed"] is True

    def test_connectivity_check(self, sample_entities, sample_triples):
        builder = KnowledgeGraphBuilder(node_dim=768, add_self_loops=True)
        graph = builder.build_from_extraction(
            entities=sample_entities,
            triples=sample_triples,
        )
        result = validate_graph(graph)
        assert "connectivity" in result

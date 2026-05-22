"""Hybrid Knowledge Graph construction using PyTorch Geometric.

Builds the heterogeneous KG from RadGraph entities/relations + UMLS
ontology relationships.  Produces ``torch_geometric.data.Data`` objects.
"""

from __future__ import annotations

import logging
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Edge-type integer codes (consistent across the entire codebase)
EDGE_TYPE_MAP: Dict[str, int] = {
    "located_at": 0,
    "modify": 1,
    "suggestive_of": 2,
    "associated_with": 3,
    "self_loop": 4,
}

# Number of distinct edge types
NUM_EDGE_TYPES: int = len(EDGE_TYPE_MAP)


@dataclass
class NodeInfo:
    """Metadata for a single KG node."""
    node_id: int
    text: str
    cui: Optional[str] = None
    entity_type: str = ""  # 'anatomy' / 'observation' / 'measurement'
    certainty: str = ""
    certainty_counts: Dict[str, int] = field(default_factory=dict)
    embedding: Optional[torch.Tensor] = None
    is_ontology_node: bool = False  # True for V_prior (UMLS-only) nodes


class KnowledgeGraphBuilder:
    """Build a PyTorch Geometric ``Data`` graph from entities and triples.

    The builder maintains an internal node registry (keyed on normalised
    text) so that duplicate mentions map to the same node.

    Parameters:
        node_dim: Embedding dimension for node features (must match the
            visual encoder dim for downstream cross-attention).
        add_self_loops: Whether to add self-loop edges for every node.
        bidirectional: If ``True``, add reverse edges for every relation.
    """

    def __init__(
        self,
        node_dim: int = 768,
        add_self_loops: bool = True,
        bidirectional: bool = False,
    ) -> None:
        self.node_dim = node_dim
        self.add_self_loops = add_self_loops
        self.bidirectional = bidirectional

        # Internal state (reset between builds if using ``build_*``)
        self._node_map: OrderedDict[str, NodeInfo] = OrderedDict()
        self._edges: List[Tuple[int, int, int]] = []  # (src, dst, edge_type)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_extraction(
        self,
        entities: list,  # List[ExtractedEntity]
        triples: list,   # List[ExtractedTriple]
        grounding_results: Optional[list] = None,  # List[GroundingResult]
        embeddings: Optional[Dict[str, torch.Tensor]] = None,
        mrrel_index: Optional[Any] = None,
    ) -> Any:  # torch_geometric.data.Data
        """Build a graph from entity extraction + grounding outputs.

        Args:
            entities: Entities from :class:`EntityExtractor`.
            triples: Triples from :class:`EntityExtractor`.
            grounding_results: Optional grounding results for CUI
                enrichment.
            embeddings: Optional pre-computed embeddings keyed by
                normalised text.

        Returns:
            ``torch_geometric.data.Data`` object.
        """
        self.reset()

        # Register nodes from entities
        grounding_map = self._grounding_map_from_results(grounding_results)

        for ent in entities:
            norm_text = self._normalize(ent.text)
            emb = embeddings.get(norm_text) if embeddings else None
            cui = None
            if norm_text in grounding_map:
                cui = grounding_map[norm_text].best_cui

            # If entity already has CUI from previous grounding pass
            if hasattr(ent, "cui") and ent.cui:
                cui = ent.cui

            self._register_node(
                text=norm_text,
                cui=cui,
                entity_type=ent.entity_type.value if hasattr(ent.entity_type, "value") else str(ent.entity_type),
                certainty=ent.certainty.value if hasattr(ent.certainty, "value") else str(ent.certainty),
                embedding=emb,
            )

        # Register edges from triples
        for triple in triples:
            head_norm = self._normalize(triple.head_text)
            tail_norm = self._normalize(triple.tail_text)
            rel_str = triple.relation.value if hasattr(triple.relation, "value") else str(triple.relation)

            self._add_edge(head_norm, tail_norm, rel_str)

        if mrrel_index is not None and grounding_map:
            self._add_ontology_edges(
                entities=entities,
                grounding_map=grounding_map,
                mrrel_index=mrrel_index,
            )

        return self._to_pyg_data()

    def build_global_graph(
        self,
        all_entities: List[list],
        all_triples: List[list],
        all_groundings: Optional[List[list]] = None,
        embeddings: Optional[Dict[str, torch.Tensor]] = None,
        mrrel_index: Optional[Any] = None,
    ) -> Any:  # torch_geometric.data.Data
        """Build a global graph aggregated from all reports.

        Nodes are deduplicated across reports.

        Args:
            all_entities: List of entity lists (one per report).
            all_triples: List of triple lists (one per report).
            all_groundings: Optional grounding results per report.
            embeddings: Optional pre-computed embeddings.

        Returns:
            ``torch_geometric.data.Data`` object.
        """
        self.reset()

        num_reports = max(
            len(all_entities),
            len(all_triples),
            len(all_groundings) if all_groundings else 0,
        )

        for report_idx in range(num_reports):
            entities = all_entities[report_idx] if report_idx < len(all_entities) else []
            triples = all_triples[report_idx] if report_idx < len(all_triples) else []
            grounding_results = (
                all_groundings[report_idx]
                if all_groundings is not None and report_idx < len(all_groundings)
                else None
            )
            grounding_map = self._grounding_map_from_results(grounding_results)

            for ent in entities:
                norm_text = self._normalize(ent.text)
                cui = getattr(ent, "cui", None)
                if norm_text in grounding_map:
                    cui = grounding_map[norm_text].best_cui
                emb = embeddings.get(norm_text) if embeddings else None

                self._register_node(
                    text=norm_text,
                    cui=cui,
                    entity_type=ent.entity_type.value if hasattr(ent.entity_type, "value") else str(ent.entity_type),
                    certainty=ent.certainty.value if hasattr(ent.certainty, "value") else str(ent.certainty),
                    embedding=emb,
                )

            for triple in triples:
                head_norm = self._normalize(triple.head_text)
                tail_norm = self._normalize(triple.tail_text)
                rel_str = triple.relation.value if hasattr(triple.relation, "value") else str(triple.relation)
                self._add_edge(head_norm, tail_norm, rel_str)

            if mrrel_index is not None and grounding_map:
                self._add_ontology_edges(
                    entities=entities,
                    grounding_map=grounding_map,
                    mrrel_index=mrrel_index,
                )

        return self._to_pyg_data()

    def reset(self) -> None:
        """Clear internal state for a fresh build."""
        self._node_map.clear()
        self._edges.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_node(
        self,
        text: str,
        cui: Optional[str] = None,
        entity_type: str = "",
        certainty: str = "",
        embedding: Optional[torch.Tensor] = None,
        is_ontology_node: bool = False,
    ) -> int:
        """Register a node (or retrieve existing) and return its integer ID."""
        if text in self._node_map:
            node = self._node_map[text]
            # Update fields if richer info is available
            if cui and not node.cui:
                node.cui = cui
            if entity_type and not node.entity_type:
                node.entity_type = entity_type
            if embedding is not None and node.embedding is None:
                node.embedding = embedding
            if certainty:
                node.certainty_counts[certainty] = node.certainty_counts.get(certainty, 0) + 1
                if not node.certainty:
                    node.certainty = certainty
            return node.node_id

        node_id = len(self._node_map)
        self._node_map[text] = NodeInfo(
            node_id=node_id,
            text=text,
            cui=cui,
            entity_type=entity_type,
            certainty=certainty,
            certainty_counts={certainty: 1} if certainty else {},
            embedding=embedding,
            is_ontology_node=is_ontology_node,
        )
        return node_id

    @staticmethod
    def _grounding_map_from_results(grounding_results: Optional[list]) -> Dict[str, Any]:
        grounding_map: Dict[str, Any] = {}
        if grounding_results:
            from src.knowledge.ontology_grounding import GroundingResult

            for gr in grounding_results:
                if isinstance(gr, GroundingResult) and gr.mapped:
                    grounding_map[gr.normalized] = gr
        return grounding_map

    def _append_edge_by_id(self, src: int, dst: int, edge_type: int) -> None:
        self._edges.append((src, dst, edge_type))

        if self.bidirectional:
            self._edges.append((dst, src, edge_type))

    def _add_edge(self, head_text: str, tail_text: str, relation: str) -> None:
        """Add an edge between two nodes (must be registered)."""
        if head_text not in self._node_map or tail_text not in self._node_map:
            logger.debug(
                "Skipping edge '%s' -[%s]-> '%s': node(s) not registered",
                head_text,
                relation,
                tail_text,
            )
            return

        src = self._node_map[head_text].node_id
        dst = self._node_map[tail_text].node_id
        etype = EDGE_TYPE_MAP.get(relation, EDGE_TYPE_MAP.get("associated_with", 3))

        self._append_edge_by_id(src, dst, etype)

    def _add_ontology_edges(
        self,
        entities: list,
        grounding_map: Dict[str, Any],
        mrrel_index: Any,
    ) -> None:
        """Add report-local ontology edges between grounded concepts.

        Ontology relations are collapsed to the existing ``associated_with``
        edge type so the downstream edge vocabulary remains unchanged.
        """
        report_node_ids_by_cui: Dict[str, Set[int]] = {}
        for ent in entities:
            norm_text = self._normalize(ent.text)
            grounding = grounding_map.get(norm_text)
            if grounding is None or not grounding.best_cui:
                continue
            node_info = self._node_map.get(norm_text)
            if node_info is None:
                continue
            report_node_ids_by_cui.setdefault(grounding.best_cui, set()).add(node_info.node_id)

        if not report_node_ids_by_cui:
            return

        seen_edges: Set[Tuple[int, int, int]] = set()
        ontology_edge_type = EDGE_TYPE_MAP["associated_with"]
        for source_cui, source_node_ids in report_node_ids_by_cui.items():
            for _relation, target_cui in mrrel_index.get_neighbours(source_cui):
                target_node_ids = report_node_ids_by_cui.get(target_cui)
                if not target_node_ids:
                    continue
                for source_node_id in source_node_ids:
                    for target_node_id in target_node_ids:
                        if source_node_id == target_node_id:
                            continue
                        edge_key = (source_node_id, target_node_id, ontology_edge_type)
                        if edge_key in seen_edges:
                            continue
                        seen_edges.add(edge_key)
                        self._append_edge_by_id(
                            source_node_id,
                            target_node_id,
                            ontology_edge_type,
                        )

    def _to_pyg_data(self) -> Any:
        """Convert internal state to a ``torch_geometric.data.Data`` object."""
        try:
            from torch_geometric.data import Data  # type: ignore
        except ImportError:
            raise ImportError(
                "torch_geometric not installed. "
                "Install with: pip install torch-geometric"
            )

        num_nodes = len(self._node_map)
        if num_nodes == 0:
            return Data(
                x=torch.zeros(0, self.node_dim),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_type=torch.zeros(0, dtype=torch.long),
            )

        # Build node features
        features = []
        node_texts: List[str] = []
        node_cuis: List[Optional[str]] = []
        node_types: List[str] = []
        node_certainties: List[str] = []
        node_certainty_counts: List[Dict[str, int]] = []
        node_is_ontology: List[bool] = []

        for text, info in self._node_map.items():
            if info.embedding is not None:
                feat = info.embedding.detach().float()
                if feat.shape[0] != self.node_dim:
                    # Project if needed
                    feat = torch.nn.functional.pad(
                        feat, (0, max(0, self.node_dim - feat.shape[0]))
                    )[: self.node_dim]
            else:
                # Initialize with small random values (seeded by hash)
                rng = np.random.RandomState(hash(text) % (2**31))
                feat = torch.from_numpy(
                    rng.randn(self.node_dim).astype(np.float32) * 0.01
                )
            features.append(feat)
            node_texts.append(text)
            node_cuis.append(info.cui)
            node_types.append(info.entity_type)
            if info.certainty_counts:
                primary_certainty = sorted(
                    info.certainty_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[0][0]
            else:
                primary_certainty = info.certainty
            node_certainties.append(primary_certainty)
            node_certainty_counts.append(dict(sorted(info.certainty_counts.items())))
            node_is_ontology.append(bool(info.is_ontology_node))

        x = torch.stack(features)  # (N, node_dim)

        # Self-loops
        if self.add_self_loops:
            for nid in range(num_nodes):
                self._edges.append((nid, nid, EDGE_TYPE_MAP["self_loop"]))

        # Edge tensors
        if self._edges:
            src_ids = [e[0] for e in self._edges]
            dst_ids = [e[1] for e in self._edges]
            edge_types = [e[2] for e in self._edges]

            edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        collapsed_edge_counter = Counter(self._edges)
        if collapsed_edge_counter:
            collapsed_edges = sorted(collapsed_edge_counter.items(), key=lambda item: item[0])
            collapsed_src = [edge[0][0] for edge in collapsed_edges]
            collapsed_dst = [edge[0][1] for edge in collapsed_edges]
            collapsed_type = [edge[0][2] for edge in collapsed_edges]
            collapsed_weight = [edge[1] for edge in collapsed_edges]
            collapsed_edge_index = torch.tensor([collapsed_src, collapsed_dst], dtype=torch.long)
            collapsed_edge_type = torch.tensor(collapsed_type, dtype=torch.long)
            collapsed_edge_weight = torch.tensor(collapsed_weight, dtype=torch.float32)
        else:
            collapsed_edge_index = torch.zeros(2, 0, dtype=torch.long)
            collapsed_edge_type = torch.zeros(0, dtype=torch.long)
            collapsed_edge_weight = torch.zeros(0, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
        )

        # Attach metadata as Python attributes (not tensors)
        data.node_texts = node_texts
        data.node_cuis = node_cuis
        data.node_types = node_types
        data.node_certainties = node_certainties
        data.node_certainty_counts = node_certainty_counts
        data.node_is_ontology = node_is_ontology
        data.collapsed_edge_index = collapsed_edge_index
        data.collapsed_edge_type = collapsed_edge_type
        data.collapsed_edge_weight = collapsed_edge_weight

        return data

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalise entity text for node deduplication."""
        from src.knowledge.extraction import normalize_text

        return normalize_text(text)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_graph(data: Any) -> Dict[str, Any]:
        """Run sanity checks on a constructed graph.

        Returns:
            Dict with check name → (passed: bool, detail: str).
        """
        checks: Dict[str, Any] = {}

        # Feature dimension
        if hasattr(data, "x") and data.x is not None:
            checks["node_feature_dim"] = {
                "passed": data.x.shape[1] == 768,
                "detail": f"shape={tuple(data.x.shape)}",
            }
            checks["no_nan_features"] = {
                "passed": not torch.isnan(data.x).any().item(),
                "detail": f"nan_count={torch.isnan(data.x).sum().item()}",
            }
        else:
            checks["node_feature_dim"] = {"passed": False, "detail": "No node features"}

        # Edge type range
        if hasattr(data, "edge_type") and data.edge_type is not None and data.edge_type.numel() > 0:
            emin = data.edge_type.min().item()
            emax = data.edge_type.max().item()
            checks["edge_type_range"] = {
                "passed": emin >= 0 and emax <= 4,
                "detail": f"range=[{emin}, {emax}]",
            }
        else:
            checks["edge_type_range"] = {"passed": True, "detail": "No edges"}

        # Connectivity (via simple BFS)
        if hasattr(data, "edge_index") and data.edge_index.numel() > 0:
            num_nodes = data.num_nodes
            adj: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
            ei = data.edge_index
            for j in range(ei.shape[1]):
                s, d = ei[0, j].item(), ei[1, j].item()
                adj[s].add(d)
                adj[d].add(s)

            visited: Set[int] = set()
            queue = [0]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                queue.extend(adj[node] - visited)

            largest_frac = len(visited) / num_nodes if num_nodes > 0 else 0
            # Global KG across many reports is naturally fragmented;
            # threshold is relaxed to 0.5 (warn below that, don't hard-fail).
            checks["connectivity"] = {
                "passed": largest_frac >= 0.5,
                "detail": f"largest_component={len(visited)}/{num_nodes} ({largest_frac:.1%})",
            }
        else:
            checks["connectivity"] = {"passed": False, "detail": "No edges"}

        # Overall "valid" flag
        checks["valid"] = all(
            v["passed"] for v in checks.values() if isinstance(v, dict) and "passed" in v
        )
        return checks


# Module-level alias for convenience
validate_graph = KnowledgeGraphBuilder.validate_graph

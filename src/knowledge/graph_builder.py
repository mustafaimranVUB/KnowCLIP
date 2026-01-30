"""
Graph Construction Module
Builds PyTorch Geometric graphs from extracted and grounded entities.

"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from torch_geometric.data import Data


class GraphBuilder:
    """
    Constructs PyTorch Geometric graph from grounded triplets.

    Features:
        - Multiple relation types (RGCN support)
        - Bidirectional edge options
        - Self-loop addition for node features
        - Configurable embedding dimensions
    """

    # RadGraph schema relation mapping [cite: 5200]
    RELATION_MAP = {
        "located_at": 0,
        "modify": 1,
        "suggestive_of": 2,
        "associated_with": 3,
    }

    def __init__(self, add_self_loops: bool = True, bidirectional: bool = False):
        """
        Initialize the graph builder.

        Args:
            add_self_loops: Whether to add self-loops for all nodes
            bidirectional: Whether to add reverse edges for each relation
        """
        self.add_self_loops = add_self_loops
        self.bidirectional = bidirectional

    def build_graph(
        self,
        triplets: List[Dict[str, Any]],
        node_metadata: Dict[str, Any],
    ) -> Data:
        """
        Constructs a PyTorch Geometric Data object from triplets.

        Args:
            triplets: List of enriched triplet dictionaries
            node_metadata: Dictionary mapping entity text -> grounded metadata with embeddings

        Returns:
            PyTorch Geometric Data object with:
                - x: Node feature matrix [num_nodes, embedding_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge relation types [num_edges]
                - node_names: List of entity names (for reference)
        """
        # 1. Create entity name -> node index mapping
        unique_entities = sorted(
            list(set([t["head"] for t in triplets] + [t["tail"] for t in triplets]))
        )
        entity_to_idx = {name: i for i, name in enumerate(unique_entities)}
        num_nodes = len(unique_entities)

        # 2. Collect node features (embeddings)
        node_features = []
        for entity_name in unique_entities:
            if entity_name in node_metadata:
                embedding = node_metadata[entity_name]["embedding"]
                # Ensure embedding is on CPU and detached
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.detach().cpu()
                node_features.append(embedding)
            else:
                # Fallback: random embedding if not grounded
                node_features.append(torch.randn(768))

        x = torch.stack(node_features)  # Shape: [num_nodes, 768]

        # 3. Build edge index and attributes
        edge_sources = []
        edge_targets = []
        edge_types = []

        for triplet in triplets:
            head_idx = entity_to_idx[triplet["head"]]
            tail_idx = entity_to_idx[triplet["tail"]]
            relation = triplet["relation"]
            rel_type = self.RELATION_MAP.get(relation, 3)  # Default to 3

            # Forward edge
            edge_sources.append(head_idx)
            edge_targets.append(tail_idx)
            edge_types.append(rel_type)

            # Optionally add reverse edge
            if self.bidirectional:
                edge_sources.append(tail_idx)
                edge_targets.append(head_idx)
                edge_types.append(rel_type)

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long)

        # 4. Add self-loops if requested
        if self.add_self_loops:
            self_loop_indices = torch.tensor(
                [[i, i] for i in range(num_nodes)], dtype=torch.long
            ).t()
            self_loop_attrs = torch.zeros(num_nodes, dtype=torch.long)

            edge_index = torch.cat([edge_index, self_loop_indices], dim=1)
            edge_attr = torch.cat([edge_attr, self_loop_attrs])

        # 5. Create PyTorch Geometric Data object
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Store metadata for reference
        graph_data.node_names = unique_entities
        graph_data.entity_to_idx = entity_to_idx
        graph_data.idx_to_entity = {v: k for k, v in entity_to_idx.items()}
        graph_data.num_relation_types = len(self.RELATION_MAP)

        return graph_data

    def add_node_metadata_to_graph(
        self, graph: Data, node_metadata: Dict[str, Any]
    ) -> Data:
        """
        Attaches rich node metadata to the graph (CUI, definition, etc.).

        Args:
            graph: PyTorch Geometric Data object
            node_metadata: Grounded entity metadata

        Returns:
            Updated graph with metadata attached
        """
        node_cuis = []
        node_definitions = []
        node_semantic_types = []

        for entity_name in graph.node_names:
            if entity_name in node_metadata:
                meta = node_metadata[entity_name]
                node_cuis.append(meta["cui"])
                node_definitions.append(meta["definition"])
                node_semantic_types.append(meta["semantic_type"])
            else:
                node_cuis.append("UNK")
                node_definitions.append(entity_name)
                node_semantic_types.append("Unknown")

        # Store as graph attributes
        graph.node_cuis = node_cuis
        graph.node_definitions = node_definitions
        graph.node_semantic_types = node_semantic_types

        return graph

    def get_node_by_name(self, graph: Data, node_name: str) -> Optional[int]:
        """
        Retrieve node index by entity name.

        Args:
            graph: PyTorch Geometric Data object
            node_name: Entity name to look up

        Returns:
            Node index or None if not found
        """
        if hasattr(graph, "entity_to_idx"):
            return graph.entity_to_idx.get(node_name)
        return None

    def get_node_neighbors(
        self, graph: Data, node_idx: int, relation_type: Optional[int] = None
    ) -> List[int]:
        """
        Get neighboring nodes for a given node.

        Args:
            graph: PyTorch Geometric Data object
            node_idx: Index of query node
            relation_type: Filter by specific relation type (None = all relations)

        Returns:
            List of neighbor node indices
        """
        neighbors = []

        for i, (src, dst) in enumerate(graph.edge_index.t()):
            src, dst = src.item(), dst.item()
            if src == node_idx:
                if relation_type is None or graph.edge_attr[i].item() == relation_type:
                    neighbors.append(dst)

        return neighbors

    def print_graph_summary(self, graph: Data) -> None:
        """
        Print a summary of the graph structure.

        Args:
            graph: PyTorch Geometric Data object
        """
        print("\n" + "=" * 50)
        print("GRAPH CONSTRUCTION SUMMARY")
        print("=" * 50)
        print(f"Number of Nodes: {graph.num_nodes}")
        print(f"Number of Edges: {graph.num_edges}")
        print(f"Node Feature Dimension: {graph.x.shape[1]}")
        print(f"Number of Relation Types: {graph.num_relation_types}")

        if hasattr(graph, "node_names"):
            print(f"\nEntities ({len(graph.node_names)}):")
            for i, name in enumerate(graph.node_names):
                semantic_type = (
                    graph.node_semantic_types[i]
                    if hasattr(graph, "node_semantic_types")
                    else "Unknown"
                )
                print(f"  {i}: {name} [{semantic_type}]")

        print("\nEdges (sample):")
        for i in range(min(10, graph.num_edges)):
            src, dst = graph.edge_index[:, i]
            rel_type = graph.edge_attr[i].item()
            rel_name = [k for k, v in self.RELATION_MAP.items() if v == rel_type][0]
            if hasattr(graph, "idx_to_entity"):
                print(
                    f"  {graph.idx_to_entity[src.item()]} --[{rel_name}]--> {graph.idx_to_entity[dst.item()]}"
                )

        print("=" * 50 + "\n")

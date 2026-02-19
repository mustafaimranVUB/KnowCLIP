"""
Dataset module for Phase II: MIMIC-CXR with Knowledge Graph Integration.

This module handles:
1. Loading MIMIC-CXR-JPG images
2. Loading reports from MIMIC-CXR-RRG
3. Integrating Phase I knowledge graph artifacts
4. Building report-specific and global knowledge graphs
5. Multi-label classification labels (CheXpert-14)
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path
from PIL import Image
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class MIMIC_CXR_Dataset(Dataset):
    """
    MIMIC-CXR Dataset with Knowledge Graph Integration.

    Supports:
    - Official train/validate/test splits (preventing data leakage)
    - Multi-label classification (14 CheXpert labels)
    - Report generation targets
    - Hybrid KG: Global ontology + Report-specific subgraphs

    Args:
        image_root: Path to MIMIC-CXR-JPG root directory
        report_root: Path to MIMIC-CXR-RRG root directory
        kg_artifacts_dir: Path to Phase I KG artifacts
        split: 'train', 'validate', or 'test'
        labels_file: Optional path to labels CSV (e.g., mimic-cxr-2.0.0-chexpert.csv)
        use_global_kg: Whether to include global medical ontology
        use_report_specific_kg: Whether to build report-specific subgraphs
        transform: Image transforms (if None, returns PIL Image)
    """

    # CheXpert-14 standard labels
    CHEXPERT_LABELS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    # RadGraph edge types (from Phase I)
    EDGE_TYPES = {
        "LOCATED_AT": 0,
        "MODIFY": 1,
        "SUGGESTIVE_OF": 2,
        "ASSOCIATED_WITH": 3,
    }

    def __init__(
        self,
        image_root: Path,
        report_root: Path,
        kg_artifacts_dir: Path,
        split: str = "train",
        labels_file: Optional[Path] = None,
        use_global_kg: bool = True,
        use_report_specific_kg: bool = True,
        transform: Optional[Any] = None,
        max_report_length: int = 128,
    ):
        super().__init__()

        assert split in ["train", "validate", "test"], f"Invalid split: {split}"

        self.image_root = Path(image_root)
        self.report_root = Path(report_root)
        self.kg_artifacts_dir = Path(kg_artifacts_dir)
        self.split = split
        self.use_global_kg = use_global_kg
        self.use_report_specific_kg = use_report_specific_kg
        self.transform = transform
        self.max_report_length = max_report_length

        # Load Phase I artifacts
        self._load_kg_artifacts()

        # Load dataset metadata
        self._load_metadata()

        # Load labels if provided
        if labels_file is not None:
            self._load_labels(labels_file)
        else:
            self.labels = None

        # Build global KG if requested
        if use_global_kg:
            self.global_kg = self._build_global_kg()
        else:
            self.global_kg = None

    def _load_kg_artifacts(self):
        """Load Phase I knowledge graph artifacts."""
        artifacts_dir = self.kg_artifacts_dir

        # Load entities with CUI mappings
        entities_path = artifacts_dir / "entities_with_cui.pkl"
        if entities_path.exists():
            with open(entities_path, "rb") as f:
                self.entities_with_cui = pickle.load(f)
        else:
            print(f"Warning: {entities_path} not found. KG features disabled.")
            self.entities_with_cui = []

        # Load mention to CUI mapping
        mention2cui_path = artifacts_dir / "mention2cui.pkl"
        if mention2cui_path.exists():
            with open(mention2cui_path, "rb") as f:
                self.mention2cui = pickle.load(f)
        else:
            print(f"Warning: {mention2cui_path} not found.")
            self.mention2cui = {}

    def _load_metadata(self):
        """
        Load MIMIC-CXR metadata and create sample index.

        Note: This is a simplified version. In practice, you would load
        the official MIMIC-CXR split files and metadata.
        """
        # TODO: Load official split information from MIMIC-CXR
        # For now, create a placeholder sample list
        # In production, this should read from mimic-cxr-2.0.0-split.csv

        # Placeholder: Generate sample IDs
        # Real implementation should load from official files
        self.samples = self._create_sample_index()

    def _create_sample_index(self) -> List[Dict]:
        """
        Create an index of samples for the current split.

        Returns:
            List of dictionaries with sample metadata:
            - subject_id
            - study_id
            - dicom_id
            - image_path
            - report_text
        """
        # Placeholder implementation
        # In production, parse official MIMIC-CXR metadata files
        samples = []

        # Example structure (simplified):
        # Real MIMIC-CXR has: subject_id/study_id/dicom_id.jpg hierarchy

        print(f"Note: Using placeholder sample index. Implement _create_sample_index()")
        print(f"      to load official MIMIC-CXR splits from metadata files.")

        # For demo purposes, return empty list
        # User should implement this based on their MIMIC-CXR directory structure
        return samples

    def _load_labels(self, labels_file: Path):
        """
        Load multi-label classification labels from CSV.

        Expected format: mimic-cxr-2.0.0-chexpert.csv with columns:
        - subject_id, study_id
        - One column per CheXpert label (1.0, 0.0, -1.0, or NaN)
        """
        # TODO: Implement CSV loading with pandas or csv module
        # Map labels to samples by study_id
        print(f"Note: Implement _load_labels() to load from {labels_file}")
        self.labels = {}

    def _build_global_kg(self) -> Optional[Data]:
        """
        Build a global medical knowledge graph from UMLS concepts.

        This creates a PyTorch Geometric Data object with:
        - Nodes: All UMLS CUIs encountered in Phase I
        - Edges: Ontology relationships (is-a, part-of, etc.)
        - Node features: CUI embeddings

        Returns:
            PyTorch Geometric Data object or None
        """
        if not self.entities_with_cui:
            return None

        # Collect all unique CUIs
        all_cuis = set()
        for report_entities in self.entities_with_cui:
            for entity_id, entity_data in report_entities.items():
                cui = entity_data.get("cui")
                if cui:
                    all_cuis.add(cui)

        # Build CUI to node index mapping
        cui_to_idx = {cui: idx for idx, cui in enumerate(sorted(all_cuis))}

        # Create node features (placeholder: use CUI hash as feature)
        # In production, use pretrained UMLS embeddings
        num_nodes = len(cui_to_idx)
        node_features = torch.randn(num_nodes, 768)  # Placeholder embeddings

        # Build edges from UMLS relationships
        # TODO: Load actual UMLS relationships from MRREL.RRF
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)

        global_kg = Data(
            x=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            cui_to_idx=cui_to_idx,
        )

        return global_kg

    def _build_report_specific_kg(self, report_idx: int) -> Optional[Data]:
        """
        Build a knowledge graph subgraph for a specific report.

        Uses entities extracted from the report via RadGraph (Phase I).

        Args:
            report_idx: Index into self.entities_with_cui

        Returns:
            PyTorch Geometric Data object with report-specific graph
        """
        if report_idx >= len(self.entities_with_cui):
            return None

        report_entities = self.entities_with_cui[report_idx]

        # Extract nodes (entities with CUIs)
        nodes = []
        node_features = []
        cui_list = []

        for entity_id, entity_data in report_entities.items():
            cui = entity_data.get("cui")
            if cui:
                nodes.append(entity_id)
                cui_list.append(cui)
                # Placeholder: random features (should use UMLS embeddings)
                node_features.append(torch.randn(768))

        if not nodes:
            return None

        # Stack node features
        x = torch.stack(node_features) if node_features else torch.empty(0, 768)

        # TODO: Extract edges from RadGraph relations
        # Phase I graph_builder.py has this logic - integrate here
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index, edge_type=edge_type, cuis=cui_list)

        return graph

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
            - image: (3, H, W) tensor
            - graph: PyTorch Geometric Data (if use_report_specific_kg=True)
            - global_graph: PyTorch Geometric Data (if use_global_kg=True)
            - labels: (14,) tensor of multi-label targets
            - report_text: String (for generation task)
            - metadata: Dict with subject_id, study_id, dicom_id
        """
        sample = self.samples[idx]

        # Load image
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Build report-specific KG
        if self.use_report_specific_kg:
            graph = self._build_report_specific_kg(idx)
        else:
            graph = None

        # Get labels
        if self.labels is not None:
            study_id = sample["study_id"]
            label_vector = self.labels.get(study_id, torch.zeros(14))
        else:
            label_vector = torch.zeros(14)

        # Get report text
        report_text = sample.get("report_text", "")

        return {
            "image": image,
            "graph": graph,
            "global_graph": self.global_kg,
            "labels": label_vector,
            "report_text": report_text,
            "metadata": {
                "subject_id": sample.get("subject_id"),
                "study_id": sample.get("study_id"),
                "dicom_id": sample.get("dicom_id"),
            },
        }


def collate_kg_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for batching samples with heterogeneous graphs.

    Handles variable-size report-specific graphs by creating a batch graph.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary with:
        - images: (B, 3, H, W)
        - graphs: Batched PyTorch Geometric Data
        - labels: (B, 14)
        - report_texts: List of strings
        - metadata: List of dicts
    """
    from torch_geometric.data import Batch as GeometricBatch

    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    report_texts = [item["report_text"] for item in batch]
    metadata = [item["metadata"] for item in batch]

    # Batch report-specific graphs
    graphs = [item["graph"] for item in batch if item["graph"] is not None]
    if graphs:
        batched_graph = GeometricBatch.from_data_list(graphs)
    else:
        batched_graph = None

    # Global graph is shared across batch
    global_graph = batch[0]["global_graph"]

    return {
        "images": images,
        "graphs": batched_graph,
        "global_graph": global_graph,
        "labels": labels,
        "report_texts": report_texts,
        "metadata": metadata,
    }


# Utility functions for loading MIMIC-CXR official metadata


def load_mimic_split(split_csv_path: Path, split: str) -> List[str]:
    """
    Load official MIMIC-CXR split file.

    Args:
        split_csv_path: Path to mimic-cxr-2.0.0-split.csv
        split: 'train', 'validate', or 'test'

    Returns:
        List of study IDs (dicom_id format) for the split
    """
    # TODO: Implement CSV parsing
    # Expected columns: dicom_id, subject_id, study_id, split
    print(f"Note: Implement load_mimic_split() to parse {split_csv_path}")
    return []


def load_chexpert_labels(labels_csv_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load CheXpert labels from official file.

    Args:
        labels_csv_path: Path to mimic-cxr-2.0.0-chexpert.csv

    Returns:
        Dictionary mapping study_id -> (14,) label tensor
        Label encoding: 1.0 (positive), 0.0 (negative), -1.0 (uncertain), NaN (unmention)
    """
    # TODO: Implement CSV parsing with pandas or csv module
    print(f"Note: Implement load_chexpert_labels() to parse {labels_csv_path}")
    return {}


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    image_root = Path("data/mimic-cxr-jpg")
    report_root = Path("data/MIMIC-CXR-RRG_small")
    kg_artifacts_dir = Path("outputs/KG")

    # Create dataset
    dataset = MIMIC_CXR_Dataset(
        image_root=image_root,
        report_root=report_root,
        kg_artifacts_dir=kg_artifacts_dir,
        split="test",
        use_global_kg=True,
        use_report_specific_kg=True,
    )

    print(f"Dataset created with {len(dataset)} samples")

    # Test single sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(
            f"Image shape: {sample['image'].shape if torch.is_tensor(sample['image']) else 'PIL Image'}"
        )
        print(f"Labels shape: {sample['labels'].shape}")
        print(f"Has graph: {sample['graph'] is not None}")

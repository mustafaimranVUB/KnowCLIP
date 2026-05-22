from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from kg_viewer.app import (
    build_study_catalog,
    extract_study_graph,
    filter_study_catalog,
    resolve_phase1_log_path,
)


def _make_graph(node_texts: list[str]) -> Data:
    graph = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_type=torch.tensor([1, 3], dtype=torch.long),
        num_nodes=len(node_texts),
    )
    graph.node_texts = node_texts
    graph.node_cuis = [""] * len(node_texts)
    graph.node_types = ["observation"] * len(node_texts)
    graph.node_certainties = ["definitely_present"] * len(node_texts)
    graph.node_is_ontology = [False] * len(node_texts)
    return graph


def test_build_study_catalog_and_filter() -> None:
    metadata = {
        "valid_studies": [
            {"subject_id": 10000032, "study_id": 50414267},
            {"subject_id": 10000033, "study_id": 50414268},
        ],
        "mention2cui": {"effusion": {"best_cui": "C0018995", "mapped": True}},
    }

    catalog = build_study_catalog(metadata, artifact_dir=Path("/tmp/kg_artifacts"))

    assert catalog.num_studies == 2
    assert catalog.study_df.loc[0, "study_key"] == "10000032_50414267"

    filtered = filter_study_catalog(catalog, "50414267")
    assert filtered["study_key"].tolist() == ["10000032_50414267"]


def test_extract_study_graph_writes_selected_graph(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "KG"
    artifact_dir.mkdir()

    report_graphs = {
        "10000032_50414267": _make_graph(["effusion", "right lung"]),
        "10000033_50414268": _make_graph(["opacity", "left lung"]),
    }
    torch.save(report_graphs, artifact_dir / "report_graphs.pt")

    cache_path = artifact_dir / ".kg_viewer_cache" / "study_graphs" / "10000032_50414267.pt"
    written_path = extract_study_graph(
        artifact_dir=artifact_dir,
        study_key="10000032_50414267",
        destination_path=cache_path,
    )

    assert written_path == cache_path
    assert cache_path.exists()

    extracted_graph = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert extracted_graph.node_texts == ["effusion", "right lung"]
    assert extracted_graph.edge_index.tolist() == [[0, 1], [1, 0]]


def test_resolve_phase1_log_path_prefers_primary_log_directory(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "outputs" / "KG"
    artifact_dir.mkdir(parents=True)

    primary_log_dir = tmp_path / "outputs" / "KG_outputs" / "logs"
    fallback_log_dir = tmp_path / "outputs" / "logs"
    primary_log_dir.mkdir(parents=True)
    fallback_log_dir.mkdir(parents=True)

    primary_log = primary_log_dir / "phase1_12022057.log"
    fallback_log = fallback_log_dir / "phase1_smoke.log"
    primary_log.write_text("Phase I complete in 1.0 s.\n", encoding="utf-8")
    fallback_log.write_text("Phase I complete in 2.0 s.\n", encoding="utf-8")

    resolved = resolve_phase1_log_path(artifact_dir)

    assert resolved == primary_log
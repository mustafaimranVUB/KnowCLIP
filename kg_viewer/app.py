from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

RELATION_LABELS: dict[int, str] = {
    0: "located_at",
    1: "modify",
    2: "suggestive_of",
    3: "associated_with",
    4: "self_loop",
}
RELATION_CODES: dict[str, int] = {name: code for code, name in RELATION_LABELS.items()}

NODE_COLORS: dict[str, str] = {
    "anatomy": "#2563eb",
    "observation": "#dc2626",
    "measurement": "#7c3aed",
    "unknown": "#64748b",
}

EDGE_COLORS: dict[str, str] = {
    "located_at": "#0f766e",
    "modify": "#b45309",
    "suggestive_of": "#7c2d12",
    "associated_with": "#4f46e5",
    "self_loop": "#94a3b8",
}


@dataclass(frozen=True)
class AppConfig:
    artifact_dir: Path
    host: str = "127.0.0.1"
    port: int = 8501
    extract_study_graph: str | None = None
    study_cache_path: Path | None = None


@dataclass(frozen=True)
class GraphIndex:
    artifact_dir: Path
    summary: dict[str, Any]
    node_df: pd.DataFrame
    relation_counts: pd.DataFrame
    node_type_counts: pd.DataFrame
    certainty_counts: pd.DataFrame
    source: np.ndarray
    target: np.ndarray
    relation_code: np.ndarray
    edge_weight: np.ndarray
    raw_num_edges: int
    graph_label: str = "Global graph"
    study_key: str | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.node_df.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.source.shape[0])

    @property
    def degrees(self) -> np.ndarray:
        return self.node_df["degree"].to_numpy(dtype=np.int32, copy=False)

    @property
    def node_ids(self) -> np.ndarray:
        return self.node_df["node_id"].to_numpy(dtype=np.int32, copy=False)

    @property
    def node_texts(self) -> np.ndarray:
        return self.node_df["text"].to_numpy(copy=False)

    @property
    def node_cuis(self) -> np.ndarray:
        return self.node_df["cui"].to_numpy(copy=False)

    @property
    def node_types(self) -> np.ndarray:
        return self.node_df["node_type"].to_numpy(copy=False)


@dataclass(frozen=True)
class StudyCatalog:
    artifact_dir: Path
    study_df: pd.DataFrame
    mention2cui: dict[str, Any]

    @property
    def num_studies(self) -> int:
        return int(self.study_df.shape[0])


@dataclass(frozen=True)
class Phase1LogSummary:
    log_path: Path
    completed: bool
    num_reports_loaded: int | None = None
    num_studies: int | None = None
    mapped_entities: int | None = None
    eligible_entities: int | None = None
    grounding_coverage_pct: float | None = None
    report_graph_count: int | None = None
    elapsed_seconds: float | None = None
    saved_global_kg: bool = False
    saved_report_graphs: bool = False
    saved_embeddings: bool = False
    saved_grounding: bool = False
    graph_validation: str | None = None
    final_summary: str | None = None


@dataclass(frozen=True)
class QuerySettings:
    focus_query: str = ""
    source_query: str = ""
    target_query: str = ""
    node_types: tuple[str, ...] = ()
    relation_names: tuple[str, ...] = ()
    expansion_radius: int = 1
    max_nodes: int = 80
    max_edges: int = 200
    preview_rows: int = 250


@dataclass(frozen=True)
class QueryResult:
    matched_nodes_df: pd.DataFrame
    matched_edges_df: pd.DataFrame
    display_nodes_df: pd.DataFrame
    display_edges_df: pd.DataFrame
    matched_node_count: int
    matched_edge_count: int
    rendered_node_count: int
    rendered_edge_count: int
    needs_narrowing: bool
    truncated: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(description="Launch the KG viewer")
    parser.add_argument(
        "--artifact-dir",
        default="outputs/KG",
        help="Directory that contains global_kg.pt and related Phase I artifacts",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for Streamlit")
    parser.add_argument("--port", default=8501, type=int, help="Port for Streamlit")
    parser.add_argument(
        "--extract-study-graph",
        default=None,
        help="Internal helper mode: extract a single per-study graph into a cache file and exit",
    )
    parser.add_argument(
        "--study-cache-path",
        default=None,
        help="Internal helper mode: destination path for an extracted per-study graph",
    )
    args, _ = parser.parse_known_args(argv)

    artifact_dir = Path(args.artifact_dir).expanduser()
    if not artifact_dir.is_absolute():
        artifact_dir = (repo_root() / artifact_dir).resolve()

    study_cache_path: Path | None = None
    if args.study_cache_path:
        study_cache_path = Path(args.study_cache_path).expanduser()
        if not study_cache_path.is_absolute():
            study_cache_path = (repo_root() / study_cache_path).resolve()

    return AppConfig(
        artifact_dir=artifact_dir,
        host=args.host,
        port=args.port,
        extract_study_graph=args.extract_study_graph,
        study_cache_path=study_cache_path,
    )


def load_summary(artifact_dir: Path) -> dict[str, Any]:
    summary_path = artifact_dir / "phase1_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_study_metadata(artifact_dir: Path) -> dict[str, Any] | None:
    metadata_path = artifact_dir / "study_metadata.pt"
    if not metadata_path.exists():
        return None
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    raise TypeError(f"Unexpected study metadata type: {type(metadata).__name__}")


def build_study_catalog(metadata: Mapping[str, Any], artifact_dir: Path) -> StudyCatalog:
    valid_studies = metadata.get("valid_studies", [])
    study_df = pd.DataFrame(valid_studies)
    if study_df.empty:
        study_df = pd.DataFrame(columns=["subject_id", "study_id", "study_key", "subject_id_str", "study_id_str"])
    else:
        study_df = study_df.copy()
        study_df["subject_id"] = study_df["subject_id"].astype(np.int64)
        study_df["study_id"] = study_df["study_id"].astype(np.int64)
        study_df["subject_id_str"] = study_df["subject_id"].astype(str)
        study_df["study_id_str"] = study_df["study_id"].astype(str)
        study_df["study_key"] = study_df["subject_id_str"] + "_" + study_df["study_id_str"]
        study_df = study_df[["subject_id", "study_id", "study_key", "subject_id_str", "study_id_str"]]

    mention2cui = metadata.get("mention2cui", {})
    if not isinstance(mention2cui, dict):
        mention2cui = {}

    return StudyCatalog(
        artifact_dir=artifact_dir,
        study_df=study_df,
        mention2cui=mention2cui,
    )


def load_study_catalog(artifact_dir: Path) -> StudyCatalog | None:
    metadata = load_study_metadata(artifact_dir)
    if metadata is None:
        return None
    return build_study_catalog(metadata, artifact_dir)


def filter_study_catalog(catalog: StudyCatalog, query: str, limit: int = 200) -> pd.DataFrame:
    normalized_query = query.strip()
    if not normalized_query:
        return catalog.study_df.iloc[0:0].copy()

    mask = (
        catalog.study_df["study_key"].str.contains(normalized_query, case=False, regex=False, na=False)
        | catalog.study_df["subject_id_str"].str.contains(normalized_query, case=False, regex=False, na=False)
        | catalog.study_df["study_id_str"].str.contains(normalized_query, case=False, regex=False, na=False)
    )
    matched = catalog.study_df.loc[mask].copy()
    if matched.empty:
        return matched

    exact_mask = (
        matched["study_key"].eq(normalized_query)
        | matched["subject_id_str"].eq(normalized_query)
        | matched["study_id_str"].eq(normalized_query)
    )
    matched["exact_match"] = exact_mask
    matched = matched.sort_values(
        ["exact_match", "subject_id", "study_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    return matched.head(limit).drop(columns=["exact_match"], errors="ignore")


def load_report_graphs_map(artifact_dir: Path) -> dict[str, Any]:
    report_graphs_path = artifact_dir / "report_graphs.pt"
    if not report_graphs_path.exists():
        raise FileNotFoundError(f"Missing per-study graph artifact: {report_graphs_path}")
    try:
        report_graphs = torch.load(
            report_graphs_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        report_graphs = torch.load(
            report_graphs_path,
            map_location="cpu",
            weights_only=False,
        )

    if not isinstance(report_graphs, dict):
        raise TypeError(f"Unexpected report graph container type: {type(report_graphs).__name__}")
    return report_graphs


def study_graph_cache_path(artifact_dir: Path, study_key: str) -> Path:
    safe_study_key = re.sub(r"[^0-9A-Za-z_.-]+", "_", study_key)
    return artifact_dir / ".kg_viewer_cache" / "study_graphs" / f"{safe_study_key}.pt"


def extract_study_graph(artifact_dir: Path, study_key: str, destination_path: Path | None = None) -> Path:
    cache_path = destination_path or study_graph_cache_path(artifact_dir, study_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    report_graphs = load_report_graphs_map(artifact_dir)
    study_graph = report_graphs.get(study_key)
    if study_graph is None:
        raise KeyError(f"Study graph `{study_key}` was not found in report_graphs.pt")

    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(study_graph, temp_path)
    temp_path.replace(cache_path)
    return cache_path


def ensure_cached_study_graph(artifact_dir: Path, study_key: str) -> Path:
    cache_path = study_graph_cache_path(artifact_dir, study_key)
    if cache_path.exists():
        return cache_path

    command = [
        sys.executable,
        str(repo_root() / "scripts" / "visualize_kg.py"),
        "--artifact-dir",
        str(artifact_dir),
        "--extract-study-graph",
        study_key,
        "--study-cache-path",
        str(cache_path),
    ]
    subprocess.run(command, cwd=str(repo_root()), check=True)
    if not cache_path.exists():
        raise FileNotFoundError(f"Expected extracted study cache at {cache_path}, but it was not created")
    return cache_path


def resolve_phase1_log_path(artifact_dir: Path) -> Path | None:
    candidate_dirs = [
        artifact_dir.parent / "KG_outputs" / "logs",
        artifact_dir.parent / "logs",
        artifact_dir / "logs",
    ]
    for candidate_dir in candidate_dirs:
        if candidate_dir.exists():
            matches = list(candidate_dir.glob("phase1*.log"))
            if matches:
                return max(matches, key=lambda path: path.stat().st_mtime)
    return None


def parse_phase1_log(log_path: Path) -> Phase1LogSummary:
    content = log_path.read_text(encoding="utf-8", errors="ignore")

    loaded_match = re.search(r"Loaded (\d+) reports \(of (\d+) studies\)\.", content)
    grounding_match = re.search(
        r"Grounding coverage: ([0-9.]+)% \((\d+)/(\d+) eligible mapped\)\.",
        content,
    )
    report_graphs_match = re.search(r"Saved (\d+) per-report graphs to", content)
    elapsed_match = re.search(r"Phase I complete in ([0-9.]+) s\.", content)
    validation_matches = re.findall(r"Graph validation: (.+)", content)
    summary_matches = re.findall(r"Phase I complete\. Summary:\n(.+)", content)

    return Phase1LogSummary(
        log_path=log_path,
        completed="Phase I complete" in content,
        num_reports_loaded=int(loaded_match.group(1)) if loaded_match else None,
        num_studies=int(loaded_match.group(2)) if loaded_match else None,
        mapped_entities=int(grounding_match.group(2)) if grounding_match else None,
        eligible_entities=int(grounding_match.group(3)) if grounding_match else None,
        grounding_coverage_pct=float(grounding_match.group(1)) if grounding_match else None,
        report_graph_count=int(report_graphs_match.group(1)) if report_graphs_match else None,
        elapsed_seconds=float(elapsed_match.group(1)) if elapsed_match else None,
        saved_global_kg="Saved global KG to" in content,
        saved_report_graphs="Saved 227835 per-report graphs to" in content or "per-report graphs to" in content,
        saved_embeddings="Saved 14587 embeddings to" in content or "Saved " in content and "embeddings to" in content,
        saved_grounding="Saved grounding to" in content,
        graph_validation=validation_matches[-1] if validation_matches else None,
        final_summary=summary_matches[-1] if summary_matches else None,
    )


def load_phase1_log_summary(artifact_dir: Path) -> Phase1LogSummary | None:
    log_path = resolve_phase1_log_path(artifact_dir)
    if log_path is None:
        return None
    return parse_phase1_log(log_path)


def build_graph_index(
    graph: Any,
    artifact_dir: Path,
    summary: dict[str, Any] | None = None,
    graph_label: str = "Global graph",
    study_key: str | None = None,
) -> GraphIndex:
    if not hasattr(graph, "edge_index") or not hasattr(graph, "edge_type"):
        raise ValueError("Graph artifact is missing edge_index or edge_type")

    edge_index = graph.edge_index.detach().cpu()
    edge_type = graph.edge_type.detach().cpu()

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")

    num_nodes = int(getattr(graph, "num_nodes", 0))
    if num_nodes <= 0:
        x = getattr(graph, "x", None)
        if x is not None:
            num_nodes = int(x.shape[0])
        else:
            num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0

    node_ids = np.arange(num_nodes, dtype=np.int32)
    node_texts = list(getattr(graph, "node_texts", []))
    node_cuis = list(getattr(graph, "node_cuis", []))
    node_types = list(getattr(graph, "node_types", []))
    node_certainties = list(getattr(graph, "node_certainties", []))
    node_is_ontology = list(getattr(graph, "node_is_ontology", []))

    if len(node_texts) < num_nodes:
        node_texts.extend([f"node_{idx}" for idx in range(len(node_texts), num_nodes)])
    if len(node_cuis) < num_nodes:
        node_cuis.extend([""] * (num_nodes - len(node_cuis)))
    if len(node_types) < num_nodes:
        node_types.extend(["unknown"] * (num_nodes - len(node_types)))
    if len(node_certainties) < num_nodes:
        node_certainties.extend([""] * (num_nodes - len(node_certainties)))
    if len(node_is_ontology) < num_nodes:
        node_is_ontology.extend([False] * (num_nodes - len(node_is_ontology)))

    raw_source = edge_index[0].numpy().astype(np.int32, copy=False)
    raw_target = edge_index[1].numpy().astype(np.int32, copy=False)
    raw_relation_code = edge_type.numpy().astype(np.int16, copy=False)
    raw_num_edges = int(raw_source.shape[0])

    if all(hasattr(graph, name) for name in ("collapsed_edge_index", "collapsed_edge_type", "collapsed_edge_weight")):
        collapsed_edge_index = graph.collapsed_edge_index.detach().cpu()
        collapsed_edge_type = graph.collapsed_edge_type.detach().cpu()
        collapsed_edge_weight = graph.collapsed_edge_weight.detach().cpu()
        source = collapsed_edge_index[0].numpy().astype(np.int32, copy=False)
        target = collapsed_edge_index[1].numpy().astype(np.int32, copy=False)
        relation_code = collapsed_edge_type.numpy().astype(np.int16, copy=False)
        edge_weight = collapsed_edge_weight.numpy().astype(np.float32, copy=False)
    else:
        collapsed_edges = np.stack([raw_source, raw_target, raw_relation_code.astype(np.int32)], axis=1)
        unique_edges, counts = np.unique(collapsed_edges, axis=0, return_counts=True)
        source = unique_edges[:, 0].astype(np.int32, copy=False)
        target = unique_edges[:, 1].astype(np.int32, copy=False)
        relation_code = unique_edges[:, 2].astype(np.int16, copy=False)
        edge_weight = counts.astype(np.float32, copy=False)

    out_degree = np.bincount(source, weights=edge_weight, minlength=num_nodes).astype(np.int32, copy=False)
    in_degree = np.bincount(target, weights=edge_weight, minlength=num_nodes).astype(np.int32, copy=False)
    degree = out_degree + in_degree

    node_df = pd.DataFrame(
        {
            "node_id": node_ids,
            "text": pd.Series(node_texts[:num_nodes], dtype="object").fillna(""),
            "cui": pd.Series(node_cuis[:num_nodes], dtype="object").fillna(""),
            "node_type": pd.Series(node_types[:num_nodes], dtype="object").fillna("unknown").replace("", "unknown"),
            "certainty": pd.Series(node_certainties[:num_nodes], dtype="object").fillna(""),
            "is_ontology": pd.Series(node_is_ontology[:num_nodes], dtype="bool").fillna(False),
            "in_degree": in_degree,
            "out_degree": out_degree,
            "degree": degree,
        }
    )
    node_df["node_type"] = node_df["node_type"].astype("category")

    relation_series = pd.Series(
        [RELATION_LABELS.get(int(code), f"relation_{int(code)}") for code in relation_code],
        dtype="object",
    )
    relation_counts = (
        pd.DataFrame({"relation": relation_series, "weight": edge_weight})
        .groupby("relation", as_index=False)
        .agg(count=("weight", "sum"), unique_edges=("weight", "size"))
        .sort_values(["count", "relation"], ascending=[False, True], ignore_index=True)
    )
    relation_counts["count"] = relation_counts["count"].astype(np.int64)
    relation_counts["unique_edges"] = relation_counts["unique_edges"].astype(np.int64)

    node_type_counts = (
        node_df["node_type"].astype("string").value_counts(dropna=False)
        .rename_axis("node_type")
        .reset_index(name="count")
        .sort_values(["count", "node_type"], ascending=[False, True], ignore_index=True)
    )
    certainty_counts = (
        node_df["certainty"].replace("", "unknown").astype("string").value_counts(dropna=False)
        .rename_axis("certainty")
        .reset_index(name="count")
        .sort_values(["count", "certainty"], ascending=[False, True], ignore_index=True)
    )

    merged_summary = dict(summary or {})
    merged_summary.setdefault("artifact_dir", str(artifact_dir))
    merged_summary.setdefault("graph_validation", {})
    merged_summary["viewer_graph_stats"] = {
        "graph_label": graph_label,
        "num_nodes": int(num_nodes),
        "num_unique_edges": int(source.shape[0]),
        "num_raw_edges": raw_num_edges,
    }
    if study_key is not None:
        merged_summary["viewer_graph_stats"]["study_key"] = study_key

    return GraphIndex(
        artifact_dir=artifact_dir,
        summary=merged_summary,
        node_df=node_df,
        relation_counts=relation_counts,
        node_type_counts=node_type_counts,
        certainty_counts=certainty_counts,
        source=source,
        target=target,
        relation_code=relation_code,
        edge_weight=edge_weight,
        raw_num_edges=raw_num_edges,
        graph_label=graph_label,
        study_key=study_key,
    )


def load_graph_index(artifact_dir: Path) -> GraphIndex:
    graph_path = artifact_dir / "global_kg.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph artifact: {graph_path}")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    return build_graph_index(
        graph=graph,
        artifact_dir=artifact_dir,
        summary=load_summary(artifact_dir),
        graph_label="Global graph",
    )


def search_node_ids(
    index: GraphIndex,
    query: str,
    allowed_node_types: Sequence[str] | None = None,
) -> np.ndarray:
    normalized_query = query.strip()
    mask = np.ones(index.num_nodes, dtype=bool)
    if allowed_node_types:
        mask &= index.node_df["node_type"].isin(list(allowed_node_types)).to_numpy(dtype=bool, copy=False)
    if not normalized_query:
        return index.node_ids[mask]

    text_match = index.node_df["text"].str.contains(normalized_query, case=False, regex=False, na=False)
    cui_match = index.node_df["cui"].str.contains(normalized_query, case=False, regex=False, na=False)
    mask &= (text_match | cui_match).to_numpy(dtype=bool, copy=False)
    return index.node_ids[mask]


def node_rows(index: GraphIndex, node_ids: np.ndarray, limit: int | None = None) -> pd.DataFrame:
    if node_ids.size == 0:
        return index.node_df.iloc[0:0].copy()
    ordered = node_ids.astype(np.int32, copy=False)
    frame = index.node_df.iloc[ordered].copy()
    frame = frame.sort_values(["degree", "text"], ascending=[False, True], ignore_index=True)
    if limit is not None:
        frame = frame.head(limit).reset_index(drop=True)
    return frame


def edge_rows(index: GraphIndex, edge_positions: np.ndarray, limit: int | None = None) -> pd.DataFrame:
    if edge_positions.size == 0:
        return pd.DataFrame(
            columns=[
                "edge_id",
                "source",
                "source_text",
                "source_cui",
                "source_type",
                "relation",
                "weight",
                "target",
                "target_text",
                "target_cui",
                "target_type",
            ]
        )

    selected = edge_positions.astype(np.int32, copy=False)
    if limit is not None:
        selected = selected[:limit]

    source = index.source[selected]
    target = index.target[selected]
    relation = np.array(
        [RELATION_LABELS.get(int(code), f"relation_{int(code)}") for code in index.relation_code[selected]],
        dtype=object,
    )
    return pd.DataFrame(
        {
            "edge_id": selected,
            "source": source,
            "source_text": index.node_texts[source],
            "source_cui": index.node_cuis[source],
            "source_type": index.node_types[source],
            "relation": relation,
            "weight": index.edge_weight[selected].astype(np.int64, copy=False),
            "target": target,
            "target_text": index.node_texts[target],
            "target_cui": index.node_cuis[target],
            "target_type": index.node_types[target],
        }
    )


def top_nodes_by_degree(index: GraphIndex, limit: int = 50) -> pd.DataFrame:
    return index.node_df.sort_values(["degree", "text"], ascending=[False, True], ignore_index=True).head(limit)


def top_edges_by_weight(index: GraphIndex, limit: int = 50) -> pd.DataFrame:
    if index.num_edges == 0:
        return edge_rows(index, np.array([], dtype=np.int32))
    positions = np.argsort(index.edge_weight, kind="stable")[::-1][:limit].astype(np.int32, copy=False)
    return edge_rows(index, positions, limit=None)


def grounding_lookup_rows(catalog: StudyCatalog, query: str, limit: int = 50) -> pd.DataFrame:
    normalized_query = query.strip()
    rows: list[dict[str, Any]] = []
    if not normalized_query:
        return pd.DataFrame(columns=["mention", "best_cui", "best_name", "mapped", "tuis"])

    for mention, payload in catalog.mention2cui.items():
        if normalized_query.lower() not in str(mention).lower():
            continue
        rows.append(
            {
                "mention": mention,
                "best_cui": payload.get("best_cui", ""),
                "best_name": payload.get("best_name", ""),
                "mapped": bool(payload.get("mapped", False)),
                "tuis": ", ".join(payload.get("tuis", []) or []),
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def _sort_node_ids_by_priority(index: GraphIndex, node_ids: np.ndarray) -> np.ndarray:
    if node_ids.size == 0:
        return node_ids
    unique_ids = np.unique(node_ids.astype(np.int32, copy=False))
    sort_order = np.lexsort((index.node_texts[unique_ids], -index.degrees[unique_ids]))
    return unique_ids[sort_order]


def _expand_node_ids(
    index: GraphIndex,
    base_edge_mask: np.ndarray,
    seed_node_ids: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius <= 0 or seed_node_ids.size == 0:
        return np.unique(seed_node_ids.astype(np.int32, copy=False))

    selected = np.unique(seed_node_ids.astype(np.int32, copy=False))
    frontier = selected.copy()
    base_source = index.source[base_edge_mask]
    base_target = index.target[base_edge_mask]

    for _ in range(radius):
        mask = np.isin(base_source, frontier) | np.isin(base_target, frontier)
        if not mask.any():
            break
        touched = np.unique(np.concatenate((base_source[mask], base_target[mask])))
        new_frontier = touched[~np.isin(touched, selected)]
        if new_frontier.size == 0:
            break
        selected = np.unique(np.concatenate((selected, new_frontier)))
        frontier = new_frontier

    return selected


def _truncate_node_ids(
    index: GraphIndex,
    display_node_ids: np.ndarray,
    priority_node_ids: np.ndarray,
    max_nodes: int,
) -> np.ndarray:
    if display_node_ids.size <= max_nodes:
        return _sort_node_ids_by_priority(index, display_node_ids)

    priority_sorted = _sort_node_ids_by_priority(index, priority_node_ids)
    remaining = display_node_ids[~np.isin(display_node_ids, priority_sorted)]
    remaining_sorted = _sort_node_ids_by_priority(index, remaining)
    combined = np.concatenate((priority_sorted, remaining_sorted))
    return combined[:max_nodes]


def _truncate_edge_positions(
    display_edge_positions: np.ndarray,
    matched_edge_positions: np.ndarray,
    max_edges: int,
) -> np.ndarray:
    if display_edge_positions.size <= max_edges:
        return display_edge_positions.astype(np.int32, copy=False)

    matched_first = display_edge_positions[np.isin(display_edge_positions, matched_edge_positions)]
    remaining = display_edge_positions[~np.isin(display_edge_positions, matched_first)]
    combined = np.concatenate((matched_first, remaining))
    return combined[:max_edges].astype(np.int32, copy=False)


def query_graph(index: GraphIndex, settings: QuerySettings) -> QueryResult:
    active_query = any(
        value.strip()
        for value in (settings.focus_query, settings.source_query, settings.target_query)
    )

    relation_names = settings.relation_names or tuple(
        relation
        for relation in index.relation_counts["relation"].tolist()
        if relation != "self_loop"
    )
    selected_codes = np.array(
        [RELATION_CODES[name] for name in relation_names if name in RELATION_CODES],
        dtype=np.int16,
    )
    base_edge_mask = np.isin(index.relation_code, selected_codes) if selected_codes.size else np.zeros(index.num_edges, dtype=bool)

    if settings.node_types:
        allowed_nodes = search_node_ids(index, query="", allowed_node_types=settings.node_types)
        base_edge_mask &= np.isin(index.source, allowed_nodes) & np.isin(index.target, allowed_nodes)

    source_node_ids = (
        search_node_ids(index, settings.source_query, settings.node_types)
        if settings.source_query.strip()
        else np.array([], dtype=np.int32)
    )
    target_node_ids = (
        search_node_ids(index, settings.target_query, settings.node_types)
        if settings.target_query.strip()
        else np.array([], dtype=np.int32)
    )
    focus_node_ids = (
        search_node_ids(index, settings.focus_query, settings.node_types)
        if settings.focus_query.strip()
        else np.array([], dtype=np.int32)
    )

    matched_edge_mask = base_edge_mask.copy()
    if settings.source_query.strip():
        matched_edge_mask &= np.isin(index.source, source_node_ids)
    if settings.target_query.strip():
        matched_edge_mask &= np.isin(index.target, target_node_ids)
    if settings.focus_query.strip() and not settings.source_query.strip() and not settings.target_query.strip():
        matched_edge_mask &= np.isin(index.source, focus_node_ids) | np.isin(index.target, focus_node_ids)

    matched_edge_positions = np.flatnonzero(matched_edge_mask).astype(np.int32, copy=False)
    matched_node_ids = np.unique(
        np.concatenate(
            [
                array
                for array in (
                    focus_node_ids,
                    source_node_ids,
                    target_node_ids,
                    index.source[matched_edge_positions],
                    index.target[matched_edge_positions],
                )
                if array.size
            ]
        )
        if any(
            array.size
            for array in (
                focus_node_ids,
                source_node_ids,
                target_node_ids,
                matched_edge_positions,
            )
        )
        else np.array([], dtype=np.int32)
    )

    too_broad = not active_query and (
        matched_edge_positions.size > settings.max_edges * 4 or matched_node_ids.size > settings.max_nodes * 4
    )
    truncated = False

    if too_broad:
        return QueryResult(
            matched_nodes_df=node_rows(index, matched_node_ids, limit=settings.preview_rows),
            matched_edges_df=edge_rows(index, matched_edge_positions, limit=settings.preview_rows),
            display_nodes_df=index.node_df.iloc[0:0].copy(),
            display_edges_df=edge_rows(index, np.array([], dtype=np.int32)),
            matched_node_count=int(matched_node_ids.size),
            matched_edge_count=int(matched_edge_positions.size),
            rendered_node_count=0,
            rendered_edge_count=0,
            needs_narrowing=True,
            truncated=False,
        )

    display_node_ids = matched_node_ids.copy()
    if display_node_ids.size and settings.expansion_radius > 0:
        display_node_ids = _expand_node_ids(
            index=index,
            base_edge_mask=base_edge_mask,
            seed_node_ids=display_node_ids,
            radius=settings.expansion_radius,
        )
    if display_node_ids.size > settings.max_nodes:
        display_node_ids = _truncate_node_ids(
            index=index,
            display_node_ids=display_node_ids,
            priority_node_ids=matched_node_ids,
            max_nodes=settings.max_nodes,
        )
        truncated = True

    if display_node_ids.size:
        display_edge_mask = base_edge_mask & np.isin(index.source, display_node_ids) & np.isin(index.target, display_node_ids)
        display_edge_positions = np.flatnonzero(display_edge_mask).astype(np.int32, copy=False)
    else:
        display_edge_positions = np.array([], dtype=np.int32)

    if display_edge_positions.size > settings.max_edges:
        display_edge_positions = _truncate_edge_positions(
            display_edge_positions=display_edge_positions,
            matched_edge_positions=matched_edge_positions,
            max_edges=settings.max_edges,
        )
        truncated = True

    display_nodes_df = node_rows(index, display_node_ids, limit=None)
    display_edges_df = edge_rows(index, display_edge_positions, limit=None)

    return QueryResult(
        matched_nodes_df=node_rows(index, matched_node_ids, limit=settings.preview_rows),
        matched_edges_df=edge_rows(index, matched_edge_positions, limit=settings.preview_rows),
        display_nodes_df=display_nodes_df,
        display_edges_df=display_edges_df,
        matched_node_count=int(matched_node_ids.size),
        matched_edge_count=int(matched_edge_positions.size),
        rendered_node_count=int(display_nodes_df.shape[0]),
        rendered_edge_count=int(display_edges_df.shape[0]),
        needs_narrowing=False,
        truncated=truncated,
    )


def build_pyvis_html(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> str:
    from pyvis.network import Network

    network = Network(
        height="760px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111827",
        directed=True,
        cdn_resources="in_line",
    )
    network.barnes_hut(gravity=-5000, central_gravity=0.18, spring_length=140, spring_strength=0.025)

    label_edges = edges_df.shape[0] <= 80
    for row in nodes_df.itertuples(index=False):
        node_type = str(row.node_type)
        ontology_suffix = "<br>Ontology-linked: yes" if bool(getattr(row, "is_ontology", False)) else ""
        title = (
            f"Text: {row.text}<br>"
            f"CUI: {row.cui or 'n/a'}<br>"
            f"Type: {node_type}<br>"
            f"Certainty: {row.certainty or 'n/a'}<br>"
            f"Degree: {row.degree}"
            f"{ontology_suffix}"
        )
        network.add_node(
            int(row.node_id),
            label=str(row.text),
            title=title,
            color=NODE_COLORS.get(node_type, NODE_COLORS["unknown"]),
        )

    for row in edges_df.itertuples(index=False):
        relation = str(row.relation)
        edge_weight = max(1.0, float(row.weight))
        network.add_edge(
            int(row.source),
            int(row.target),
            label=relation if label_edges else "",
            title=f"{relation}<br>Occurrences: {int(edge_weight)}",
            color=EDGE_COLORS.get(relation, "#94a3b8"),
            arrows="to",
            value=float(np.log1p(edge_weight)),
        )

    network.set_options(
        json.dumps(
            {
                "interaction": {
                    "hover": True,
                    "navigationButtons": True,
                    "keyboard": True,
                },
                "nodes": {
                    "shape": "dot",
                    "size": 18,
                    "font": {"size": 13, "face": "Helvetica"},
                },
                "edges": {
                    "smooth": {"enabled": True, "type": "dynamic"},
                    "font": {"align": "middle"},
                },
                "physics": {
                    "stabilization": {"iterations": 200},
                    "barnesHut": {
                        "gravitationalConstant": -5000,
                        "centralGravity": 0.18,
                        "springLength": 140,
                        "springConstant": 0.025,
                    },
                },
            }
        )
    )
    return network.generate_html(notebook=False)


def _render_phase1_summary(index: GraphIndex, log_summary: Phase1LogSummary | None, st: Any) -> None:
    st.subheader("Phase I Summary")
    summary_cols = st.columns(4)
    if "num_studies" in index.summary:
        summary_cols[0].metric("Studies", f"{int(index.summary['num_studies']):,}")
    if "total_entities" in index.summary:
        summary_cols[1].metric("Entities", f"{int(index.summary['total_entities']):,}")
    if "total_triples" in index.summary:
        summary_cols[2].metric("Triples", f"{int(index.summary['total_triples']):,}")
    coverage = index.summary.get("grounding_coverage", {}) or {}
    if coverage.get("coverage_pct") is not None:
        summary_cols[3].metric("Grounding Coverage", f"{float(coverage['coverage_pct']):.2f}%")

    if log_summary is not None:
        st.subheader("Phase I Log Audit")
        audit_cols = st.columns(5)
        audit_cols[0].metric("Run Status", "Complete" if log_summary.completed else "Incomplete")
        if log_summary.num_reports_loaded is not None:
            audit_cols[1].metric("Reports Loaded", f"{log_summary.num_reports_loaded:,}")
        if log_summary.report_graph_count is not None:
            audit_cols[2].metric("Per-Study Graphs", f"{log_summary.report_graph_count:,}")
        if log_summary.grounding_coverage_pct is not None:
            audit_cols[3].metric("Logged Coverage", f"{log_summary.grounding_coverage_pct:.2f}%")
        if log_summary.elapsed_seconds is not None:
            audit_cols[4].metric("Elapsed", f"{log_summary.elapsed_seconds / 3600:.2f} h")

        st.caption(f"Latest Phase I log: {log_summary.log_path}")
        if log_summary.graph_validation:
            st.code(log_summary.graph_validation, language="text")


def render_streamlit_app(config: AppConfig) -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="Knowledge Graph Viewer", layout="wide")
    st.title("Knowledge Graph Viewer")
    st.caption(
        "Standalone viewer for KG outputs with weighted-edge summaries, per-study browsing, certainty diagnostics, and optional Phase I log auditing."
    )

    @st.cache_resource(show_spinner=False)
    def _cached_global_index(path_str: str) -> GraphIndex:
        return load_graph_index(Path(path_str))

    @st.cache_resource(show_spinner=False)
    def _cached_study_catalog(path_str: str) -> StudyCatalog | None:
        return load_study_catalog(Path(path_str))

    @st.cache_resource(show_spinner=False)
    def _cached_phase1_log(path_str: str) -> Phase1LogSummary | None:
        return load_phase1_log_summary(Path(path_str))

    @st.cache_resource(show_spinner=False)
    def _cached_study_index(path_str: str, study_key: str) -> GraphIndex:
        artifact_dir = Path(path_str)
        cache_path = ensure_cached_study_graph(artifact_dir, study_key)
        study_graph = torch.load(cache_path, map_location="cpu", weights_only=False)
        return build_graph_index(
            graph=study_graph,
            artifact_dir=artifact_dir,
            summary=load_summary(artifact_dir),
            graph_label=f"Per-study graph: {study_key}",
            study_key=study_key,
        )

    try:
        with st.spinner("Loading global KG artifacts and metadata..."):
            global_index = _cached_global_index(str(config.artifact_dir))
            study_catalog = _cached_study_catalog(str(config.artifact_dir))
            log_summary = _cached_phase1_log(str(config.artifact_dir))
    except Exception as exc:
        st.error(f"Failed to load graph artifacts from {config.artifact_dir}: {exc}")
        st.stop()

    active_index = global_index
    selected_scope = "Global graph"
    selected_study_key: str | None = None
    matched_studies_df = pd.DataFrame()
    st.session_state.setdefault("kg_viewer_loaded_study_key", None)

    with st.sidebar:
        st.header("Scope")
        selected_scope = st.radio(
            "Graph scope",
            options=["Global graph", "Per-study graph"],
            index=0,
        )

        if selected_scope == "Per-study graph":
            if study_catalog is None or study_catalog.study_df.empty:
                st.warning("study_metadata.pt was not found, so per-study browsing is unavailable.")
                selected_scope = "Global graph"
            else:
                sample_study_key = study_catalog.study_df.iloc[0]["study_key"]
                study_query = st.text_input(
                    "Study key / subject / study",
                    value="",
                    placeholder=sample_study_key,
                    help="Examples: full study key `10000032_50414267`, a subject ID, or a study ID.",
                )
                if study_query.strip():
                    matched_studies_df = filter_study_catalog(study_catalog, study_query)
                    if matched_studies_df.empty:
                        st.warning("No studies matched that query.")
                        selected_scope = "Global graph"
                    else:
                        selected_study_key = st.selectbox(
                            "Matched studies",
                            options=matched_studies_df["study_key"].tolist(),
                            index=0,
                        )
                        st.caption(
                            f"Showing {matched_studies_df.shape[0]:,} matches from {study_catalog.num_studies:,} indexed studies."
                        )
                        cached_path = study_graph_cache_path(config.artifact_dir, selected_study_key)
                        if cached_path.exists():
                            st.caption("A cached extracted study graph is already available for this study.")

                        if st.button("Load selected study graph", use_container_width=True, type="primary"):
                            st.session_state["kg_viewer_loaded_study_key"] = selected_study_key

                        if st.session_state.get("kg_viewer_loaded_study_key") == selected_study_key:
                            spinner_message = (
                                "Loading cached study graph..."
                                if cached_path.exists()
                                else "Extracting the selected study graph in a worker process. The first extraction can take a few minutes because report_graphs.pt is a monolithic archive."
                            )
                            try:
                                with st.spinner(spinner_message):
                                    active_index = _cached_study_index(str(config.artifact_dir), selected_study_key)
                            except Exception as exc:
                                st.error(f"Failed to load study graph `{selected_study_key}`: {exc}")
                                selected_scope = "Global graph"
                                st.session_state["kg_viewer_loaded_study_key"] = None
                            else:
                                st.success(f"Loaded per-study graph `{selected_study_key}`.")
                        else:
                            st.info(
                                "Choose a matching study and click `Load selected study graph`. The first extraction for a study can take a few minutes, then the extracted graph is cached for reuse."
                            )
                            selected_scope = "Global graph"
                else:
                    st.info("Enter a study key or ID to load a single report graph. Until then the viewer stays on the global graph.")
                    selected_scope = "Global graph"

        st.header("Filters")
        focus_query = st.text_input("Node text or CUI contains", value="")
        source_query = st.text_input("Source node contains", value="")
        target_query = st.text_input("Target node contains", value="")

        all_node_types = [str(value) for value in active_index.node_df["node_type"].cat.categories.tolist()]
        relation_options = active_index.relation_counts["relation"].tolist()
        default_relations = [relation for relation in relation_options if relation != "self_loop"]

        selected_node_types = st.multiselect(
            "Node types",
            options=all_node_types,
            default=all_node_types,
        )
        selected_relations = st.multiselect(
            "Relation types",
            options=relation_options,
            default=default_relations,
        )
        expansion_radius = st.slider("Neighborhood hops", min_value=0, max_value=2, value=1)
        max_nodes = st.slider("Max nodes in graph panel", min_value=20, max_value=300, value=90, step=10)
        max_edges = st.slider("Max edges in graph panel", min_value=20, max_value=600, value=220, step=20)
        preview_rows = st.slider("Preview rows in tables", min_value=25, max_value=500, value=150, step=25)

    settings = QuerySettings(
        focus_query=focus_query,
        source_query=source_query,
        target_query=target_query,
        node_types=tuple(selected_node_types),
        relation_names=tuple(selected_relations),
        expansion_radius=expansion_radius,
        max_nodes=max_nodes,
        max_edges=max_edges,
        preview_rows=preview_rows,
    )
    result = query_graph(active_index, settings)

    st.caption(f"Active graph: {active_index.graph_label}")
    if selected_scope == "Per-study graph" and selected_study_key is not None:
        st.info(
            f"Per-study mode is active for `{selected_study_key}`. Search filters now run only inside that single report graph."
        )

    metrics = st.columns(5)
    metrics[0].metric("Nodes", f"{active_index.num_nodes:,}")
    metrics[1].metric("Unique Edges", f"{active_index.num_edges:,}")
    metrics[2].metric("Raw Edge Occurrences", f"{active_index.raw_num_edges:,}")
    metrics[3].metric("Matched Relations", f"{result.matched_edge_count:,}")
    metrics[4].metric("Rendered Subgraph", f"{result.rendered_node_count:,} / {result.rendered_edge_count:,}")

    overview_tab, search_tab, graph_tab, diagnostics_tab = st.tabs(
        ["Overview", "Search Results", "Graph", "Diagnostics"]
    )

    with overview_tab:
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.subheader("Relation Distribution")
            relation_chart = active_index.relation_counts.set_index("relation")
            st.bar_chart(relation_chart)
            st.dataframe(active_index.relation_counts, width="stretch", hide_index=True)
        with stats_cols[1]:
            st.subheader("Node Type Distribution")
            node_type_chart = active_index.node_type_counts.set_index("node_type")
            st.bar_chart(node_type_chart)
            st.dataframe(active_index.node_type_counts, width="stretch", hide_index=True)
        with stats_cols[2]:
            st.subheader("Certainty Distribution")
            certainty_chart = active_index.certainty_counts.set_index("certainty")
            st.bar_chart(certainty_chart)
            st.dataframe(active_index.certainty_counts, width="stretch", hide_index=True)

        _render_phase1_summary(active_index, log_summary, st)

    with search_tab:
        if result.needs_narrowing:
            st.info(
                "The current selection is too broad to render safely. Add a node/CUI, source, or target search to isolate a smaller subgraph."
            )
        elif result.matched_edge_count == 0 and result.matched_node_count == 0:
            st.warning("No nodes or relations matched the current filters.")

        if result.truncated:
            st.warning(
                "The rendered subgraph was truncated to respect the node and edge limits. Increase the limits or narrow the filters for more detail."
            )

        st.subheader("Matched Nodes")
        st.dataframe(result.matched_nodes_df, width="stretch", hide_index=True)
        st.download_button(
            label="Download matched nodes CSV",
            data=result.matched_nodes_df.to_csv(index=False),
            file_name="kg_viewer_matched_nodes.csv",
            mime="text/csv",
        )

        st.subheader("Matched Relations")
        st.dataframe(result.matched_edges_df, width="stretch", hide_index=True)
        st.download_button(
            label="Download matched relations CSV",
            data=result.matched_edges_df.to_csv(index=False),
            file_name="kg_viewer_matched_relations.csv",
            mime="text/csv",
        )

        if selected_study_key is not None and not matched_studies_df.empty:
            st.subheader("Matched Study Candidates")
            st.dataframe(matched_studies_df, width="stretch", hide_index=True)

    with graph_tab:
        if result.needs_narrowing:
            st.info("Render a narrower selection to open the interactive graph view.")
        elif result.display_edges_df.empty or result.display_nodes_df.empty:
            st.info("No renderable subgraph is available for the current filters.")
        else:
            st.subheader("Interactive Subgraph")
            graph_html = build_pyvis_html(result.display_nodes_df, result.display_edges_df)
            components.html(graph_html, height=780, scrolling=True)
            st.dataframe(result.display_edges_df, width="stretch", hide_index=True)

    with diagnostics_tab:
        st.subheader("Top Nodes by Weighted Degree")
        st.dataframe(top_nodes_by_degree(active_index, limit=preview_rows), width="stretch", hide_index=True)

        st.subheader("Heaviest Weighted Edges")
        st.dataframe(top_edges_by_weight(active_index, limit=preview_rows), width="stretch", hide_index=True)

        if study_catalog is not None:
            st.subheader("Grounding Lookup")
            grounding_query = st.text_input("Mention contains", value="")
            grounding_rows = grounding_lookup_rows(study_catalog, grounding_query, limit=preview_rows)
            if grounding_query.strip() and grounding_rows.empty:
                st.info("No grounding entries matched that mention query.")
            elif not grounding_rows.empty:
                st.dataframe(grounding_rows, width="stretch", hide_index=True)


def _is_running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


def _launch_streamlit(script_path: Path, config: AppConfig) -> None:
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(script_path),
        f"--server.address={config.host}",
        f"--server.port={config.port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--",
        "--artifact-dir",
        str(config.artifact_dir),
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    raise SystemExit(stcli.main())


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    if config.extract_study_graph is not None:
        extract_study_graph(
            artifact_dir=config.artifact_dir,
            study_key=config.extract_study_graph,
            destination_path=config.study_cache_path,
        )
        return
    if _is_running_under_streamlit():
        render_streamlit_app(config)
        return
    _launch_streamlit(script_path=repo_root() / "scripts" / "visualize_kg.py", config=config)


if __name__ == "__main__":
    main()
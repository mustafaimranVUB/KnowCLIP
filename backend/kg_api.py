"""Knowledge Graph viewer routes for the KnoCLIP-XAI API.

Reuses kg_viewer/app.py data functions and serves them as REST endpoints,
replacing the standalone Streamlit kg_viewer with integrated API routes.

All endpoints are lazy: the global KG index is loaded on first request and
cached for the lifetime of the process. Per-study graphs are extracted via
memory-mapped loading and cached to disk under KG_ARTIFACTS_DIR/.kg_viewer_cache/.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger("knoclip_api.kg")

router = APIRouter(prefix="/kg", tags=["knowledge-graph"])

# ---------------------------------------------------------------------------
# Singleton KG state (lazy-loaded on first request)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_kg_index: Any = None
_kg_catalog: Any = None
_kg_artifact_dir: Optional[Path] = None
_kg_available: bool = False
_kg_load_error: str = ""


def _get_configured_dir() -> Optional[Path]:
    raw = os.environ.get("KG_ARTIFACTS_DIR", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _ensure_kg_loaded() -> None:
    global _kg_index, _kg_catalog, _kg_artifact_dir, _kg_available, _kg_load_error

    if _kg_available:
        return

    with _lock:
        if _kg_available:
            return

        artifact_dir = _get_configured_dir()
        if artifact_dir is None:
            _kg_load_error = "KG_ARTIFACTS_DIR is not set."
            return
        if not artifact_dir.exists():
            _kg_load_error = f"KG_ARTIFACTS_DIR does not exist: {artifact_dir}"
            return
        if not (artifact_dir / "global_kg.pt").exists():
            _kg_load_error = f"global_kg.pt not found in {artifact_dir}"
            return

        try:
            from kg_viewer.app import load_graph_index, load_study_catalog  # noqa: PLC0415

            logger.info("Loading global KG index from %s ...", artifact_dir)
            _kg_index = load_graph_index(artifact_dir)
            _kg_catalog = load_study_catalog(artifact_dir)
            _kg_artifact_dir = artifact_dir
            _kg_available = True
            logger.info(
                "KG index ready: %d nodes, %d unique edges",
                _kg_index.num_nodes,
                _kg_index.num_edges,
            )
        except Exception as exc:
            _kg_load_error = str(exc)
            logger.exception("Failed to load KG index: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph_index_to_vis_json(
    index: Any,
    max_nodes: int = 150,
    max_edges: int = 400,
) -> Dict[str, Any]:
    """Convert a GraphIndex to vis-network compatible nodes/edges JSON."""
    from kg_viewer.app import RELATION_LABELS, NODE_COLORS, EDGE_COLORS  # noqa: PLC0415

    node_df = index.node_df.sort_values("degree", ascending=False).head(max_nodes)
    node_id_set: set = set()
    nodes: List[Dict[str, Any]] = []
    for row in node_df.itertuples(index=False):
        node_type = str(row.node_type)
        node_id_set.add(int(row.node_id))
        nodes.append(
            {
                "id": int(row.node_id),
                "label": str(row.text)[:36],
                "title": f"{row.text}<br>CUI: {row.cui or 'n/a'}<br>Type: {node_type}<br>Degree: {row.degree}",
                "color": NODE_COLORS.get(node_type, NODE_COLORS["unknown"]),
                "node_type": node_type,
                "cui": str(row.cui or ""),
            }
        )

    edges: List[Dict[str, Any]] = []
    for i in range(len(index.source)):
        if len(edges) >= max_edges:
            break
        src, tgt = int(index.source[i]), int(index.target[i])
        if src not in node_id_set or tgt not in node_id_set:
            continue
        rel = RELATION_LABELS.get(int(index.relation_code[i]), "unknown")
        if rel == "self_loop":
            continue
        edges.append(
            {
                "from": src,
                "to": tgt,
                "label": rel,
                "title": f"{rel} (×{int(index.edge_weight[i])})",
                "color": EDGE_COLORS.get(rel, "#94a3b8"),
                "arrows": "to",
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "num_nodes": index.num_nodes,
        "num_unique_edges": index.num_edges,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
def kg_status() -> JSONResponse:
    """Whether the KG index is loaded and which artifacts are present."""
    _ensure_kg_loaded()
    return JSONResponse(
        {
            "available": _kg_available,
            "error": _kg_load_error if not _kg_available else None,
            "artifact_dir": str(_kg_artifact_dir) if _kg_artifact_dir else None,
            "has_study_catalog": bool(
                _kg_catalog is not None and not _kg_catalog.study_df.empty
            ),
            "has_report_graphs": bool(
                _kg_artifact_dir is not None
                and (_kg_artifact_dir / "report_graphs.pt").exists()
            ),
        }
    )


@router.get("/stats")
def kg_stats() -> JSONResponse:
    """Global KG statistics: node/edge counts, type distributions, top nodes."""
    _ensure_kg_loaded()
    if not _kg_available or _kg_index is None:
        raise HTTPException(status_code=503, detail=f"KG not available: {_kg_load_error}")

    from kg_viewer.app import top_nodes_by_degree  # noqa: PLC0415

    index = _kg_index
    top_nodes = top_nodes_by_degree(index, limit=20)

    return JSONResponse(
        {
            "num_nodes": index.num_nodes,
            "num_unique_edges": index.num_edges,
            "num_raw_edges": index.raw_num_edges,
            "num_studies": _kg_catalog.num_studies if _kg_catalog else None,
            "relation_distribution": index.relation_counts[["relation", "count"]].to_dict(
                orient="records"
            ),
            "node_type_distribution": index.node_type_counts[["node_type", "count"]].to_dict(
                orient="records"
            ),
            "top_nodes_by_degree": top_nodes[["text", "cui", "node_type", "degree"]].to_dict(
                orient="records"
            ),
        }
    )


@router.get("/nodes")
def kg_search_nodes(
    q: str = Query(default="", description="Substring to match against node text or CUI"),
    limit: int = Query(default=30, ge=1, le=200),
) -> JSONResponse:
    """Search nodes in the global KG by text or CUI."""
    _ensure_kg_loaded()
    if not _kg_available or _kg_index is None:
        raise HTTPException(status_code=503, detail=f"KG not available: {_kg_load_error}")

    from kg_viewer.app import search_node_ids, node_rows  # noqa: PLC0415

    node_ids = search_node_ids(_kg_index, q)
    rows = node_rows(_kg_index, node_ids, limit=limit)
    return JSONResponse(
        {
            "query": q,
            "total_matched": int(node_ids.size),
            "nodes": rows[["node_id", "text", "cui", "node_type", "degree"]].to_dict(
                orient="records"
            ),
        }
    )


@router.get("/graph/global")
def kg_global_graph(
    max_nodes: int = Query(default=80, ge=10, le=300),
    max_edges: int = Query(default=200, ge=10, le=600),
) -> JSONResponse:
    """Top nodes by degree from the global KG, formatted for vis-network."""
    _ensure_kg_loaded()
    if not _kg_available or _kg_index is None:
        raise HTTPException(status_code=503, detail=f"KG not available: {_kg_load_error}")

    return JSONResponse(
        _graph_index_to_vis_json(_kg_index, max_nodes=max_nodes, max_edges=max_edges)
    )


@router.get("/study/{subject_id}/{study_id}")
def kg_study_graph(subject_id: int, study_id: int) -> JSONResponse:
    """Per-study KG formatted for vis-network.

    Extracts the study graph from report_graphs.pt via memory-mapped loading.
    First call per study caches the result under KG_ARTIFACTS_DIR/.kg_viewer_cache/;
    subsequent calls return from cache immediately.
    """
    _ensure_kg_loaded()
    if not _kg_available:
        raise HTTPException(status_code=503, detail=f"KG not available: {_kg_load_error}")

    if _kg_artifact_dir is None:
        raise HTTPException(status_code=503, detail="KG artifact directory not configured.")

    if not (_kg_artifact_dir / "report_graphs.pt").exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "report_graphs.pt is not present. Per-study graphs require the full "
                "17 GB artifact to be synced to Hydra."
            ),
        )

    study_key = f"{subject_id}_{study_id}"
    try:
        import torch
        from kg_viewer.app import (  # noqa: PLC0415
            ensure_cached_study_graph,
            build_graph_index,
            load_summary,
        )

        cache_path = ensure_cached_study_graph(_kg_artifact_dir, study_key)
        study_graph = torch.load(cache_path, map_location="cpu", weights_only=False)
        index = build_graph_index(
            graph=study_graph,
            artifact_dir=_kg_artifact_dir,
            summary=load_summary(_kg_artifact_dir),
            graph_label=f"Study {study_key}",
            study_key=study_key,
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Study {study_key} not found in report_graphs.pt.",
        )
    except Exception as exc:
        logger.exception("Failed to extract study graph %s: %s", study_key, exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to extract study graph: {exc}"
        ) from exc

    result = _graph_index_to_vis_json(index, max_nodes=150, max_edges=400)
    result["study_key"] = study_key
    return JSONResponse(result)

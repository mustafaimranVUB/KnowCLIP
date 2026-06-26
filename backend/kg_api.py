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

# Cache for report_graphs.pt (loaded once, reused for all per-study KG requests)
_report_graphs_lock = threading.Lock()
_report_graphs: Any = None
_report_graphs_loaded: bool = False


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
        from kg_viewer.app import build_graph_index, load_summary  # noqa: PLC0415

        # Load report_graphs.pt once and cache it — bypasses the missing
        # scripts/visualize_kg.py subprocess that ensure_cached_study_graph uses.
        global _report_graphs, _report_graphs_loaded
        if not _report_graphs_loaded:
            with _report_graphs_lock:
                if not _report_graphs_loaded:
                    report_graphs_path = _kg_artifact_dir / "report_graphs.pt"
                    logger.info("Loading report_graphs.pt for study KG endpoint...")
                    _report_graphs = torch.load(
                        report_graphs_path, map_location="cpu", weights_only=False
                    )
                    _report_graphs_loaded = True
                    logger.info("report_graphs.pt cached (%d studies).", len(_report_graphs))

        study_graph = _report_graphs.get(study_key) if _report_graphs else None
        if study_graph is None:
            raise KeyError(study_key)

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


# ---------------------------------------------------------------------------
# Trace bundle viewer
# ---------------------------------------------------------------------------

_XAI_DIR = Path("inference_explainability")


@router.get("/trace/{study_key}")
def kg_trace_data(study_key: str) -> JSONResponse:
    """Load trace_bundle.pt for a study and return JSON-serialisable attention data.

    Used by the frontend to render the interactive token-concept attention heatmap.
    """
    import torch  # noqa: PLC0415

    bundle_path = _XAI_DIR / study_key / "trace_bundle.pt"
    if not bundle_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"trace_bundle.pt not found for study {study_key}. Run inference with explainability enabled.",
        )

    try:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load trace bundle: {exc}") from exc

    result: Dict[str, Any] = {"study_key": study_key}

    # ── Token labels & classification ────────────────────────────────────────
    result["generated_token_labels"] = list(bundle.get("generated_token_labels", []))
    result["graph_node_texts"]       = list(bundle.get("graph_node_texts", []))
    result["graph_node_types"]       = list(bundle.get("graph_node_types", []))
    result["graph_node_certainties"] = list(bundle.get("graph_node_certainties", []))

    probs = bundle.get("classification_probs")
    if isinstance(probs, torch.Tensor):
        result["classification_probs"] = probs.detach().cpu().tolist()

    exp = bundle.get("explainability", {})
    if not isinstance(exp, dict):
        return JSONResponse(result)

    # ── Decoder token-to-concept attention ──────────────────────────────────
    # Shape: list of [batch, heads, seq_len, num_queries+num_patches]
    decoder = exp.get("decoder", {})
    if isinstance(decoder, dict):
        per_layer = decoder.get("per_layer", [])
        layer_tensors = [
            layer.detach().cpu()
            for layer in per_layer
            if isinstance(layer, torch.Tensor) and layer.ndim == 4
        ]
        if layer_tensors:
            stacked = torch.stack([layer[0] for layer in layer_tensors], dim=0).mean(dim=(0, 1))
            concept_count = len(result["graph_node_texts"])
            concept_attn = stacked[:, :concept_count]  # [seq, concepts]
            result["decoder_token_concept_attn"] = concept_attn.tolist()

    # ── Pooling weights (concept importance) ────────────────────────────────
    pooling = exp.get("pooling", {})
    if isinstance(pooling, dict):
        weights = pooling.get("weights")
        mask    = pooling.get("mask")
        if isinstance(weights, torch.Tensor):
            w = weights.detach().cpu()
            if w.ndim == 2:
                w = w[0]
            if isinstance(mask, torch.Tensor):
                m = mask.detach().cpu()
                if m.ndim == 2:
                    m = m[0]
                w = w[:int(m.sum().item())]
            concept_count = len(result["graph_node_texts"])
            result["concept_importance"] = w[:concept_count].tolist()

    # ── GATv2 top edges (last layer) ────────────────────────────────────────
    graph = exp.get("graph", {})
    if isinstance(graph, dict):
        layers = graph.get("edge_attention_layers", [])
        if layers:
            last = layers[-1]
            edge_index = last.get("edge_index")
            alpha      = last.get("alpha")
            edge_type  = last.get("edge_type")
            if isinstance(edge_index, torch.Tensor) and isinstance(alpha, torch.Tensor):
                ei = edge_index.detach().cpu()
                al = alpha.detach().cpu()
                scores = al.mean(dim=-1) if al.ndim > 1 else al
                top_k  = min(20, int(scores.numel()))
                top_idx = torch.topk(scores, k=top_k).indices.tolist()
                node_texts = result["graph_node_texts"]
                et_cpu = edge_type.detach().cpu() if isinstance(edge_type, torch.Tensor) else None
                edges: List[Dict[str, Any]] = []
                for idx in top_idx:
                    src = int(ei[0, idx])
                    dst = int(ei[1, idx])
                    rel = int(et_cpu[idx].item()) if et_cpu is not None and idx < et_cpu.numel() else -1
                    edges.append({
                        "src": node_texts[src] if src < len(node_texts) else f"node_{src}",
                        "dst": node_texts[dst] if dst < len(node_texts) else f"node_{dst}",
                        "relation": rel,
                        "score": float(scores[idx]),
                    })
                result["top_edges"] = edges

    return JSONResponse(result)

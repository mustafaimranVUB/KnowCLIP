"""KnoCLIP-XAI Inference API.

Wraps InferencePipeline as a real-time REST endpoint.

Environment variables (set before starting):
    MODEL_CONFIG      - path to YAML config (required)
    MODEL_CHECKPOINT  - path to best_model.pt (required)
    KG_ARTIFACTS_DIR  - path to KG artifacts dir (optional)
    DEVICE            - cpu / cuda:0 / ... (optional, auto-detected if absent)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Optional system-monitoring deps — degrade gracefully if absent.
try:
    import psutil as _psutil
    _HAVE_PSUTIL = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    _HAVE_PSUTIL = False

_pynvml = None
_HAVE_PYNVML = False
try:
    import pynvml as _pynvml_mod  # type: ignore[import]
    _pynvml_mod.nvmlInit()
    _pynvml = _pynvml_mod
    _HAVE_PYNVML = True
except Exception:  # noqa: BLE001
    pass

from backend.kg_api import router as kg_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("knoclip_api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

CHEXPERT_CLASSES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# ---------------------------------------------------------------------------
# Singleton model state
# ---------------------------------------------------------------------------
_pipeline: Any = None  # InferencePipeline instance (lazy-loaded on startup)
_checkpoint_path: str = ""
_model_loaded: bool = False
_device_str: str = "unknown"
_thresholds: Dict[str, float] = {}
_thresholds_source: str = ""

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="KnoCLIP-XAI Inference API",
    description="Real-time radiology report generation and CheXpert classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kg_router)

# Serve explainability PNGs directly — mounted after router so /kg/* takes priority.
# The directory is created lazily on first inference; mount fails gracefully if absent.
_XAI_DIR = Path("inference_explainability")
_XAI_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/explainability", StaticFiles(directory=str(_XAI_DIR), html=False), name="explainability")


@app.middleware("http")
async def _no_cache_xai(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/explainability/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _local_path_to_xai_url(path_str: str) -> str:
    """Convert a local explainability file path to its /explainability/... URL.

    Finds the inference_explainability segment and strips everything before it.
    e.g. D:/KnoClip/.../inference_explainability/10000032_50414267/cam_x.png
         → /explainability/10000032_50414267/cam_x.png
    """
    parts = Path(path_str).parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == _XAI_DIR.name)
        return "/explainability/" + "/".join(parts[idx + 1:])
    except StopIteration:
        return path_str  # not under the XAI dir; return unchanged


def _xai_paths_to_urls(summary: Any) -> Any:
    """Recursively replace local .png file paths with /explainability/... URLs."""
    if isinstance(summary, dict):
        out: Dict[str, Any] = {}
        for k, v in summary.items():
            if isinstance(v, str) and v.endswith(".png"):
                out[k] = _local_path_to_xai_url(v)
            elif isinstance(v, (dict, list)):
                out[k] = _xai_paths_to_urls(v)
            else:
                out[k] = v
        return out
    if isinstance(summary, list):
        return [_xai_paths_to_urls(item) for item in summary]
    return summary


# ---------------------------------------------------------------------------
# Decision thresholds (per-class, tuned on the validation split during evaluate)
# ---------------------------------------------------------------------------
def _extract_thresholds(obj: Any, class_names: list) -> Dict[str, float]:
    """Pull a {class: threshold} mapping out of a metrics.json / thresholds.json."""
    cand: Any = None
    if isinstance(obj, dict):
        ct = obj.get("classification_thresholds")
        cl = obj.get("classification")
        if isinstance(ct, dict):
            cand = ct
        elif isinstance(cl, dict) and isinstance(cl.get("thresholds"), dict):
            cand = cl["thresholds"]
        elif obj and all(isinstance(v, (int, float)) for v in obj.values()):
            cand = obj  # already a flat {class: threshold} file
    if not isinstance(cand, dict):
        return {}
    known = set(class_names)
    out: Dict[str, float] = {}
    for key, value in cand.items():
        if key in known:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def _load_thresholds(checkpoint: str, class_names: list) -> tuple[Dict[str, float], str]:
    """Discover per-class decision thresholds without needing the val dataset.

    Search order:
      1. $CLASSIFICATION_THRESHOLDS (explicit path to metrics.json or thresholds.json)
      2. <checkpoint_dir>/thresholds.json, <checkpoint_dir>/metrics.json
      3. <root>/outputs/**/evaluation/<run>/**/metrics.json for root in
         [cwd, $OUTPUTS_ROOT]  (matches the `evaluate` artifact layout)
    Returns (thresholds, source_path); ({}, "") if nothing is found.
    """
    ckpt = Path(checkpoint)
    run = ckpt.parent.name
    candidates: list[Path] = []

    env_path = os.environ.get("CLASSIFICATION_THRESHOLDS", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(ckpt.parent / "thresholds.json")
    candidates.append(ckpt.parent / "metrics.json")

    roots = [Path.cwd()]
    extra_root = os.environ.get("OUTPUTS_ROOT", "").strip()
    if extra_root:
        roots.append(Path(extra_root).expanduser())
    for root in roots:
        out_dir = root / "outputs"
        if not out_dir.is_dir():
            continue
        for pattern in (
            f"**/evaluation/{run}/best_model/metrics.json",
            f"**/evaluation/{run}/**/metrics.json",
            f"**/{run}/**/metrics.json",
        ):
            try:
                candidates.extend(sorted(out_dir.glob(pattern)))
            except OSError:
                continue

    seen: set = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        thresholds = _extract_thresholds(obj, class_names)
        if thresholds:
            return thresholds, str(path)
    return {}, ""


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    """Load the model into memory once at startup."""
    global _pipeline, _checkpoint_path, _model_loaded, _device_str, _thresholds, _thresholds_source

    config_path = os.environ.get("MODEL_CONFIG", "").strip()
    checkpoint = os.environ.get("MODEL_CHECKPOINT", "").strip()
    kg_dir = os.environ.get("KG_ARTIFACTS_DIR", "").strip()
    device = os.environ.get("DEVICE", "").strip() or None

    if not config_path:
        logger.error("MODEL_CONFIG env var is required but not set.")
        return
    if not checkpoint:
        logger.error("MODEL_CHECKPOINT env var is required but not set.")
        return
    if not Path(config_path).exists():
        logger.error("MODEL_CONFIG path does not exist: %s", config_path)
        return
    if not Path(checkpoint).exists():
        logger.error("MODEL_CHECKPOINT path does not exist: %s", checkpoint)
        return

    # Propagate KG_ARTIFACTS_DIR so the pipeline config can pick it up via
    # the standard override mechanism used elsewhere in the codebase.
    if kg_dir:
        os.environ["KG_ARTIFACTS_DIR"] = kg_dir
        logger.info("KG_ARTIFACTS_DIR set to: %s", kg_dir)

    try:
        # Import here to avoid polluting the module namespace at import time
        # and to allow the server to start even if src is not yet on sys.path
        # when the file is first loaded by uvicorn.
        from src.pipelines.inference_pipeline import InferencePipeline  # noqa: PLC0415

        logger.info("Loading InferencePipeline (config=%s, device=%s)...", config_path, device or "auto")
        _pipeline = InferencePipeline(config=config_path, device=device)
        _checkpoint_path = checkpoint
        _device_str = str(_pipeline.device)
        _model_loaded = True
        try:
            _class_names = list(_pipeline.config.model.classification_head.class_names)
            _thresholds, _thresholds_source = _load_thresholds(checkpoint, _class_names)
            if _thresholds:
                logger.info("Loaded %d classification thresholds from %s", len(_thresholds), _thresholds_source)
            else:
                logger.warning(
                    "No classification thresholds found (raw probabilities will be shown). "
                    "Set CLASSIFICATION_THRESHOLDS=/path/to/metrics.json to enable thresholded decisions."
                )
        except Exception as _thr_exc:  # noqa: BLE001
            logger.warning("Threshold loading skipped: %s", _thr_exc)
        # Eagerly build the model + load checkpoint at startup so the first
        # /predict request is not delayed by model construction.
        logger.info("Pre-loading model onto %s (this may take ~30 s)...", _device_str)
        _pipeline._prepare_model(checkpoint)

        # Share the report_graphs cache with the KG API so report_graphs.pt is
        # only loaded once (it's ~17 GB — loading it twice would waste 34 GB RAM).
        try:
            from backend.kg_api import _report_graphs_lock as _kg_rg_lock  # noqa: PLC0415
            import backend.kg_api as _kg_api_mod  # noqa: PLC0415
            if _pipeline._report_graphs_loaded and _pipeline._report_graphs is not None:
                with _kg_rg_lock:
                    _kg_api_mod._report_graphs = _pipeline._report_graphs
                    _kg_api_mod._report_graphs_loaded = True
                logger.info("Shared report_graphs cache with KG API.")
        except Exception as _share_exc:  # noqa: BLE001
            logger.debug("Could not share report_graphs cache: %s", _share_exc)

        logger.info(
            "KnoCLIP-XAI API ready | device=%s | config=%s | checkpoint=%s",
            _device_str,
            config_path,
            checkpoint,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load InferencePipeline: %s", exc)
        _model_loaded = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _resource_snapshot() -> Dict[str, Any]:
    """Return a JSON-serialisable snapshot of current CPU/RAM/GPU usage.

    All fields are best-effort: missing deps or unavailable devices silently
    omit the relevant keys so callers never need to guard against KeyError.
    """
    snap: Dict[str, Any] = {}

    if _HAVE_PSUTIL and _psutil is not None:
        try:
            snap["cpu_percent"] = _psutil.cpu_percent(interval=None)
            vm = _psutil.virtual_memory()
            snap["ram_used_mb"] = round(vm.used / 1e6, 1)
            snap["ram_total_mb"] = round(vm.total / 1e6, 1)
            snap["process_rss_mb"] = round(_psutil.Process().memory_info().rss / 1e6, 1)
        except Exception:  # noqa: BLE001
            pass

    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            snap["gpu_name"] = torch.cuda.get_device_name(dev)
            snap["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated(dev) / 1e6, 1)
            snap["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved(dev) / 1e6, 1)
            snap["gpu_mem_total_mb"] = round(
                torch.cuda.get_device_properties(dev).total_memory / 1e6, 1
            )
            if _HAVE_PYNVML and _pynvml is not None:
                try:
                    handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
                    snap["gpu_util_pct"] = util.gpu
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass

    return snap


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness / readiness probe."""
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": _model_loaded,
            "device": _device_str,
        }
    )


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """Live server resource snapshot (CPU/RAM/GPU). Safe to poll frequently."""
    try:
        return JSONResponse(content=_resource_snapshot())
    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics snapshot failed: %s", exc)
        return JSONResponse(content={})


@app.get("/classes")
async def get_classes() -> JSONResponse:
    """Return the 14 CheXpert class names used by the classifier."""
    if _pipeline is not None:
        try:
            class_names = list(_pipeline.config.model.classification_head.class_names)
            return JSONResponse(content={"classes": class_names})
        except Exception:  # noqa: BLE001
            pass
    return JSONResponse(content={"classes": CHEXPERT_CLASSES})


@app.get("/config")
async def get_config() -> JSONResponse:
    """Return non-sensitive model configuration information."""
    if not _model_loaded or _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cfg = _pipeline.config
    try:
        info: Dict[str, Any] = {
            "use_kg": cfg.model.use_kg,
            "enable_classification": cfg.model.enable_classification,
            "enable_report_generation": cfg.model.enable_report_generation,
            "backbone_type": cfg.model.visual_encoder.backbone_type,
            "image_size": cfg.model.visual_encoder.image_size,
            "decoder_type": cfg.model.report_generation.decoder_type,
            "max_report_length": cfg.model.report_generation.max_report_length,
            "num_classes": len(cfg.model.classification_head.class_names),
            "class_names": list(cfg.model.classification_head.class_names),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to read config: {exc}") from exc

    return JSONResponse(content=info)


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Chest X-ray image (JPG or PNG)"),
    subject_id: Optional[int] = Form(default=None),
    study_id: Optional[int] = Form(default=None),
    save_explainability: bool = Form(default=False),
) -> JSONResponse:
    """Run classification + report generation on the uploaded image."""
    if not _model_loaded or _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    # ---- validate file extension ----
    original_filename = image.filename or "upload"
    suffix = Path(original_filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # ---- read & validate file size ----
    image_bytes = await image.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File too large ({len(image_bytes)} bytes). Maximum allowed is {MAX_FILE_SIZE_BYTES} bytes.",
        )

    tmp_dir: Optional[str] = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="knoclip_")
        tmp_image_path = Path(tmp_dir) / f"input{suffix}"
        tmp_image_path.write_bytes(image_bytes)

        # Capture pre-inference resource baseline.
        rss_before: Optional[float] = None
        try:
            if _HAVE_PSUTIL and _psutil is not None:
                rss_before = _psutil.Process().memory_info().rss / 1e6
            import torch as _torch  # noqa: PLC0415
            _cuda_available = _torch.cuda.is_available()
            if _cuda_available:
                _torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            _cuda_available = False

        start_ts = time.perf_counter()
        result = _pipeline.run(
            checkpoint_path=_checkpoint_path,
            image_path=str(tmp_image_path),
            subject_id=subject_id,
            study_id=study_id,
            save_explainability=save_explainability,
        )
        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0

        # Build per-inference resource block — never let it break the response.
        inference_resources: Dict[str, Any] = {"device": _device_str, "processing_time_ms": round(elapsed_ms, 2)}
        try:
            if _cuda_available:
                inference_resources["peak_gpu_mem_mb"] = round(
                    _torch.cuda.max_memory_allocated() / 1e6, 1
                )
            if rss_before is not None and _HAVE_PSUTIL and _psutil is not None:
                rss_after = _psutil.Process().memory_info().rss / 1e6
                inference_resources["rss_delta_mb"] = round(rss_after - rss_before, 1)
        except Exception:  # noqa: BLE001
            pass

        xai = result.get("explainability")
        response_body: Dict[str, Any] = {
            "classification": result.get("classification", {}),
            "generated_report": result.get("generated_report", ""),
            "explainability": _xai_paths_to_urls(xai) if xai else None,
            "image_filename": original_filename,
            "processing_time_ms": round(elapsed_ms, 2),
            "thresholds": _thresholds,
            "thresholds_source": _thresholds_source,
            "resources": inference_resources,
        }
        return JSONResponse(content=response_body)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

"""Single-image inference and optional explainability export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

from src.core.config import ProjectConfig, load_config
from src.data.dataset import _get_report_tokenizer
from src.data.dicom_loader import DICOMLoader
from src.data.report_loader import ReportLoader
from src.data.transforms import get_eval_transforms
from src.evaluation.explainability import ExplainabilityExporter, ExplainabilitySample
from src.pipelines.training_pipeline import TrainingPipeline


class InferencePipeline:
    """Run a checkpoint on a single JPG/PNG image."""

    def __init__(
        self,
        config: ProjectConfig | str | Path | None = None,
        device: Optional[str] = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.training_pipeline = TrainingPipeline(load_config(Path(config)), device=device)
        else:
            self.training_pipeline = TrainingPipeline(config=config, device=device)

        self.config = self.training_pipeline.config
        self.device = self.training_pipeline.device

        # Cache the built + checkpoint-loaded model so repeated run() calls
        # (e.g. a long-lived REST server) do not rebuild BioMedCLIP/GATv2/GPT-2
        # and re-read the checkpoint from disk on every request. The one-shot
        # CLI path still builds exactly once.
        self._model: Any = None
        self._prepared_checkpoint: Optional[str] = None
        # Cache for report_graphs.pt (17 GB) — loaded at most once per process.
        self._report_graphs: Optional[Dict[str, Any]] = None
        self._report_graphs_loaded: bool = False

    def _prepare_model(self, checkpoint_path: str | Path) -> Any:
        """Build and checkpoint-load the model once, reusing it on later calls."""
        checkpoint_key = str(Path(checkpoint_path))
        if self._model is not None and self._prepared_checkpoint == checkpoint_key:
            return self._model

        self.training_pipeline.setup()
        model = self.training_pipeline.build_model()
        self.training_pipeline.load_checkpoint(checkpoint_path)
        model = model.to(self.device)
        model.eval()

        self._model = model
        self._prepared_checkpoint = checkpoint_key
        return model

    def run(
        self,
        checkpoint_path: str | Path,
        image_path: str | Path,
        *,
        subject_id: Optional[int] = None,
        study_id: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
        save_explainability: bool = False,
    ) -> Dict[str, Any]:
        model = self._prepare_model(checkpoint_path)

        pixel_values = self._load_image_tensor(image_path)
        graph_payload = self._load_graph_payload(subject_id, study_id)

        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                return_attention=save_explainability,
                **{k: v for k, v in graph_payload.items() if isinstance(v, torch.Tensor)},
            )

        class_names = self.config.model.classification_head.class_names
        classification_probs = torch.sigmoid(outputs.get("classification_logits", torch.zeros(1, len(class_names))))[0].detach().cpu()
        generated_report = ""
        generated_token_ids: list[int] = []
        generated_token_labels: list[str] = []
        if self.config.model.enable_report_generation:
            generated = model.generate_report(
                pixel_values=pixel_values,
                max_length=self.config.model.report_generation.max_report_length,
                **{k: v for k, v in graph_payload.items() if isinstance(v, torch.Tensor)},
            )
            if isinstance(generated, str):
                generated_report = generated
            elif isinstance(generated, list) and generated:
                first_item = generated[0]
                if isinstance(first_item, str):
                    generated_report = first_item
                elif isinstance(first_item, list):
                    generated_token_ids = first_item
                    generated_report = self.training_pipeline._decode_generated_tokens(generated_token_ids)
                    generated_token_labels = self._token_labels_from_ids(generated_token_ids)

        result = {
            "image_path": str(image_path),
            "checkpoint_path": str(checkpoint_path),
            "subject_id": subject_id,
            "study_id": study_id,
            "classification": {
                class_name: float(classification_probs[index])
                for index, class_name in enumerate(class_names)
            },
            "generated_report": generated_report,
        }

        if save_explainability:
            export_root = Path(output_dir) if output_dir is not None else self.config.training.log_dir.parent / "inference_explainability"
            exporter = ExplainabilityExporter(export_root)
            study_key = f"{subject_id}_{study_id}" if subject_id is not None and study_id is not None else Path(image_path).stem
            reference_report = ""
            if subject_id is not None and study_id is not None:
                reference_report = ReportLoader(
                    self.config.data.reports_root,
                    section_preference=self.config.data.report_section_preference,
                ).load_report(subject_id, study_id) or ""

            explainability_summary = exporter.export_sample(
                ExplainabilitySample(
                    study_key=study_key,
                    image=pixel_values[0].detach().cpu(),
                    explainability=outputs.get("explainability", {}),
                    class_names=list(class_names),
                    classification_probs=classification_probs,
                    reference_report=reference_report,
                    generated_report=generated_report,
                    generated_token_ids=generated_token_ids,
                    generated_token_labels=generated_token_labels,
                    graph_node_texts=graph_payload.get("graph_node_texts", []),
                    graph_node_cuis=graph_payload.get("graph_node_cuis", []),
                    graph_node_types=graph_payload.get("graph_node_types", []),
                    graph_node_certainties=graph_payload.get("graph_node_certainties", []),
                )
            )
            result["explainability"] = explainability_summary

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        return result

    def _token_labels_from_ids(self, token_ids: list[int]) -> list[str]:
        if not token_ids:
            return []
        try:
            tokenizer = _get_report_tokenizer()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            return [self._normalise_token_label(token) for token in tokens]
        except Exception:
            return [str(token_id) for token_id in token_ids]

    @staticmethod
    def _normalise_token_label(token: str) -> str:
        return token.replace("Ġ", " ").replace("Ċ", "\\n")

    def _load_image_tensor(self, image_path: str | Path) -> torch.Tensor:
        path = Path(image_path)
        allowed_suffixes = {suffix.lower() for suffix in self.config.data.image_suffixes}
        if path.suffix.lower() not in allowed_suffixes:
            raise ValueError(f"Expected a JPG/PNG image, got {path.suffix}")

        loader = DICOMLoader(image_size=self.config.model.visual_encoder.image_size)
        image = loader.load_image(path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        transform = get_eval_transforms(image_size=self.config.model.visual_encoder.image_size)
        return transform(image).unsqueeze(0).to(self.device)

    def _graph_data_to_payload(self, graph: Any) -> Dict[str, Any]:
        return {
            "graph_x": graph.x.to(self.device),
            "graph_edge_index": graph.edge_index.to(self.device),
            "graph_edge_type": graph.edge_type.to(self.device),
            "graph_batch": torch.zeros(graph.x.shape[0], dtype=torch.long, device=self.device),
            "graph_node_texts": list(getattr(graph, "node_texts", [])),
            "graph_node_cuis": list(getattr(graph, "node_cuis", [])),
            "graph_node_types": list(getattr(graph, "node_types", [])),
            "graph_node_certainties": list(getattr(graph, "node_certainties", [])),
        }

    def _load_global_graph_payload(self) -> Dict[str, Any]:
        if not self.config.model.use_kg:
            return {}
        global_kg_path = Path(self.config.data.kg_artifacts_dir) / "global_kg.pt"
        if not global_kg_path.exists():
            logger.warning("Global KG not found at %s; continuing without graph context.", global_kg_path)
            return {}
        try:
            graph = torch.load(global_kg_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load global KG (%s); continuing without graph context.", exc)
            return {}
        logger.info("Using global KG (%d nodes) for inference.", graph.x.shape[0])
        return self._graph_data_to_payload(graph)

    def _load_graph_payload(self, subject_id: Optional[int], study_id: Optional[int]) -> Dict[str, Any]:
        if subject_id is None or study_id is None:
            return self._load_global_graph_payload()

        report_graphs_path = Path(self.config.data.kg_artifacts_dir) / "report_graphs.pt"
        if not report_graphs_path.exists():
            return self._load_global_graph_payload()

        if not self._report_graphs_loaded:
            try:
                logger.info("Loading report_graphs.pt (first time, may take a while)...")
                self._report_graphs = torch.load(report_graphs_path, map_location="cpu", weights_only=False)
                logger.info("report_graphs.pt cached (%d studies).", len(self._report_graphs or {}))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load KG artifact %s (%s); falling back to global KG.",
                    report_graphs_path,
                    exc,
                )
            finally:
                self._report_graphs_loaded = True  # don't retry on failure

        if not self._report_graphs:
            return self._load_global_graph_payload()

        study_key = f"{subject_id}_{study_id}"
        graph = self._report_graphs.get(study_key)
        if graph is None:
            logger.info("Study %s not in report_graphs.pt; falling back to global KG.", study_key)
            return self._load_global_graph_payload()

        return self._graph_data_to_payload(graph)
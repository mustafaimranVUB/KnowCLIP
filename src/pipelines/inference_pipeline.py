"""Single-image inference and optional explainability export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

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
        self.training_pipeline.setup()
        model = self.training_pipeline.build_model()
        self.training_pipeline.load_checkpoint(checkpoint_path)
        model = model.to(self.device)
        model.eval()

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

    def _load_graph_payload(self, subject_id: Optional[int], study_id: Optional[int]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if subject_id is None or study_id is None:
            return payload

        report_graphs_path = Path(self.config.data.kg_artifacts_dir) / "report_graphs.pt"
        if not report_graphs_path.exists():
            return payload

        graphs = torch.load(report_graphs_path, map_location="cpu", weights_only=False)
        study_key = f"{subject_id}_{study_id}"
        graph = graphs.get(study_key)
        if graph is None:
            return payload

        payload.update({
            "graph_x": graph.x.to(self.device),
            "graph_edge_index": graph.edge_index.to(self.device),
            "graph_edge_type": graph.edge_type.to(self.device),
            "graph_batch": torch.zeros(graph.x.shape[0], dtype=torch.long, device=self.device),
            "graph_node_texts": list(getattr(graph, "node_texts", [])),
            "graph_node_cuis": list(getattr(graph, "node_cuis", [])),
            "graph_node_types": list(getattr(graph, "node_types", [])),
            "graph_node_certainties": list(getattr(graph, "node_certainties", [])),
        })
        return payload
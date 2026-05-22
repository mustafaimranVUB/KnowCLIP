"""Pipeline for exporting sample-level explainability artefacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.core.config import ProjectConfig, load_config
from src.data.dataset import _get_report_tokenizer, collate_mimic
from src.evaluation.explainability import ExplainabilityExporter, ExplainabilitySample
from src.pipelines.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


class ExplainabilityPipeline:
    """Run a trained model on individual studies and export self-explainability plots."""

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
        *,
        split: str = "test",
        max_samples: int = 8,
        output_dir: Path | str | None = None,
    ) -> Dict[str, Any]:
        self.training_pipeline.setup()
        loader = self._build_single_item_loader(split)
        model = self.training_pipeline.build_model()
        self.training_pipeline.load_checkpoint(checkpoint_path)
        model = model.to(self.device)
        model.eval()

        export_root = Path(output_dir) if output_dir is not None else self.config.training.log_dir.parent / "explainability"
        exporter = ExplainabilityExporter(export_root)

        summaries: List[Dict[str, Any]] = []
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= max_samples:
                    break
                summaries.append(self._export_batch(batch, exporter))

        manifest = {
            "checkpoint_path": str(checkpoint_path),
            "split": split,
            "max_samples": max_samples,
            "samples_exported": len(summaries),
            "output_dir": str(export_root),
            "samples": summaries,
        }

        export_root.mkdir(parents=True, exist_ok=True)
        manifest_path = export_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        markdown_path = export_root / "manifest.md"
        markdown_lines = [
            "# Explainability Export Manifest",
            "",
            f"- checkpoint: {checkpoint_path}",
            f"- split: {split}",
            f"- samples_exported: {len(summaries)}",
            "",
            "## Samples",
        ]
        for sample in summaries:
            markdown_lines.append(f"- {sample['study_key']}: {sample['sample_dir']}")
        markdown_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")

        logger.info("Explainability artefacts exported to %s", export_root)
        return manifest

    def _build_single_item_loader(self, split: str) -> DataLoader:
        train_loader, val_loader, test_loader = self.training_pipeline.build_dataloaders()
        loaders = {
            "train": train_loader,
            "validate": val_loader,
            "test": test_loader,
        }
        if split not in loaders:
            raise ValueError(f"Unsupported split '{split}'. Expected one of {sorted(loaders)}")

        dataset = loaders[split].dataset
        data_config = self.config.data
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=data_config.pin_memory and torch.cuda.is_available(),
            collate_fn=collate_mimic,
        )

    def _export_batch(self, batch: Dict[str, Any], exporter: ExplainabilityExporter) -> Dict[str, Any]:
        model = self.training_pipeline._model
        assert model is not None

        model_inputs = self._prepare_model_inputs(batch)
        outputs = model(return_attention=True, **model_inputs)

        generated_token_ids: List[int] = []
        generated_token_labels: List[str] = []
        generated_report = ""
        decoder_trace = self._collect_decoder_trace(model, model_inputs)
        if decoder_trace is not None:
            generated_token_ids = decoder_trace["token_ids"]
            generated_token_labels = decoder_trace["token_labels"]
            generated_report = decoder_trace["generated_report"]
            outputs.setdefault("explainability", {})["decoder"] = decoder_trace["trace"]

        logits = outputs.get("classification_logits")
        probs = torch.sigmoid(logits[0]).detach().cpu() if isinstance(logits, torch.Tensor) else None

        sample = ExplainabilitySample(
            study_key=str(batch["study_key"][0]),
            image=batch["image"][0].detach().cpu(),
            explainability=outputs.get("explainability", {}),
            class_names=list(self.config.model.classification_head.class_names),
            classification_probs=probs,
            reference_report=batch.get("report_text", [""])[0],
            generated_report=generated_report,
            generated_token_ids=generated_token_ids,
            generated_token_labels=generated_token_labels,
            graph_node_texts=list(batch.get("graph_node_texts", [[]])[0]),
            graph_node_cuis=list(batch.get("graph_node_cuis", [[]])[0]),
            graph_node_types=list(batch.get("graph_node_types", [[]])[0]),
            graph_node_certainties=list(batch.get("graph_node_certainties", [[]])[0]),
        )
        return exporter.export_sample(sample)

    def _prepare_model_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {
            "pixel_values": batch["image"].to(self.device),
        }
        for key in ("graph_x", "graph_edge_index", "graph_edge_type", "graph_batch"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)
            elif value is not None:
                inputs[key] = value
        return inputs

    def _collect_decoder_trace(
        self,
        model: torch.nn.Module,
        model_inputs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.config.model.enable_report_generation:
            return None
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            return None

        generated = model.generate_report(
            max_length=self.config.model.report_generation.max_report_length,
            **model_inputs,
        )
        if not generated:
            return None

        token_ids = generated[0] if isinstance(generated[0], list) else []
        if not token_ids:
            return None

        target_ids, token_labels = self._build_decoder_targets(token_ids)
        if target_ids is None:
            return None

        decoder_outputs = model(
            return_attention=True,
            target_ids=target_ids,
            **model_inputs,
        )
        explainability = decoder_outputs.get("explainability", {})
        trace = explainability.get("decoder")
        if trace is None:
            return None

        return {
            "token_ids": token_ids,
            "token_labels": token_labels,
            "generated_report": self.training_pipeline._decode_generated_tokens(token_ids),
            "trace": trace,
        }

    def _build_decoder_targets(self, token_ids: List[int]) -> tuple[Optional[torch.Tensor], List[str]]:
        if not token_ids:
            return None, []

        tokenizer = self._get_tokenizer()
        bos_id = getattr(tokenizer, "eos_token_id", 50256) if tokenizer is not None else 50256
        prefix = str(getattr(self.config.model.report_generation, "prompt_prefix", "") or "").strip()
        prefix_ids: List[int] = []
        prefix_labels: List[str] = []
        if tokenizer is not None and prefix:
            try:
                prefix_ids = [int(t) for t in tokenizer.encode(prefix, add_special_tokens=False)]
                prefix_labels = [self._normalise_token_label(token) for token in tokenizer.convert_ids_to_tokens(prefix_ids)]
            except Exception:
                prefix_ids = []
                prefix_labels = []

        usable_len = max(0, self.config.model.report_generation.max_report_length - 1)
        tail_len = max(0, usable_len - len(prefix_ids))
        target_sequence = [bos_id] + prefix_ids[:usable_len] + token_ids[:tail_len]
        target_tensor = torch.tensor([target_sequence], dtype=torch.long, device=self.device)

        if tokenizer is not None:
            token_labels = ["<BOS>"]
            token_labels.extend(prefix_labels[:usable_len])
            token_labels.extend(self._normalise_token_label(token) for token in tokenizer.convert_ids_to_tokens(token_ids))
            token_labels = token_labels[: len(target_sequence)]
        else:
            token_labels = ["<BOS>"] + [str(token_id) for token_id in prefix_ids[:usable_len]]
            token_labels.extend(str(token_id) for token_id in token_ids[: max(0, len(target_sequence) - len(token_labels))])

        return target_tensor, token_labels

    def _get_tokenizer(self) -> Any:
        if self.training_pipeline._report_tokenizer is None:
            try:
                self.training_pipeline._report_tokenizer = _get_report_tokenizer()
            except Exception:
                self.training_pipeline._report_tokenizer = False
        tokenizer = self.training_pipeline._report_tokenizer
        return None if tokenizer is False else tokenizer

    @staticmethod
    def _normalise_token_label(token: str) -> str:
        return token.replace("Ġ", " ").replace("Ċ", "\\n")
"""Centralised project configuration with dataclasses and YAML support."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Visual Encoder
# ---------------------------------------------------------------------------

@dataclass
class VisualEncoderConfig:
    """Configuration for the Visual Encoder (E_V) component."""

    backbone_type: Literal[
        "biomedclip", "pubmedclip", "vit-b-16", "vit-l-14"
    ] = "biomedclip"

    CHECKPOINTS: dict = field(default_factory=lambda: {
        "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "pubmedclip": "flaviagiammarino/pubmed-clip-vit-base-patch32",
        "vit-b-16": "openai/clip-vit-base-patch16",
        "vit-l-14": "openai/clip-vit-large-patch14",
    }, repr=False)

    PATCH_COUNTS: dict = field(default_factory=lambda: {
        "biomedclip": 196,   # (224/16)^2
        "pubmedclip": 49,    # (224/32)^2
        "vit-b-16": 196,     # (224/16)^2
        "vit-l-14": 256,     # (224/14)^2
    }, repr=False)

    HIDDEN_DIMS: dict = field(default_factory=lambda: {
        "biomedclip": 768,
        "pubmedclip": 768,
        "vit-b-16": 768,
        "vit-l-14": 1024,
    }, repr=False)

    image_size: int = 224
    output_dim: int = 768  # Unified output dimension after projection
    freeze_backbone: bool = False
    freeze_layers: Optional[List[int]] = None

    @property
    def checkpoint(self) -> str:
        return self.CHECKPOINTS[self.backbone_type]

    @property
    def num_patches(self) -> int:
        return self.PATCH_COUNTS[self.backbone_type]

    @property
    def hidden_dim(self) -> int:
        return self.HIDDEN_DIMS[self.backbone_type]


# ---------------------------------------------------------------------------
# Knowledge Encoder
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeEncoderConfig:
    """Configuration for the Knowledge Encoder (E_K) — GATv2."""

    num_gat_layers: int = 2
    hidden_channels: int = 256
    num_attention_heads: int = 4
    dropout: float = 0.1
    num_edge_types: int = 5  # 4 RadGraph + 1 self-loop
    edge_dim: int = 64
    input_dim: int = 768
    output_dim: int = 768
    normalize_outputs: bool = True


# ---------------------------------------------------------------------------
# Fusion Module
# ---------------------------------------------------------------------------

@dataclass
class FusionModuleConfig:
    """Configuration for Knowledge-Visual Fusion."""

    fusion_type: Literal[
        "cross_attention", "gated_fusion"
    ] = "cross_attention"
    num_fusion_layers: int = 2
    num_heads: int = 8
    hidden_dim: int = 768
    use_bottleneck: bool = True
    bottleneck_dim: int = 256
    qkv_bias: bool = True
    attention_dropout: float = 0.1
    projection_dropout: float = 0.1
    use_residual: bool = True


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

CHEXPERT_LABELS: List[str] = [
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


@dataclass
class ClassificationHeadConfig:
    """Configuration for multi-label pathology classification."""

    num_classes: int = 14
    class_names: List[str] = field(default_factory=lambda: list(CHEXPERT_LABELS))
    hidden_dim: int = 512
    dropout: float = 0.3
    use_batch_norm: bool = True


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

@dataclass
class ReportGenerationConfig:
    """Configuration for radiology report generation decoder."""

    decoder_type: Literal["transformer", "gpt2"] = "transformer"
    vocab_size: int = 50257  # GPT-2 tokenizer default
    max_report_length: int = 128
    num_decoder_layers: int = 6
    decoder_dim: int = 768
    num_decoder_heads: int = 8
    decoder_ffn_dim: int = 2048
    decoder_dropout: float = 0.1
    beam_size: int = 3
    length_penalty: float = 1.0
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    scheduled_sampling_enabled: bool = False
    scheduled_sampling_start_ratio: float = 1.0
    scheduled_sampling_end_ratio: float = 0.7
    scheduled_sampling_decay_epochs: int = 20


# ---------------------------------------------------------------------------
# Model Config (Orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Top-level model configuration."""

    use_kg: bool = True
    enable_classification: bool = True
    enable_report_generation: bool = True

    visual_encoder: VisualEncoderConfig = field(
        default_factory=VisualEncoderConfig
    )
    knowledge_encoder: KnowledgeEncoderConfig = field(
        default_factory=KnowledgeEncoderConfig
    )
    fusion_module: FusionModuleConfig = field(
        default_factory=FusionModuleConfig
    )
    classification_head: ClassificationHeadConfig = field(
        default_factory=ClassificationHeadConfig
    )
    report_generation: ReportGenerationConfig = field(
        default_factory=ReportGenerationConfig
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise config for logging / checkpoints."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Training / Data / Project Configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Paths and loading settings for MIMIC-CXR data."""

    mimic_root: Path = field(default_factory=lambda: Path(
        os.environ.get("MIMIC_ROOT", "data/mimic-cxr")
    ))
    reports_root: Path = field(default_factory=lambda: Path(
        os.environ.get("MIMIC_REPORTS", "data/mimic-cxr-reports/files")
    ))
    split_csv: Path = field(default_factory=lambda: Path(
        os.environ.get("MIMIC_SPLIT_CSV", "data/mimic-cxr-2.0.0-split.csv")
    ))
    chexpert_csv: Path = field(default_factory=lambda: Path(
        os.environ.get("MIMIC_CHEXPERT_CSV", "data/mimic-cxr-2.0.0-chexpert.csv")
    ))
    kg_artifacts_dir: Path = Path("outputs/KG")

    # Split handling:
    # - official: use PhysioNet-provided split column.
    # - subset_hash: deterministic hash split over available studies.
    # - auto: use official unless val/test available counts are too small.
    split_strategy: Literal["official", "subset_hash", "auto"] = "auto"
    subset_seed: int = 42
    subset_train_ratio: float = 0.8
    subset_val_ratio: float = 0.1
    subset_test_ratio: float = 0.1
    auto_min_val_samples: int = 100
    auto_min_test_samples: int = 100

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    # Subset for development — fraction of official split
    dev_subset_frac: Optional[float] = None  # e.g. 0.05 for 5%


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    num_epochs: int = 30
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = True
    early_stopping_patience: int = 5

    classification_loss_weight: float = 1.0
    generation_loss_weight: float = 1.0

    # Classification loss settings for class imbalance.
    classification_loss_type: Literal["bce", "focal"] = "bce"
    use_pos_weight: bool = True
    max_pos_weight: float = 30.0
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None

    # Two-stage training: stage-1 trains KG/fusion/generation first, then joint fine-tuning.
    two_stage_training: bool = False
    stage1_epochs: int = 0
    stage1_classification_weight: float = 0.0
    stage1_freeze_visual_encoder: bool = True
    stage1_freeze_classification_head: bool = True
    stage1_enable_early_stopping: bool = False
    stage1_early_stopping_patience: Optional[int] = None
    reset_best_metric_on_stage2: bool = True

    checkpoint_selection_mode: Literal["val_loss", "classification_priority"] = "val_loss"
    checkpoint_generation_guard_max_val_gen_loss: Optional[float] = None

    # Generation-eval debug logging.
    generation_debug_samples: int = 3

    seed: int = 42

    checkpoint_dir: Path = Path("outputs/checkpoints")
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3

    log_dir: Path = Path("outputs/logs")
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1


@dataclass
class KGPipelineConfig:
    """Configuration for Phase I KG construction pipeline."""

    mrconso_path: Path = Path(
        "data/umls-2025AB-metathesaurus-full/2025AB/META/MRCONSO.RRF"
    )
    mrsty_path: Path = Path(
        "data/umls-2025AB-metathesaurus-full/2025AB/META/MRSTY.RRF"
    )
    mrrel_path: Path = Path(
        "data/umls-2025AB-metathesaurus-full/2025AB/META/MRREL.RRF"
    )
    radgraph_model_type: str = "modern-radgraph-xl"
    output_dir: Path = Path("outputs/KG")
    sab_preference: List[str] = field(
        default_factory=lambda: ["SNOMEDCT_US", "RXNORM", "MSH"]
    )
    top_k_candidates: int = 5

    # ----------------------------------------------------------------
    # scispaCy Stage-2 fallback grounding
    # ----------------------------------------------------------------
    #: Whether to use scispaCy EntityLinker as a fallback when MRCONSO
    #: exact matching finds no candidate for a given mention.
    use_scispacy_fallback: bool = True

    #: Directory where en_core_sci_lg was installed via
    #: ``pip install --target``.  Reads from the ``SCISPACY_CACHE``
    #: environment variable if this field is empty / not set.
    #: On Hydra HPC set to ``$VSC_SCRATCH/scispacy_cache``.
    scispacy_cache_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("SCISPACY_CACHE", "")
    ))

    #: Minimum confidence score (0–1) for scispaCy EntityLinker hits.
    scispacy_confidence_threshold: float = 0.85

    #: Optional minimum grounding coverage threshold used by Phase II preflight checks.
    min_grounding_coverage: float = 0.0

    #: If True and grounding coverage is below threshold, abort Phase II run.
    fail_on_low_grounding_coverage: bool = False


@dataclass
class ProjectConfig:
    """Root configuration aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    kg_pipeline: KGPipelineConfig = field(default_factory=KGPipelineConfig)

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_baseline_config() -> ModelConfig:
    """Pure-vision baseline: classification only, no KG."""
    return ModelConfig(
        use_kg=False,
        enable_classification=True,
        enable_report_generation=False,
    )


def get_neurosymbolic_config() -> ModelConfig:
    """Full neuro-symbolic model with KG and report generation."""
    return ModelConfig(
        use_kg=True,
        enable_classification=True,
        enable_report_generation=True,
    )


def load_config(yaml_path: Optional[Path] = None) -> ProjectConfig:
    """Load a ProjectConfig from a YAML file, or return defaults.

    Args:
        yaml_path: Path to YAML config file. If None, returns defaults.

    Returns:
        ProjectConfig instance.
    """
    if yaml_path is None:
        return ProjectConfig()

    import yaml  # type: ignore

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    config = ProjectConfig()

    # Overlay YAML values onto dataclass defaults
    def _overlay(dc_obj: Any, d: dict) -> None:
        import dataclasses
        for key, val in d.items():
            if hasattr(dc_obj, key):
                current = getattr(dc_obj, key)
                if dataclasses.is_dataclass(current) and isinstance(val, dict):
                    _overlay(current, val)
                elif isinstance(current, Path) and isinstance(val, str):
                    setattr(dc_obj, key, Path(os.path.expandvars(val)))
                else:
                    setattr(dc_obj, key, val)

    _overlay(config, raw)
    return config

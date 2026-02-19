"""
Phase II: Neuro-Symbolic Architecture for Medical Imaging.

This package provides a modular, comparative framework for benchmarking:
- Neuro-Symbolic models (Vision + Knowledge Graph)
- Baseline models (Pure Vision)

Main components:
- config: Configuration classes for all hyperparameters
- modules: Core neural network components
- model_factory: MedicalVLM wrapper and builder functions
- dataset: MIMIC-CXR dataset with KG integration
"""

from .config import (
    ModelConfig,
    VisualEncoderConfig,
    KnowledgeEncoderConfig,
    FusionModuleConfig,
    ClassificationHeadConfig,
    ReportGenerationConfig,
    DataConfig,
    TrainingConfig,
    get_baseline_config,
    get_neurosymbolic_config,
    get_config_for_backbone,
)

from .modules import (
    VisualEncoder,
    KnowledgeEncoder,
    FusionModule,
    MultiHeadCrossAttention,
    SelfAttentionPooling,
)

from .model_factory import (
    MedicalVLM,
    ClassificationHead,
    ReportGenerator,
    build_model,
    build_baseline_model,
    build_neurosymbolic_model,
)

from .dataset import (
    MIMIC_CXR_Dataset,
    collate_kg_batch,
    load_mimic_split,
    load_chexpert_labels,
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "ModelConfig",
    "VisualEncoderConfig",
    "KnowledgeEncoderConfig",
    "FusionModuleConfig",
    "ClassificationHeadConfig",
    "ReportGenerationConfig",
    "DataConfig",
    "TrainingConfig",
    "get_baseline_config",
    "get_neurosymbolic_config",
    "get_config_for_backbone",
    # Modules
    "VisualEncoder",
    "KnowledgeEncoder",
    "FusionModule",
    "MultiHeadCrossAttention",
    "SelfAttentionPooling",
    # Model Factory
    "MedicalVLM",
    "ClassificationHead",
    "ReportGenerator",
    "build_model",
    "build_baseline_model",
    "build_neurosymbolic_model",
    # Dataset
    "MIMIC_CXR_Dataset",
    "collate_kg_batch",
    "load_mimic_split",
    "load_chexpert_labels",
]

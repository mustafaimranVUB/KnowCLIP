"""
Configuration module for Phase II: Neuro-Symbolic Architecture.

This module defines all hyperparameters and architectural choices for the
comparative framework, enabling easy switching between:
- Neuro-Symbolic (with KG) vs. Pure Vision (without KG)
- Different visual backbones (BioMedCLIP, PubMedCLIP, ViT)
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List
from pathlib import Path


@dataclass
class VisualEncoderConfig:
    """Configuration for the Visual Encoder (E_V) component."""

    # Backbone selection
    backbone_type: Literal["biomedclip", "pubmedclip", "vit-b-16", "vit-l-14"] = (
        "biomedclip"
    )

    # Model identifiers for HuggingFace/local loading
    model_checkpoints = {
        "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "pubmedclip": "flaviagiammarino/pubmed-clip-vit-base-patch32",
        "vit-b-16": "openai/clip-vit-base-patch16",
        "vit-l-14": "openai/clip-vit-large-patch14",
    }

    # Image preprocessing
    image_size: int = 224
    patch_size: int = 16

    # Output dimensions
    hidden_dim: int = 768  # ViT-B/16 default (512 for ViT-L/14)
    num_patches: int = 196  # (224/16)^2 = 14x14 patches

    # Freezing strategy
    freeze_backbone: bool = False
    freeze_layers: Optional[List[int]] = None  # Which transformer layers to freeze


@dataclass
class KnowledgeEncoderConfig:
    """Configuration for the Knowledge Encoder (E_K) - Graph Attention Network."""

    # GAT architecture
    num_gat_layers: int = 2
    hidden_channels: int = 256
    num_attention_heads: int = 4
    dropout: float = 0.1

    # Edge type handling
    num_edge_types: int = 4  # LOCATED_AT, MODIFY, SUGGESTIVE_OF, ASSOCIATED_WITH
    edge_dim: int = 64  # Edge feature dimension

    # Node feature initialization
    use_pretrained_embeddings: bool = True  # Use UMLS concept embeddings
    concept_embedding_dim: int = 768  # Match CLIP dimension for fusion

    # Global vs. Report-specific KG
    use_global_kg: bool = True
    use_report_specific_kg: bool = True  # Hybrid mode: both global + report-specific

    # Output normalization
    normalize_outputs: bool = True


@dataclass
class FusionModuleConfig:
    """Configuration for Knowledge-Visual Fusion via Cross-Attention."""

    # Cross-attention mechanism
    num_fusion_layers: int = 2
    num_heads: int = 8
    hidden_dim: int = 768

    # Attention bottleneck
    use_bottleneck: bool = True
    bottleneck_dim: int = 256

    # Query-Key-Value projections
    qkv_bias: bool = True
    attention_dropout: float = 0.1
    projection_dropout: float = 0.1

    # Fusion strategy
    fusion_type: Literal["cross_attention", "co_attention", "gated_fusion"] = (
        "cross_attention"
    )

    # Residual connections
    use_residual: bool = True


@dataclass
class ClassificationHeadConfig:
    """Configuration for multi-label pathology classification."""

    # Output classes (CheXpert/MIMIC-CXR-14 standard labels)
    num_classes: int = 14
    class_names: List[str] = field(
        default_factory=lambda: [
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
    )

    # Architecture
    hidden_dim: int = 512
    dropout: float = 0.3
    use_batch_norm: bool = True


@dataclass
class ReportGenerationConfig:
    """Configuration for radiology report generation."""

    # Decoder architecture
    decoder_type: Literal["transformer", "lstm", "gpt2"] = "transformer"
    vocab_size: int = 10000  # To be set from tokenizer
    max_report_length: int = 128

    # Transformer decoder
    num_decoder_layers: int = 6
    decoder_dim: int = 768
    num_decoder_heads: int = 8
    decoder_ffn_dim: int = 2048
    decoder_dropout: float = 0.1

    # Generation parameters
    beam_size: int = 3
    length_penalty: float = 1.0

    # Knowledge-grounded generation
    use_concept_guided_decoding: bool = True


@dataclass
class ModelConfig:
    """Main model configuration - the orchestrator for all components."""

    # Architecture selection
    use_kg: bool = True  # Toggle Neuro-Symbolic vs. Pure Vision

    # Task configuration
    enable_classification: bool = True
    enable_report_generation: bool = True

    # Component configs
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    knowledge_encoder: KnowledgeEncoderConfig = field(
        default_factory=KnowledgeEncoderConfig
    )
    fusion_module: FusionModuleConfig = field(default_factory=FusionModuleConfig)
    classification_head: ClassificationHeadConfig = field(
        default_factory=ClassificationHeadConfig
    )
    report_generation: ReportGenerationConfig = field(
        default_factory=ReportGenerationConfig
    )

    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    # Loss weights for multi-task learning
    classification_loss_weight: float = 1.0
    generation_loss_weight: float = 1.0

    # Device
    device: str = "cuda"

    def to_dict(self):
        """Convert config to dictionary for logging."""
        return {
            "use_kg": self.use_kg,
            "backbone_type": self.visual_encoder.backbone_type,
            "enable_classification": self.enable_classification,
            "enable_report_generation": self.enable_report_generation,
            "num_gat_layers": self.knowledge_encoder.num_gat_layers,
            "num_fusion_layers": self.fusion_module.num_fusion_layers,
        }


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""

    # Dataset paths
    mimic_cxr_jpg_root: Path = Path("data/mimic-cxr-jpg")
    mimic_cxr_rrg_root: Path = Path("data/MIMIC-CXR-RRG_small")
    kg_artifacts_dir: Path = Path("outputs/KG")

    # Split configuration
    split: Literal["train", "validate", "test"] = "train"
    use_official_splits: bool = True  # Prevent data leakage

    # Data loading
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: float = 0.5

    # Knowledge graph loading
    load_global_kg: bool = True
    load_report_specific_kg: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    num_epochs: int = 50
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1

    # Checkpointing
    checkpoint_dir: Path = Path("outputs/checkpoints")
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3

    # Logging
    log_dir: Path = Path("outputs/logs")
    log_every_n_steps: int = 100

    # Evaluation
    eval_every_n_epochs: int = 1

    # Random seed
    seed: int = 42


def get_baseline_config() -> ModelConfig:
    """
    Get configuration for baseline (Pure Vision) model.

    Returns:
        ModelConfig with use_kg=False and linear probe classifier.
    """
    config = ModelConfig(
        use_kg=False,
        enable_classification=True,
        enable_report_generation=False,  # Baseline: classification only
    )
    config.visual_encoder.freeze_backbone = False
    return config


def get_neurosymbolic_config() -> ModelConfig:
    """
    Get configuration for full Neuro-Symbolic model.

    Returns:
        ModelConfig with use_kg=True and all components enabled.
    """
    config = ModelConfig(
        use_kg=True, enable_classification=True, enable_report_generation=True
    )
    return config


def get_config_for_backbone(backbone: str, use_kg: bool = True) -> ModelConfig:
    """
    Helper to create config for a specific visual backbone.

    Args:
        backbone: One of 'biomedclip', 'pubmedclip', 'vit-b-16', 'vit-l-14'
        use_kg: Whether to use knowledge graph

    Returns:
        ModelConfig with specified backbone.
    """
    config = ModelConfig(use_kg=use_kg)
    config.visual_encoder.backbone_type = backbone

    # Adjust dimensions for ViT-L
    if backbone == "vit-l-14":
        config.visual_encoder.hidden_dim = 1024
        config.visual_encoder.num_patches = 256  # (224/14)^2

    return config

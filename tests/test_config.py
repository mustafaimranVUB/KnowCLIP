"""Tests for configuration loading and validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.core.config import (
    DataConfig,
    KGPipelineConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    get_baseline_config,
    get_neurosymbolic_config,
    load_config,
)


class TestProjectConfig:
    def test_default_construction(self):
        config = ProjectConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.kg_pipeline, KGPipelineConfig)

    def test_to_dict(self):
        config = ProjectConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "model" in d
        assert "data" in d
        assert "training" in d


class TestModelConfig:
    def test_baseline_no_kg(self):
        config = get_baseline_config()
        assert config.use_kg is False
        assert config.enable_classification is True
        assert config.enable_report_generation is False

    def test_neurosymbolic_full(self):
        config = get_neurosymbolic_config()
        assert config.use_kg is True
        assert config.enable_classification is True
        assert config.enable_report_generation is True


class TestVisualEncoderConfig:
    def test_checkpoint_lookup(self):
        config = ProjectConfig()
        ve = config.model.visual_encoder
        assert "BiomedCLIP" in ve.checkpoint

    def test_patch_count(self):
        config = ProjectConfig()
        ve = config.model.visual_encoder
        assert ve.num_patches == 196

    def test_hidden_dim(self):
        config = ProjectConfig()
        ve = config.model.visual_encoder
        assert ve.hidden_dim == 768


class TestLoadConfig:
    def test_load_none_returns_defaults(self):
        config = load_config(None)
        assert isinstance(config, ProjectConfig)
        assert config.training.seed == 42

    def test_load_yaml_overlay(self):
        yaml_content = {
            "training": {
                "seed": 99,
                "learning_rate": 5e-5,
            },
            "model": {
                "use_kg": False,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            config = load_config(Path(f.name))

        assert config.training.seed == 99
        assert config.training.learning_rate == 5e-5
        assert config.model.use_kg is False
        # Default values not overridden
        assert config.training.num_epochs == 30

    def test_load_yaml_nested(self):
        yaml_content = {
            "model": {
                "visual_encoder": {
                    "backbone_type": "pubmedclip",
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            config = load_config(Path(f.name))

        assert config.model.visual_encoder.backbone_type == "pubmedclip"
        assert config.model.visual_encoder.checkpoint == "flaviagiammarino/pubmed-clip-vit-base-patch32"


class TestDataConfig:
    def test_default_paths(self):
        config = DataConfig()
        assert config.batch_size == 16
        assert config.num_workers == 4
        assert config.subset_test_ratio == 0.1


class TestTrainingConfig:
    def test_two_stage_defaults(self):
        config = TrainingConfig()
        assert config.two_stage_training is False
        assert config.stage1_epochs == 0
        assert config.stage1_freeze_visual_encoder is True
        assert config.stage1_freeze_classification_head is True
        assert config.generation_debug_samples == 3


class TestKGPipelineConfig:
    def test_default_model_type(self):
        config = KGPipelineConfig()
        assert config.radgraph_model_type == "modern-radgraph-xl"
        assert config.top_k_candidates == 5

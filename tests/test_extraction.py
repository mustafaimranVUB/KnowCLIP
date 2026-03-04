"""Tests for src.knowledge.extraction module."""

from __future__ import annotations

import pytest

from src.knowledge.extraction import (
    Certainty,
    EntityExtractor,
    EntityType,
    ExtractedEntity,
    ExtractedTriple,
    RelationType,
    canonicalize_surface,
    is_measurement,
    normalize_text,
    parse_entity_label,
)


class TestNormalizeText:
    def test_basic(self):
        assert normalize_text("  Heart  SIZE ") == "heart size"

    def test_multispace(self):
        assert normalize_text("lung   opacity") == "lung opacity"

    def test_empty(self):
        assert normalize_text("") == ""


class TestCanonicalizeSurface:
    def test_hyphens_removed(self):
        result = canonicalize_surface("right-sided")
        assert "-" not in result or result == "right sided"

    def test_idempotent(self):
        text = "cardiomegaly"
        assert canonicalize_surface(canonicalize_surface(text)) == canonicalize_surface(text)


class TestIsMeasurement:
    def test_mm(self):
        assert is_measurement("5 mm")
        assert is_measurement("3.2 cm")

    def test_not_measurement(self):
        assert not is_measurement("heart")
        assert not is_measurement("opacity")


class TestParseEntityLabel:
    def test_anatomy_present(self):
        etype, cert = parse_entity_label("Anatomy::definitely present")
        assert etype == EntityType.ANATOMY
        assert cert == Certainty.DEFINITELY_PRESENT

    def test_observation_absent(self):
        etype, cert = parse_entity_label("Observation::definitely absent")
        assert etype == EntityType.OBSERVATION
        assert cert == Certainty.DEFINITELY_ABSENT

    def test_measurement(self):
        etype, cert = parse_entity_label("Observation::Measurement::definitely present")
        assert etype == EntityType.MEASUREMENT
        assert cert == Certainty.DEFINITELY_PRESENT

    def test_uncertain(self):
        etype, cert = parse_entity_label("Observation::uncertain")
        assert etype == EntityType.OBSERVATION
        assert cert == Certainty.UNCERTAIN


class TestFilterEligible:
    def test_excludes_measurements(self):
        entities = [
            ExtractedEntity(text="heart", entity_type=EntityType.ANATOMY, certainty=Certainty.DEFINITELY_PRESENT),
            ExtractedEntity(text="5 mm", entity_type=EntityType.MEASUREMENT, certainty=Certainty.DEFINITELY_PRESENT),
            ExtractedEntity(text="opacity", entity_type=EntityType.OBSERVATION, certainty=Certainty.UNCERTAIN),
        ]
        eligible = EntityExtractor.filter_eligible(entities)
        assert len(eligible) == 2
        assert all(e.entity_type != EntityType.MEASUREMENT for e in eligible)

    def test_excludes_measurement_text(self):
        entities = [
            ExtractedEntity(text="3.2 cm nodule", entity_type=EntityType.OBSERVATION, certainty=Certainty.DEFINITELY_PRESENT),
        ]
        eligible = EntityExtractor.filter_eligible(entities)
        assert len(eligible) == 0


class TestGroupByType:
    def test_groups(self):
        entities = [
            ExtractedEntity(text="heart", entity_type=EntityType.ANATOMY, certainty=Certainty.DEFINITELY_PRESENT),
            ExtractedEntity(text="lungs", entity_type=EntityType.ANATOMY, certainty=Certainty.DEFINITELY_PRESENT),
            ExtractedEntity(text="opacity", entity_type=EntityType.OBSERVATION, certainty=Certainty.UNCERTAIN),
        ]
        groups = EntityExtractor.group_by_type(entities)
        assert len(groups[EntityType.ANATOMY]) == 2
        assert len(groups[EntityType.OBSERVATION]) == 1


class TestExtractedEntity:
    def test_normalized_text(self):
        e = ExtractedEntity(text="  Heart  ", entity_type=EntityType.ANATOMY, certainty=Certainty.DEFINITELY_PRESENT)
        assert e.normalized_text == "heart"


class TestRelationType:
    def test_values(self):
        assert RelationType.LOCATED_AT.value == "located_at"
        assert RelationType.MODIFY.value == "modify"
        assert RelationType.SUGGESTIVE_OF.value == "suggestive_of"
        assert RelationType.ASSOCIATED_WITH.value == "associated_with"

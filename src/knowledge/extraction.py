"""Entity and relation extraction from radiology reports using RadGraph-XL.

Wraps the ``radgraph`` package to extract medical entities (Anatomy,
Observation) and relations (located_at, modify, suggestive_of,
associated_with) from free-text radiology reports.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & data-classes
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    """Coarse entity type extracted by RadGraph."""
    ANATOMY = "anatomy"
    OBSERVATION = "observation"
    MEASUREMENT = "measurement"


class Certainty(str, Enum):
    """Certainty qualifier from RadGraph labels."""
    DEFINITELY_PRESENT = "definitely_present"
    DEFINITELY_ABSENT = "definitely_absent"
    UNCERTAIN = "uncertain"


class RelationType(str, Enum):
    """Relation types from RadGraph."""
    LOCATED_AT = "located_at"
    MODIFY = "modify"
    SUGGESTIVE_OF = "suggestive_of"
    ASSOCIATED_WITH = "associated_with"


RELATION_TO_INT: Dict[str, int] = {
    "located_at": 0,
    "modify": 1,
    "suggestive_of": 2,
    "associated_with": 3,
}


@dataclass
class ExtractedEntity:
    """A single entity extracted from a report."""
    text: str
    entity_type: EntityType
    certainty: Certainty
    start_ix: int = -1
    end_ix: int = -1
    raw_label: str = ""
    cui: Optional[str] = None               # filled by grounding
    cui_candidates: List[str] = field(default_factory=list)

    @property
    def normalized_text(self) -> str:
        return normalize_text(self.text)


@dataclass
class ExtractedTriple:
    """A directed relation between two entities."""
    head_text: str
    relation: RelationType
    tail_text: str
    head_label: str = ""
    tail_label: str = ""


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_MEASUREMENT_RE = re.compile(r"\b\d+(\.\d+)?\s*(mm|cm|m|ml|cc|mg|g|l)\b", re.IGNORECASE)
_HYPHENS_RE = re.compile(r"[-–—/]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 \-]")
_MULTISPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, keep hyphens."""
    t = text.lower().strip()
    t = _MULTISPACE_RE.sub(" ", t)
    return t


def canonicalize_surface(text: str) -> str:
    """Aggressive canonicalization: unify hyphens, strip punctuation."""
    t = normalize_text(text)
    t = _HYPHENS_RE.sub(" ", t)
    t = _NON_ALNUM_RE.sub("", t)
    t = _MULTISPACE_RE.sub(" ", t).strip()
    return t


def is_measurement(text: str) -> bool:
    """Check if an entity text is a measurement."""
    return bool(_MEASUREMENT_RE.search(text))


# ---------------------------------------------------------------------------
# RadGraph label parsing
# ---------------------------------------------------------------------------

def parse_entity_label(label: str) -> Tuple[EntityType, Certainty]:
    """Parse a RadGraph label string into (EntityType, Certainty).

    Examples::
        'Anatomy::definitely present'      → (ANATOMY, DEFINITELY_PRESENT)
        'Observation::definitely absent'   → (OBSERVATION, DEFINITELY_ABSENT)
        'Observation::Measurement::definitely present' → (MEASUREMENT, DEFINITELY_PRESENT)

    Args:
        label: Raw label from RadGraph output.

    Returns:
        Tuple of (EntityType, Certainty).
    """
    parts = [p.strip().lower() for p in label.split("::")]

    # Determine entity type
    if "measurement" in parts:
        etype = EntityType.MEASUREMENT
    elif parts[0] == "anatomy":
        etype = EntityType.ANATOMY
    elif parts[0] == "observation":
        etype = EntityType.OBSERVATION
    else:
        logger.warning("Unknown entity type in label '%s', defaulting to OBSERVATION", label)
        etype = EntityType.OBSERVATION

    # Determine certainty from last part
    cert_str = parts[-1].replace(" ", "_")
    try:
        certainty = Certainty(cert_str)
    except ValueError:
        logger.warning("Unknown certainty in label '%s', defaulting to DEFINITELY_PRESENT", label)
        certainty = Certainty.DEFINITELY_PRESENT

    return etype, certainty


def parse_relation_type(rel_str: str) -> RelationType:
    """Parse a relation string from RadGraph.

    Args:
        rel_str: Relation string (e.g. ``'located_at'``, ``'modify'``).

    Returns:
        RelationType enum value.
    """
    key = rel_str.lower().strip().replace("-", "_").replace(" ", "_")
    try:
        return RelationType(key)
    except ValueError:
        logger.warning("Unknown relation '%s', defaulting to ASSOCIATED_WITH", rel_str)
        return RelationType.ASSOCIATED_WITH


# ---------------------------------------------------------------------------
# RadGraph wrapper
# ---------------------------------------------------------------------------

class EntityExtractor:
    """Extract entities and relations from radiology reports using RadGraph-XL.

    Parameters:
        model_type: RadGraph model identifier (default ``'modern-radgraph-xl'``).
        device: Torch device string/object.
    """

    def __init__(
        self,
        model_type: str = "modern-radgraph-xl",
        device: Optional[str] = None,
    ) -> None:
        self.model_type = model_type
        self.device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Lazy-load the RadGraph model."""
        if self._model is None:
            logger.info("Loading RadGraph model: %s", self.model_type)
            try:
                from radgraph import RadGraph  # type: ignore

                kwargs: Dict[str, Any] = {"model_type": self.model_type}
                if self.device:
                    kwargs["device"] = self.device
                self._model = RadGraph(**kwargs)
            except ImportError:
                raise ImportError(
                    "radgraph package not installed.  "
                    "Install with: pip install radgraph"
                )
        return self._model

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        reports: Sequence[str],
    ) -> List[Tuple[List[ExtractedEntity], List[ExtractedTriple]]]:
        """Extract entities and triples from a batch of reports.

        Args:
            reports: Sequence of cleaned report strings.

        Returns:
            List of ``(entities, triples)`` tuples, one per report.
        """
        cleaned = [self._clean_report(r) for r in reports]
        raw_preds = self.model(cleaned)

        results: List[Tuple[List[ExtractedEntity], List[ExtractedTriple]]] = []
        for idx in range(len(cleaned)):
            key = str(idx)
            pred = raw_preds.get(key, raw_preds.get(idx, {}))
            entities, triples = self._normalize_output(pred)
            results.append((entities, triples))

        return results

    def extract_single(
        self,
        report: str,
    ) -> Tuple[List[ExtractedEntity], List[ExtractedTriple]]:
        """Extract entities and triples from a single report."""
        results = self.extract_batch([report])
        return results[0]

    # ------------------------------------------------------------------
    # Output normalisation
    # ------------------------------------------------------------------

    #: Flag so we only log the raw output structure once (on first call)
    _logged_raw_structure: bool = False

    def _normalize_output(
        self,
        prediction: Dict[str, Any],
    ) -> Tuple[List[ExtractedEntity], List[ExtractedTriple]]:
        """Convert raw RadGraph output to structured entities & triples.

        RadGraph returns a dict with ``'entities'`` and ``'relations'``
        (or similar schema).  The exact structure depends on the
        radgraph version — we handle the common formats.
        """
        # --- One-time diagnostic log to reveal the raw output schema ---
        if not EntityExtractor._logged_raw_structure:
            EntityExtractor._logged_raw_structure = True
            top_keys = list(prediction.keys()) if isinstance(prediction, dict) else type(prediction).__name__
            raw_ents = prediction.get("entities", {}) if isinstance(prediction, dict) else {}
            sample_ent_id, sample_ent_data = next(iter(raw_ents.items())) if raw_ents else (None, {})
            sample_rels = sample_ent_data.get("relations", []) if sample_ent_data else []
            logger.info(
                "[RadGraph diagnostic] top-level keys=%s | "
                "sample entity keys=%s | "
                "sample entity relations (first 3)=%s",
                top_keys,
                list(sample_ent_data.keys()) if sample_ent_data else [],
                sample_rels[:3],
            )

        entities_dict: Dict[str, ExtractedEntity] = {}
        triples: List[ExtractedTriple] = []

        # Parse entities
        raw_entities = prediction.get("entities", {})
        for ent_id, ent_data in raw_entities.items():
            tokens = ent_data.get("tokens", "")
            label = ent_data.get("label", "")
            start_ix = ent_data.get("start_ix", -1)
            end_ix = ent_data.get("end_ix", -1)

            etype, certainty = parse_entity_label(label)
            entity = ExtractedEntity(
                text=tokens,
                entity_type=etype,
                certainty=certainty,
                start_ix=start_ix,
                end_ix=end_ix,
                raw_label=label,
            )
            entities_dict[str(ent_id)] = entity

        # Parse relations.
        # RadGraph (radgraph package) primarily embeds relations *inside* each
        # entity under the key "relations": [[rel_type, target_id], ...]
        # We always scan entity-level relations first, then fall back to any
        # top-level "relations" list/dict for other output schemas.

        seen_triples: set = set()  # deduplicate (head_id, tail_id, rel)

        def _add_triple(head_id: str, tail_id: str, rel_type: str) -> None:
            key = (head_id, tail_id, rel_type)
            if key in seen_triples:
                return
            seen_triples.add(key)
            if head_id in entities_dict and tail_id in entities_dict:
                head_ent = entities_dict[head_id]
                tail_ent = entities_dict[tail_id]
                triples.append(
                    ExtractedTriple(
                        head_text=head_ent.text,
                        relation=parse_relation_type(rel_type),
                        tail_text=tail_ent.text,
                        head_label=head_ent.raw_label,
                        tail_label=tail_ent.raw_label,
                    )
                )

        # -- Primary: entity-level relations (standard radgraph format) ------
        for ent_id, ent_data in raw_entities.items():
            ent_relations = ent_data.get("relations", [])
            for rel in ent_relations:
                if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                    _add_triple(str(ent_id), str(rel[1]), str(rel[0]))
                elif isinstance(rel, dict):
                    _add_triple(
                        str(ent_id),
                        str(rel.get("target", rel.get("tail", ""))),
                        str(rel.get("type", rel.get("relation", ""))),
                    )

        # -- Fallback: top-level relations list/dict -------------------------
        raw_relations = prediction.get("relations", [])
        if isinstance(raw_relations, list):
            for rel_data in raw_relations:
                if isinstance(rel_data, dict):
                    _add_triple(
                        str(rel_data.get("head", rel_data.get("source", ""))),
                        str(rel_data.get("tail", rel_data.get("target", ""))),
                        str(rel_data.get("relation", rel_data.get("type", ""))),
                    )
        elif isinstance(raw_relations, dict):
            for head_id, rel_list in raw_relations.items():
                for rel in (rel_list if isinstance(rel_list, list) else []):
                    if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                        _add_triple(str(head_id), str(rel[1]), str(rel[0]))
                    elif isinstance(rel, dict):
                        _add_triple(
                            str(head_id),
                            str(rel.get("target", "")),
                            str(rel.get("type", "")),
                        )

        entities_list = list(entities_dict.values())
        return entities_list, triples

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_report(text: str) -> str:
        """Clean report text before extraction."""
        text = re.sub(r"_{3,}", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def filter_eligible(
        entities: List[ExtractedEntity],
    ) -> List[ExtractedEntity]:
        """Return only entities eligible for CUI grounding.

        Excludes:
        - Measurement entities
        - Entities whose text matches a measurement pattern
        """
        eligible = []
        for ent in entities:
            if ent.entity_type == EntityType.MEASUREMENT:
                continue
            if is_measurement(ent.text):
                continue
            eligible.append(ent)
        return eligible

    @staticmethod
    def group_by_type(
        entities: List[ExtractedEntity],
    ) -> Dict[EntityType, List[ExtractedEntity]]:
        """Group entities by their coarse type."""
        groups: Dict[EntityType, List[ExtractedEntity]] = {}
        for ent in entities:
            groups.setdefault(ent.entity_type, []).append(ent)
        return groups

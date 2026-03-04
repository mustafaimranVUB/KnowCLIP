"""UMLS ontology grounding via deterministic exact matching on MRCONSO.RRF.

Implements the two-stage grounding strategy described in the design
document (Section 7.3):

1. **Deterministic exact matching** against MRCONSO.RRF (primary).
2. scispaCy fuzzy matching (fallback — separate module).
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.knowledge.extraction import (
    EntityType,
    ExtractedEntity,
    canonicalize_surface,
    normalize_text,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SAB preference order (higher = better)
SAB_PRIORITY: Dict[str, int] = {
    "SNOMEDCT_US": 100,
    "RXNORM": 90,
    "MSH": 80,
    "NCI": 70,
    "AOD": 60,
    "HPO": 50,
}

# Preferred term types (boolean flag: is preferred?)
PREFERRED_TTYS: Set[str] = {"PT", "PN", "HT", "MH", "PEP", "FN"}

# Stopwords for head-noun generation
_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "in", "on", "at", "to", "with", "for",
    "is", "are", "was", "were", "no", "not", "and", "or", "but", "by",
    "from", "this", "that", "there", "their", "its", "some", "any",
}

# Laterality / severity modifiers to strip for head-noun extraction
_MODIFIERS: Set[str] = {
    "left", "right", "bilateral", "upper", "lower", "mild", "moderate",
    "severe", "small", "large", "diffuse", "focal", "acute", "chronic",
    "minimal", "significant", "marked", "subtle", "slight",
}

# TUIs for semantic type validation
ANATOMY_TUIS: Set[str] = {
    "T017", "T021", "T022", "T023", "T024", "T025", "T029", "T030",
}
OBSERVATION_TUIS: Set[str] = {
    "T033", "T034", "T037", "T046", "T047", "T048", "T184", "T190", "T191",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CUICandidate:
    """A candidate CUI mapping for a mention."""
    cui: str
    preferred_name: str
    source: str  # SAB
    term_type: str  # TTY
    is_preferred: bool  # ISPREF
    key_type: str  # 'full_norm', 'full_can', 'head'
    score: float = 0.0  # Computed ranking score


@dataclass
class GroundingResult:
    """Result of grounding a single entity mention."""
    mention: str
    normalized: str
    canonical: str
    best_cui: Optional[str] = None
    best_name: Optional[str] = None
    candidates: List[CUICandidate] = field(default_factory=list)
    mapped: bool = False
    tuis: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MRCONSO index
# ---------------------------------------------------------------------------

class MRCONSOIndex:
    """In-memory inverted index over MRCONSO.RRF for fast exact matching.

    Parameters:
        mrconso_path: Path to ``MRCONSO.RRF``.
        language: ISO-639 language code to filter on (default ``ENG``).
        sab_filter: Optional set of source abbreviations to include.
    """

    def __init__(
        self,
        mrconso_path: Path | str,
        language: str = "ENG",
        sab_filter: Optional[Set[str]] = None,
    ) -> None:
        self.mrconso_path = Path(mrconso_path)
        self.language = language
        self.sab_filter = sab_filter

        # canonical_surface → list of (CUI, STR, SAB, TTY, ISPREF)
        self._index: Dict[str, List[Tuple[str, str, str, str, bool]]] = defaultdict(list)
        self._loaded = False

    def load(self) -> None:
        """Build the index by scanning MRCONSO.RRF.

        MRCONSO.RRF columns (pipe-delimited, no header):
        CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
          0   1   2   3   4    5    6    7     8    9   10  11  12  13   14  15  16       17
        """
        import sys
        csv.field_size_limit(sys.maxsize)
        logger.info("Loading MRCONSO.RRF from %s ...", self.mrconso_path)
        count = 0
        with open(self.mrconso_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) < 17:
                    continue
                lat = row[1]
                if lat != self.language:
                    continue

                sab = row[11]
                if self.sab_filter and sab not in self.sab_filter:
                    continue

                cui = row[0]
                tty = row[12]
                ispref = row[6] == "Y"
                raw_str = row[14]
                suppress = row[16]

                # Skip suppressed entries
                if suppress in ("O", "E", "Y"):
                    continue

                canonical = canonicalize_surface(raw_str)
                if not canonical:
                    continue

                self._index[canonical].append((cui, raw_str, sab, tty, ispref))
                count += 1

        self._loaded = True
        logger.info(
            "MRCONSO index built: %d unique surfaces, %d total entries",
            len(self._index),
            count,
        )

    def lookup(self, canonical_key: str) -> List[Tuple[str, str, str, str, bool]]:
        """Look up a canonicalized surface string.

        Returns:
            List of ``(CUI, STR, SAB, TTY, ISPREF)`` tuples.
        """
        if not self._loaded:
            self.load()
        return self._index.get(canonical_key, [])

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# MRSTY index (semantic types)
# ---------------------------------------------------------------------------

class MRSTYIndex:
    """Index of CUI → semantic types from MRSTY.RRF.

    MRSTY.RRF columns:
    CUI|TUI|STN|STY|ATUI|CVF
      0   1   2   3   4    5
    """

    def __init__(self, mrsty_path: Path | str) -> None:
        self.mrsty_path = Path(mrsty_path)
        self._cui_to_tuis: Dict[str, Set[str]] = defaultdict(set)
        self._loaded = False

    def load(self) -> None:
        import sys
        csv.field_size_limit(sys.maxsize)
        logger.info("Loading MRSTY.RRF from %s ...", self.mrsty_path)
        with open(self.mrsty_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) < 4:
                    continue
                cui, tui = row[0], row[1]
                self._cui_to_tuis[cui].add(tui)
        self._loaded = True
        logger.info("MRSTY index: %d CUIs loaded", len(self._cui_to_tuis))

    def get_tuis(self, cui: str) -> Set[str]:
        if not self._loaded:
            self.load()
        return self._cui_to_tuis.get(cui, set())

    def is_anatomy(self, cui: str) -> bool:
        return bool(self.get_tuis(cui) & ANATOMY_TUIS)

    def is_observation(self, cui: str) -> bool:
        return bool(self.get_tuis(cui) & OBSERVATION_TUIS)

    def validate_entity_type(
        self, cui: str, entity_type: EntityType
    ) -> bool:
        """Check if a CUI's semantic types are compatible with the entity type."""
        tuis = self.get_tuis(cui)
        if not tuis:
            return True  # No TUI data → pass by default

        if entity_type == EntityType.ANATOMY:
            return bool(tuis & ANATOMY_TUIS)
        elif entity_type == EntityType.OBSERVATION:
            return bool(tuis & OBSERVATION_TUIS)
        return True  # Measurement or unknown → always valid


# ---------------------------------------------------------------------------
# MRREL index (relationship validation)
# ---------------------------------------------------------------------------

class MRRELIndex:
    """Index CUI1-REL-CUI2 triples from MRREL.RRF for validation.

    Only loads the relationship types we care about for ontology edges.

    MRREL.RRF columns:
    CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|RG|DIR|SUPPRESS|CVF
      0    1    2     3    4    5     6      7   8    9   10  11  12 13    14      15
    """

    VALID_RELS: Set[str] = {"PAR", "CHD", "RB", "RN", "SIB"}

    def __init__(self, mrrel_path: Path | str) -> None:
        self.mrrel_path = Path(mrrel_path)
        # (CUI1, REL, CUI2) set for fast lookup
        self._triples: Set[Tuple[str, str, str]] = set()
        self._loaded = False

    def load(self) -> None:
        import sys
        csv.field_size_limit(sys.maxsize)
        logger.info("Loading MRREL.RRF from %s ...", self.mrrel_path)
        count = 0
        with open(self.mrrel_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) < 8:
                    continue
                rel = row[3]
                if rel not in self.VALID_RELS:
                    continue
                cui1, cui2 = row[0], row[4]
                self._triples.add((cui1, rel, cui2))
                count += 1
        self._loaded = True
        logger.info("MRREL index: %d relationship triples loaded", count)

    def has_relation(self, cui1: str, rel: str, cui2: str) -> bool:
        if not self._loaded:
            self.load()
        return (cui1, rel, cui2) in self._triples

    def get_neighbours(self, cui: str) -> List[Tuple[str, str]]:
        """Get all (rel, CUI2) neighbours of a CUI."""
        if not self._loaded:
            self.load()
        return [
            (rel, c2) for (c1, rel, c2) in self._triples if c1 == cui
        ]


# ---------------------------------------------------------------------------
# Ontology Grounder
# ---------------------------------------------------------------------------

class OntologyGrounder:
    """Ground extracted entities to UMLS CUIs via deterministic exact matching.

    Stage 1 (primary) uses exact MRCONSO.RRF matching.  Stage 2 (optional
    fallback) uses scispaCy EntityLinker for mentions that receive no
    candidate from Stage 1.

    Parameters:
        mrconso_path: Path to MRCONSO.RRF.
        mrsty_path: Optional path to MRSTY.RRF for semantic-type validation.
        mrrel_path: Optional path to MRREL.RRF for relationship validation.
        top_k: Number of candidate CUIs to keep per mention.
        validate_semantic_types: Whether to check TUI compatibility.
        scispacy_grounder: Optional :class:`~src.knowledge.scispacy_grounding.ScispaCyGrounder`
            instance used as a Stage-2 fallback for unmapped mentions.
    """

    def __init__(
        self,
        mrconso_path: Path | str,
        mrsty_path: Optional[Path | str] = None,
        mrrel_path: Optional[Path | str] = None,
        top_k: int = 5,
        validate_semantic_types: bool = True,
        scispacy_grounder: Optional[Any] = None,
    ) -> None:
        self.mrconso_index = MRCONSOIndex(mrconso_path)
        self.mrsty_index = MRSTYIndex(mrsty_path) if mrsty_path else None
        self.mrrel_index = MRRELIndex(mrrel_path) if mrrel_path else None
        self.top_k = top_k
        self.validate_semantic_types = validate_semantic_types
        self.scispacy_grounder = scispacy_grounder  # Stage-2 fallback

    def load_indices(self) -> None:
        """Eagerly load all UMLS indices."""
        self.mrconso_index.load()
        if self.mrsty_index:
            self.mrsty_index.load()
        if self.mrrel_index:
            self.mrrel_index.load()

    # ------------------------------------------------------------------
    # Main grounding entry point
    # ------------------------------------------------------------------

    def ground_entities(
        self,
        entities: List[ExtractedEntity],
    ) -> List[GroundingResult]:
        """Ground a list of entities to UMLS CUIs.

        Args:
            entities: Entities from :class:`EntityExtractor`.

        Returns:
            List of :class:`GroundingResult`, one per entity.
        """
        if not self.mrconso_index.is_loaded:
            self.mrconso_index.load()

        results: List[GroundingResult] = []
        for ent in entities:
            result = self._ground_single(ent)
            results.append(result)

        return results

    def ground_to_mention2cui(
        self,
        entities: List[ExtractedEntity],
    ) -> Dict[str, Dict[str, Any]]:
        """Ground entities and return a mention → CUI mapping dict.

        Returns:
            Dict mapping ``normalized_text`` →
            ``{best_cui, best_name, candidates, mapped, tuis}``.
        """
        results = self.ground_entities(entities)
        mention2cui: Dict[str, Dict[str, Any]] = {}
        for res in results:
            if res.normalized not in mention2cui:
                mention2cui[res.normalized] = {
                    "best_cui": res.best_cui,
                    "best_name": res.best_name,
                    "candidates": [
                        {"cui": c.cui, "name": c.preferred_name, "source": c.source}
                        for c in res.candidates
                    ],
                    "mapped": res.mapped,
                    "tuis": res.tuis,
                }
        return mention2cui

    # ------------------------------------------------------------------
    # Single-entity grounding
    # ------------------------------------------------------------------

    def _ground_single(self, entity: ExtractedEntity) -> GroundingResult:
        """Ground a single entity through the multi-key lookup pipeline.

        Stage 1: Deterministic exact matching against MRCONSO.RRF.
        Stage 2 (fallback): scispaCy EntityLinker, if configured and Stage
        1 produced no candidates.
        """
        norm = normalize_text(entity.text)
        canon = canonicalize_surface(entity.text)

        result = GroundingResult(
            mention=entity.text,
            normalized=norm,
            canonical=canon,
        )

        # ---- Stage 1: MRCONSO exact matching -------------------------
        # Generate lookup keys in priority order
        keys = self._generate_keys(norm, canon)

        # Collect candidates across all keys
        all_candidates: List[CUICandidate] = []
        for key, key_type in keys:
            hits = self.mrconso_index.lookup(key)
            for cui, raw_str, sab, tty, ispref in hits:
                candidate = CUICandidate(
                    cui=cui,
                    preferred_name=raw_str,
                    source=sab,
                    term_type=tty,
                    is_preferred=ispref,
                    key_type=key_type,
                )
                candidate.score = self._rank_score(candidate, key_type)
                all_candidates.append(candidate)

        if all_candidates:
            # De-duplicate by CUI, keeping the best-scoring entry per CUI
            best_per_cui: Dict[str, CUICandidate] = {}
            for c in all_candidates:
                if c.cui not in best_per_cui or c.score > best_per_cui[c.cui].score:
                    best_per_cui[c.cui] = c

            # Sort by score (descending) and take top K
            ranked = sorted(best_per_cui.values(), key=lambda c: c.score, reverse=True)

            # Optional semantic-type validation
            if self.validate_semantic_types and self.mrsty_index:
                ranked = self._filter_by_semantic_type(ranked, entity.entity_type)

            ranked = ranked[: self.top_k]
            result.candidates = ranked

            if ranked:
                best = ranked[0]
                result.best_cui = best.cui
                result.best_name = best.preferred_name
                result.mapped = True

                # Retrieve TUIs
                if self.mrsty_index:
                    result.tuis = sorted(self.mrsty_index.get_tuis(best.cui))

            return result

        # ---- Stage 2: scispaCy fallback (if configured) -------------
        if self.scispacy_grounder is not None:
            from src.knowledge.scispacy_grounding import ScispaCyResult

            sci_result: Optional[ScispaCyResult] = self.scispacy_grounder.ground_mention(
                entity.text
            )
            if sci_result is not None:
                logger.debug(
                    "scispaCy fallback grounded '%s' → %s (%.3f)",
                    entity.text,
                    sci_result.cui,
                    sci_result.confidence,
                )
                # Wrap in a CUICandidate-compatible structure for consistency
                fallback_candidate = CUICandidate(
                    cui=sci_result.cui,
                    preferred_name=sci_result.canonical_name,
                    source="scispacy",
                    term_type="scispacy",
                    is_preferred=True,
                    key_type="scispacy",
                    score=float(sci_result.confidence) * 1000,  # normalise to score scale
                )
                result.candidates = [fallback_candidate]
                result.best_cui = sci_result.cui
                result.best_name = sci_result.canonical_name
                result.mapped = True

                if self.mrsty_index:
                    result.tuis = sorted(self.mrsty_index.get_tuis(sci_result.cui))

        return result

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_keys(norm: str, canon: str) -> List[Tuple[str, str]]:
        """Generate lookup keys in priority order.

        Returns:
            List of ``(key_string, key_type)`` tuples.
        """
        keys: List[Tuple[str, str]] = []

        # 1. Full normalised
        keys.append((canon, "full_norm"))

        # 2. Full canonical (if different)
        if canon != norm:
            keys.append((canonicalize_surface(norm), "full_can"))

        # 3. Head-noun candidates
        head_nouns = OntologyGrounder._extract_head_nouns(norm)
        for hn in head_nouns:
            keys.append((canonicalize_surface(hn), "head"))

        return keys

    @staticmethod
    def _extract_head_nouns(text: str) -> List[str]:
        """Extract head-noun candidates from a normalised mention.

        Strategy: take the last 1, 2, 3 content words after removing
        stopwords, laterality, and severity modifiers.
        """
        words = text.split()
        content = [w for w in words if w not in _STOPWORDS and w not in _MODIFIERS]

        if not content:
            return []

        heads: List[str] = []
        for n in (1, 2, 3):
            if n <= len(content):
                candidate = " ".join(content[-n:])
                if candidate != text:  # avoid duplicate of full string
                    heads.append(candidate)

        return heads

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_score(candidate: CUICandidate, key_type: str) -> float:
        """Compute a deterministic ranking score for a candidate.

        Higher = better.
        """
        score = 0.0

        # Key-type priority
        key_priority = {"full_norm": 1000, "full_can": 800, "head": 500}
        score += key_priority.get(key_type, 0)

        # Preferred term flag
        if candidate.is_preferred:
            score += 200

        # Source preference
        score += SAB_PRIORITY.get(candidate.source, 10)

        # Term-type preference
        if candidate.term_type in PREFERRED_TTYS:
            score += 50

        return score

    def _filter_by_semantic_type(
        self,
        candidates: List[CUICandidate],
        entity_type: EntityType,
    ) -> List[CUICandidate]:
        """Remove candidates whose TUIs are incompatible with the entity type."""
        if self.mrsty_index is None:
            return candidates

        filtered = []
        for c in candidates:
            if self.mrsty_index.validate_entity_type(c.cui, entity_type):
                filtered.append(c)

        # If all filtered out, fall back to unfiltered (don't lose all candidates)
        return filtered if filtered else candidates

    # ------------------------------------------------------------------
    # Coverage statistics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_coverage(results: List[GroundingResult]) -> Dict[str, Any]:
        """Compute grounding coverage statistics.

        Returns:
            Dict with keys: total, mapped, unmapped, coverage_pct,
            unique_cuis, unique_mentions.
        """
        total = len(results)
        mapped = sum(1 for r in results if r.mapped)
        unique_cuis = len({r.best_cui for r in results if r.best_cui})
        unique_mentions = len({r.normalized for r in results})

        return {
            "total": total,
            "mapped": mapped,
            "unmapped": total - mapped,
            "coverage_pct": (mapped / total * 100) if total > 0 else 0.0,
            "unique_cuis": unique_cuis,
            "unique_mentions": unique_mentions,
        }

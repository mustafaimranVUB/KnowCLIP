"""scispaCy-based fuzzy entity grounding (Stage 2 fallback).

Used as a fallback when UMLS exact MRCONSO matching produces no candidates
for a given entity mention.  Relies on ``en_core_sci_lg`` + the
``scispacy.linking.EntityLinker`` UMLS linker.

Cache configuration
-------------------
The ``en_core_sci_lg`` model and the UMLS KB index can be large.  On
Hydra HPC the preferred storage location is ``$VSC_SCRATCH`` (fast,
large, not backed up).  Point the environment variable
``SCISPACY_CACHE`` at a directory on scratch::

    export SCISPACY_CACHE=$VSC_SCRATCH/scispacy_cache

The model package should be installed into that directory::

    pip install \\
        https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_lg-0.5.5.tar.gz \\
        --target $VSC_SCRATCH/scispacy_cache

The :class:`ScispaCyGrounder` will then prepend the cache dir to
``sys.path`` so Python can find the package without polluting the main
environment.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScispaCyResult:
    """A single scispaCy grounding result."""
    mention: str
    cui: str
    canonical_name: str
    confidence: float
    source: str = "scispacy"


# ---------------------------------------------------------------------------
# ScispaCy grounder
# ---------------------------------------------------------------------------

class ScispaCyGrounder:
    """Fuzzy UMLS grounding via ``en_core_sci_lg`` + EntityLinker.

    Parameters:
        cache_dir: Directory where the ``en_core_sci_lg`` model package
            was installed via ``pip install --target``.  Falls back to
            the ``SCISPACY_CACHE`` environment variable, and finally to
            the standard Python path if neither is set.
        confidence_threshold: Minimum linker confidence to accept a CUI
            (default 0.85, as per design spec Section 7.3).
        resolve_abbreviations: Whether to run the abbreviation resolver
            (adds latency but improves coverage for abbreviation-heavy
            impressions).
    """

    #: Environment variable name for the model cache directory.
    ENV_VAR = "SCISPACY_CACHE"

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        confidence_threshold: float = 0.85,
        resolve_abbreviations: bool = True,
    ) -> None:
        resolved = cache_dir or os.environ.get(self.ENV_VAR, "")
        self.cache_dir: Optional[Path] = Path(resolved) if resolved else None
        self.confidence_threshold = confidence_threshold
        self.resolve_abbreviations = resolve_abbreviations

        self._nlp: Any = None  # lazy-loaded spaCy pipeline
        self._linker_added: bool = False

    # ------------------------------------------------------------------
    # Lazy pipeline access
    # ------------------------------------------------------------------

    @property
    def nlp(self) -> Any:
        """Return the loaded spaCy + EntityLinker pipeline."""
        if self._nlp is None:
            self._nlp = self._load_pipeline()
        return self._nlp

    def _load_pipeline(self) -> Any:
        """Load the ``en_core_sci_lg`` pipeline and attach the UMLS linker."""
        spacy = self._import_spacy()

        # --- Try loading from cache_dir first -------------------------
        if self.cache_dir is not None and self.cache_dir.is_dir():
            # Option A: the model was installed directly as a subdirectory
            model_subdir = self.cache_dir / "en_core_sci_lg"
            if model_subdir.exists():
                try:
                    nlp = spacy.load(str(model_subdir))
                    logger.info("Loaded en_core_sci_lg from %s", model_subdir)
                    return self._attach_linker(nlp, spacy)
                except Exception as exc:
                    logger.debug(
                        "Direct load from %s failed (%s); trying sys.path injection",
                        model_subdir,
                        exc,
                    )

            # Option B: pip install --target puts the package into cache_dir;
            # inject into sys.path so Python can find it.
            if str(self.cache_dir) not in sys.path:
                sys.path.insert(0, str(self.cache_dir))
                logger.debug("Injected %s into sys.path for scispaCy", self.cache_dir)

            try:
                nlp = spacy.load("en_core_sci_lg")
                logger.info(
                    "Loaded en_core_sci_lg from sys.path (cache=%s)", self.cache_dir
                )
                return self._attach_linker(nlp, spacy)
            except OSError as exc:
                logger.warning(
                    "Could not load en_core_sci_lg with cache_dir=%s: %s. "
                    "Falling back to standard spaCy search path.",
                    self.cache_dir,
                    exc,
                )

        # --- Standard spaCy search path (model installed normally) ---
        try:
            nlp = spacy.load("en_core_sci_lg")
            logger.info("Loaded en_core_sci_lg from standard spaCy path")
            return self._attach_linker(nlp, spacy)
        except OSError as exc:
            raise RuntimeError(
                "en_core_sci_lg not found.  Install it with:\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
                "/releases/v0.5.5/en_core_sci_lg-0.5.5.tar.gz\n"
                "Or for HPC (store in $VSC_SCRATCH/scispacy_cache):\n"
                "  pip install <url> --target $VSC_SCRATCH/scispacy_cache\n"
                "  export SCISPACY_CACHE=$VSC_SCRATCH/scispacy_cache"
            ) from exc

    def _attach_linker(self, nlp: Any, spacy: Any) -> Any:
        """Attach the scispaCy UMLS EntityLinker to a loaded pipeline."""
        self._ensure_scispacy_factories_registered()

        # Abbreviation detector (optional but recommended)
        if self.resolve_abbreviations and "abbreviation_detector" not in nlp.pipe_names:
            try:
                nlp.add_pipe("abbreviation_detector", before="ner")
                logger.debug("Added abbreviation_detector to pipeline")
            except Exception as exc:
                logger.debug("Could not add abbreviation_detector: %s", exc)

        # UMLS entity linker
        if "scispacy_linker" not in nlp.pipe_names:
            nlp.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": self.resolve_abbreviations,
                    "linker_name": "umls",
                    "filter_for_definitions": False,
                    "k": 10,
                    "threshold": self.confidence_threshold,
                },
            )
            logger.info(
                "Attached scispaCy UMLS linker (threshold=%.2f)",
                self.confidence_threshold,
            )
            self._linker_added = True

        return nlp

    @staticmethod
    def _ensure_scispacy_factories_registered() -> None:
        """Import scispaCy modules that register custom spaCy factories.

        Some HPC setups have the model package available on ``sys.path`` while
        the factory registration side-effects are missing until these modules
        are imported explicitly.
        """
        try:
            # Import side-effects register: abbreviation_detector, scispacy_linker.
            import scispacy.abbreviation  # noqa: F401
            import scispacy.linking  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "scispaCy is required for Stage-2 fuzzy grounding but is not "
                "installed in the active environment. Install with:\n"
                "  pip install scispacy==0.5.5\n"
                "and ensure compatible spaCy is installed."
            ) from exc

    @staticmethod
    def _import_spacy() -> Any:
        """Import spaCy, raising a clear error if not installed."""
        try:
            import spacy  # type: ignore

            return spacy
        except ImportError:
            raise ImportError(
                "spaCy is not installed.  Install with:\n"
                "  pip install scispacy==0.5.5 spacy"
            )

    # ------------------------------------------------------------------
    # Grounding API
    # ------------------------------------------------------------------

    def ground_mention(
        self,
        mention: str,
    ) -> Optional[ScispaCyResult]:
        """Ground a single mention to a UMLS CUI.

        Args:
            mention: Entity surface form (e.g. ``'pleural effusion'``).

        Returns:
            :class:`ScispaCyResult` with the best CUI above threshold,
            or ``None`` if no confident match is found.
        """
        if not mention or not mention.strip():
            return None

        doc = self.nlp(mention.strip())

        best: Optional[ScispaCyResult] = None
        for ent in doc.ents:
            if not hasattr(ent._, "kb_ents"):
                continue
            for cui, score in ent._.kb_ents:
                score = float(score)
                if score < self.confidence_threshold:
                    continue
                if best is None or score > best.confidence:
                    canonical = self._get_canonical_name(cui)
                    best = ScispaCyResult(
                        mention=mention,
                        cui=cui,
                        canonical_name=canonical,
                        confidence=score,
                    )

        return best

    def ground_batch(
        self,
        mentions: List[str],
    ) -> Dict[str, Optional[ScispaCyResult]]:
        """Ground a list of mentions.

        Args:
            mentions: List of entity surface forms.

        Returns:
            Dict mapping each mention string to its best
            :class:`ScispaCyResult` (or ``None`` if not matched).
        """
        results: Dict[str, Optional[ScispaCyResult]] = {}
        unique = list(dict.fromkeys(m for m in mentions if m))  # dedup & preserve order

        logger.debug("scispaCy grounding %d unique mentions ...", len(unique))

        # Process in batches to avoid re-tokenising repeated strings
        for mention in unique:
            results[mention] = self.ground_mention(mention)

        # Fill in any missing keys from the original list
        for m in mentions:
            results.setdefault(m, None)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_canonical_name(self, cui: str) -> str:
        """Retrieve the canonical name for a CUI from the linker KB."""
        try:
            linker = self.nlp.get_pipe("scispacy_linker")
            if hasattr(linker, "kb") and hasattr(linker.kb, "cui_to_entity"):
                entity = linker.kb.cui_to_entity.get(cui)
                if entity is not None:
                    return entity.canonical_name
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Installation helper (standalone utility)
    # ------------------------------------------------------------------

    @staticmethod
    def is_available(cache_dir: Optional[Path | str] = None) -> bool:
        """Check whether the ``en_core_sci_lg`` model can be loaded.

        Performs a cheap import check without actually loading the full
        pipeline.  Returns ``True`` if the model package is importable
        (either from *cache_dir* injected into ``sys.path`` or from the
        standard Python path).
        """
        resolved = cache_dir or os.environ.get(ScispaCyGrounder.ENV_VAR, "")
        cache_path = Path(resolved) if resolved else None

        # Temporarily add cache_dir to sys.path for the probe
        added = False
        if cache_path is not None and cache_path.is_dir():
            if str(cache_path) not in sys.path:
                sys.path.insert(0, str(cache_path))
                added = True
        try:
            import importlib.util
            spec = importlib.util.find_spec("en_core_sci_lg")
            if spec is not None:
                return True
            # Also check if the model directory exists directly
            if cache_path is not None:
                model_subdir = cache_path / "en_core_sci_lg"
                if model_subdir.exists():
                    return True
            return False
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
        finally:
            if added:
                try:
                    sys.path.remove(str(cache_path))
                except ValueError:
                    pass

    @classmethod
    def download_model(cls, target_dir: Path | str) -> None:
        """Download and install ``en_core_sci_lg`` to *target_dir*.

        Intended to be called once on HPC before running the pipeline::

            from src.knowledge.scispacy_grounding import ScispaCyGrounder
            ScispaCyGrounder.download_model("$VSC_SCRATCH/scispacy_cache")

        Args:
            target_dir: Directory to install the model package into.
        """
        import subprocess

        url = (
            "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.5/en_core_sci_lg-0.5.5.tar.gz"
        )
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        logger.info("Installing en_core_sci_lg to %s ...", target)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", url, "--target", str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pip install failed:\n{result.stderr}"
            )
        logger.info("en_core_sci_lg installed to %s", target)

    @classmethod
    def is_available(cls, cache_dir: Optional[Path | str] = None) -> bool:
        """Check whether scispaCy + en_core_sci_lg can be loaded.

        Args:
            cache_dir: Optional cache directory to check.

        Returns:
            True if the model is importable.
        """
        resolved = cache_dir or os.environ.get(cls.ENV_VAR, "")
        cache = Path(resolved) if resolved else None

        try:
            import spacy  # type: ignore  # noqa: F401
            import scispacy  # type: ignore  # noqa: F401
        except ImportError:
            return False

        # Temporarily inject cache dir
        injected = False
        if cache and cache.is_dir() and str(cache) not in sys.path:
            sys.path.insert(0, str(cache))
            injected = True

        try:
            import en_core_sci_lg  # type: ignore  # noqa: F401

            return True
        except ImportError:
            pass
        finally:
            if injected:
                try:
                    sys.path.remove(str(cache))
                except ValueError:
                    pass

        # Try spacy.util.is_package
        try:
            import spacy  # type: ignore

            return spacy.util.is_package("en_core_sci_lg")
        except Exception:
            return False

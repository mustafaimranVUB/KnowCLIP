"""Radiology report text loading and cleaning for MIMIC-CXR.

Reports are stored as plain-text files in the PhysioNet MIMIC-CXR-Reports
distribution under ``files/p{10..19}/p{patient_id}/s{study_id}.txt``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section headers found in MIMIC-CXR reports
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(
    r"^(FINDINGS|IMPRESSION|INDICATION|TECHNIQUE|COMPARISON|HISTORY|EXAMINATION|"
    r"CLINICAL INFORMATION|REASON FOR EXAMINATION|WET READ):?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class ReportLoader:
    """Parse and extract sections from MIMIC-CXR radiology reports.

    Parameters:
        reports_root: Root directory containing the report text files
            (typically ``mimic-cxr-reports/files``).
        section_preference: Which report section to prioritise. Supported:
            ``impression``, ``findings``, ``auto``.
        prefer_impression: Deprecated compatibility flag. If provided, it
            maps to ``section_preference`` and is ignored otherwise.
    """

    def __init__(
        self,
        reports_root: Path | str,
        section_preference: str = "impression",
        prefer_impression: Optional[bool] = None,
    ) -> None:
        self.reports_root = Path(reports_root)
        if prefer_impression is not None:
            section_preference = "impression" if prefer_impression else "findings"

        pref = str(section_preference or "impression").strip().lower()
        if pref == "auto":
            pref = "impression"
        if pref not in {"impression", "findings"}:
            logger.warning(
                "Unknown report section preference '%s'; defaulting to 'impression'",
                section_preference,
            )
            pref = "impression"
        self.section_preference = pref

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_report_path(self, subject_id: int, study_id: int) -> Path:
        """Build the path to a report file given subject and study IDs.

        MIMIC-CXR stores reports as::
            files/p{prefix}/p{subject_id}/s{study_id}.txt

        where *prefix* is the first two digits of ``subject_id`` prefixed
        with 'p' (e.g. ``p10``, ``p11``, …).
        """
        prefix = f"p{str(subject_id)[:2]}"
        return self.reports_root / prefix / f"p{subject_id}" / f"s{study_id}.txt"

    def load_report(
        self,
        subject_id: int,
        study_id: int,
    ) -> Optional[str]:
        """Load and return the preferred section of a report.

        Returns the configured preferred section (impression/findings), then
        falls back to the alternate section, then report body.
        """
        path = self.get_report_path(subject_id, study_id)
        if not path.exists():
            logger.debug("Report not found: %s", path)
            return None

        raw = path.read_text(encoding="utf-8", errors="replace")
        sections = self.parse_sections(raw)

        impression = sections.get("impression", "").strip()
        findings = sections.get("findings", "").strip()

        if self.section_preference == "findings":
            if findings:
                return self.clean_text(findings)
            if impression:
                return self.clean_text(impression)
        else:
            if impression:
                return self.clean_text(impression)
            if findings:
                return self.clean_text(findings)

        # Fall back to entire report body (rare edge-case)
        body = self.clean_text(raw)
        return body if body else None

    def load_raw(self, subject_id: int, study_id: int) -> Optional[str]:
        """Load the complete raw report text."""
        path = self.get_report_path(subject_id, study_id)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8", errors="replace")

    def load_sections(
        self,
        subject_id: int,
        study_id: int,
    ) -> Dict[str, str]:
        """Load and return **all** parsed sections of a report."""
        raw = self.load_raw(subject_id, study_id)
        if raw is None:
            return {}
        return self.parse_sections(raw)

    def scan_all_studies(
        self,
        max_studies: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Discover (subject_id, study_id) pairs from the directory tree.

        Walks ``self.reports_root`` looking for files matching the MIMIC-CXR
        convention ``p{prefix}/p{subject_id}/s{study_id}.txt``.

        Args:
            max_studies: If set, stop after discovering this many unique
                (subject_id, study_id) pairs.

        Returns:
            List of dicts with ``subject_id`` (int) and ``study_id`` (int).
        """
        studies: List[Dict[str, Any]] = []
        seen: set = set()

        if not self.reports_root.exists() or not self.reports_root.is_dir():
            logger.warning(
                "Reports root does not exist or is not a directory: %s",
                self.reports_root,
            )
            return studies

        # Walk p{prefix}/p{subject_id}/s{study_id}.txt
        for report_file in sorted(self.reports_root.rglob("s*.txt")):
            name = report_file.stem  # e.g. "s12345678"
            parent = report_file.parent.name  # e.g. "p10000032"

            # Validate naming patterns
            if not name.startswith("s") or not parent.startswith("p"):
                continue

            try:
                study_id = int(name[1:])
                subject_id = int(parent[1:])
            except (ValueError, IndexError):
                continue

            key = (subject_id, study_id)
            if key in seen:
                continue
            seen.add(key)
            studies.append({"subject_id": subject_id, "study_id": study_id})

            if max_studies is not None and len(studies) >= max_studies:
                break

        logger.info(
            "Scanned %d unique studies from %s",
            len(studies),
            self.reports_root,
        )
        return studies

    # ------------------------------------------------------------------
    # Section parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_sections(text: str) -> Dict[str, str]:
        """Parse a report string into a dictionary of named sections.

        Returns:
            Mapping of lower-case section names to their text content.
        """
        matches = list(_SECTION_RE.finditer(text))
        if not matches:
            return {"body": text.strip()}

        sections: Dict[str, str] = {}
        for i, m in enumerate(matches):
            name = m.group(1).strip().lower()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections[name] = content
        return sections

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalise report text for downstream processing.

        * Replaces ``___`` de-identification placeholders with empty string.
        * Collapses multi-whitespace to single space.
        * Strips leading/trailing whitespace.
        """
        text = re.sub(r"_{3,}", "", text)       # Remove ___ placeholders
        text = re.sub(r"\s+", " ", text)         # Collapse whitespace
        text = text.strip()
        return text

    # ------------------------------------------------------------------
    # Convenience: study-level score for view selection
    # ------------------------------------------------------------------

    @staticmethod
    def score_view(view_position: Optional[str]) -> int:
        """Return a priority score for DICOM ViewPosition.

        PA > AP > LATERAL > other.  Higher score = preferred.
        """
        if view_position is None:
            return 0
        view = view_position.upper().strip()
        priorities = {"PA": 4, "AP": 3, "LATERAL": 2, "LL": 1}
        return priorities.get(view, 0)

    @staticmethod
    def select_canonical_image(
        records: list[Tuple[str, Optional[str]]],
    ) -> Optional[str]:
        """Select the best single image per study.

        Args:
            records: List of ``(dicom_id, view_position)`` tuples.

        Returns:
            The ``dicom_id`` of the best view, or ``None`` if empty.
        """
        if not records:
            return None
        scored = sorted(
            records,
            key=lambda r: ReportLoader.score_view(r[1]),
            reverse=True,
        )
        return scored[0][0]

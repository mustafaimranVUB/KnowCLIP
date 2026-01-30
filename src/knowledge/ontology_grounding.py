"""Deterministic UMLS grounding mirroring the tested notebook pipeline."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


class UMLSExactMatcher:
    """Exact-match MRCONSO grounding as implemented in the notebook."""

    STOP = {
        "no",
        "not",
        "without",
        "with",
        "of",
        "in",
        "on",
        "at",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "for",
    }
    LATERAL = {"left", "right", "bilateral", "bl", "b", "l", "r"}
    SEVERITY = {
        "mild",
        "mildly",
        "moderate",
        "moderately",
        "severe",
        "marked",
        "slight",
        "small",
        "tiny",
        "minimal",
        "possible",
        "likely",
    }

    MEAS_RE = re.compile(r"\b\d+(\.\d+)?\s*(mm|cm|m)\b")

    KEYTYPE_PRIORITY = {"full_norm": 0, "full_can": 1, "head": 2}

    def __init__(
        self,
        mrconso_path: str,
        sab_preference: Optional[List[str]] = None,
        top_k: int = 5,
        use_mrsty_filter: bool = False,
        cui_to_tuis: Optional[Dict[str, Iterable[str]]] = None,
        anatomy_tuis: Optional[Iterable[str]] = None,
        observation_tuis: Optional[Iterable[str]] = None,
    ) -> None:
        self.mrconso_path = mrconso_path
        self.sab_preference = sab_preference or ["SNOMEDCT_US", "RXNORM", "MSH"]
        self.top_k = top_k
        self.use_mrsty_filter = use_mrsty_filter
        self.cui_to_tuis = cui_to_tuis or {}
        self.anatomy_tuis = set(anatomy_tuis or [])
        self.observation_tuis = set(observation_tuis or [])

    @staticmethod
    def normalize_text(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-z0-9\s\-]", "", s)
        return s

    @staticmethod
    def canonicalize_surface(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[-/]", " ", s)
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def is_measurement_like(self, s: str) -> bool:
        return bool(self.MEAS_RE.search(s))

    def head_candidates(self, canonical: str) -> List[str]:
        toks = [t for t in canonical.split() if t and (t not in self.STOP)]
        toks = [t for t in toks if t not in self.LATERAL and t not in self.SEVERITY]
        cands: List[str] = []
        if len(toks) >= 1:
            cands.append(toks[-1])
        if len(toks) >= 2:
            cands.append(" ".join(toks[-2:]))
        if len(toks) >= 3:
            cands.append(" ".join(toks[-3:]))
        return sorted(set(cands), key=lambda x: (-len(x.split()), x))

    def _rank_candidate(self, cand: Dict[str, Any]) -> tuple:
        ktype_rank = self.KEYTYPE_PRIORITY.get(cand.get("matched_key_type", "head"), 9)
        ispref_rank = 0 if cand.get("ispref") == "Y" else 1

        sab_rank = 999
        if self.sab_preference is not None:
            sab = cand.get("sab")
            if sab in self.sab_preference:
                sab_rank = self.sab_preference.index(sab)
        else:
            sab_rank = 0

        tty = cand.get("tty", "")
        tty_rank = 0 if tty in {"PT", "PN", "HT", "MH"} else 1

        return (
            ktype_rank,
            ispref_rank,
            sab_rank,
            tty_rank,
            cand.get("sab", ""),
            cand.get("tty", ""),
            cand.get("cui", ""),
        )

    def _apply_mrsty_filter(self, cui: str, target_type: str) -> bool:
        if not self.use_mrsty_filter:
            return True
        tuis = set(self.cui_to_tuis.get(cui, []))
        if target_type == "ANATOMY":
            return len(tuis & self.anatomy_tuis) > 0
        if target_type == "OBSERVATION":
            return len(tuis & self.observation_tuis) > 0
        return True

    def build_keys(
        self, mention_norm: str, types_seen: Iterable[str]
    ) -> Dict[str, Any]:
        m_can = self.canonicalize_surface(mention_norm)
        meas = self.is_measurement_like(m_can)

        keys = [("full_norm", mention_norm)]
        if m_can and m_can != mention_norm:
            keys.append(("full_can", m_can))
        for hc in self.head_candidates(m_can):
            keys.append(("head", hc))

        seen = set()
        keys_unique = []
        for ktype, k in keys:
            if k and k not in seen:
                keys_unique.append((ktype, k))
                seen.add(k)

        return {
            "canonical": m_can,
            "is_measurement": meas,
            "keys": keys_unique,
            "types_seen": sorted(set(types_seen)),
        }

    def map_mentions_to_cui(
        self,
        mentions: List[str],
        mention_types: Dict[str, Iterable[str]],
    ) -> Dict[str, Any]:
        """Return mention -> {best_cui, candidates, types_seen, excluded_reason}."""

        mention_keys: Dict[str, Dict[str, Any]] = {}
        key_to_mentions: defaultdict[str, List[tuple]] = defaultdict(list)

        unique_mentions = sorted(set(mentions))
        for m in unique_mentions:
            keys_info = self.build_keys(m, mention_types.get(m, []))
            mention_keys[m] = keys_info
            for ktype, k in keys_info["keys"]:
                key_to_mentions[k].append((m, ktype))

        alias_to_candidates: defaultdict[str, List[Dict[str, Any]]] = defaultdict(list)
        seen_pairs = set()
        kept_rows = 0

        with open(self.mrconso_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 15:
                    continue

                cui = parts[0]
                lat = parts[1]
                ispref = parts[6]
                sab = parts[11]
                tty = parts[12]
                s = parts[14]

                if lat != "ENG":
                    continue

                s_key = self.canonicalize_surface(s)
                if s_key not in key_to_mentions:
                    continue

                for mention_norm, ktype in key_to_mentions[s_key]:
                    if mention_keys[mention_norm]["is_measurement"]:
                        continue

                    key = (mention_norm, cui, sab, tty)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    cand = {
                        "cui": cui,
                        "sab": sab,
                        "tty": tty,
                        "ispref": ispref,
                        "str": s,
                        "matched_key": s_key,
                        "matched_key_type": ktype,
                    }
                    alias_to_candidates[mention_norm].append(cand)
                    kept_rows += 1

        mention2cui: Dict[str, Dict[str, Any]] = {}

        for m in unique_mentions:
            m_info = mention_keys[m]
            types_here = m_info["types_seen"]

            if m_info["is_measurement"]:
                mention2cui[m] = {
                    "best_cui": None,
                    "candidates": [],
                    "types_seen": types_here,
                    "excluded_reason": "measurement_like",
                }
                continue

            cands = alias_to_candidates.get(m, [])

            filtered = []
            for cand in cands:
                cui = cand["cui"]
                ok = True
                if types_here:
                    ok = any(self._apply_mrsty_filter(cui, t) for t in types_here)
                if ok:
                    filtered.append(cand)

            filtered.sort(key=self._rank_candidate)
            top = filtered[: self.top_k]
            best = top[0]["cui"] if top else None

            mention2cui[m] = {
                "best_cui": best,
                "candidates": top,
                "types_seen": types_here,
                "excluded_reason": None,
            }

        return mention2cui

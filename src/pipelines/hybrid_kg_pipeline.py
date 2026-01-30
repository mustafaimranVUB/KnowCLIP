"""End-to-end hybrid KG pipeline aligned with the tested notebook."""

from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import load_dataset

from src.knowledge.extraction import ExtractedEntity, ExtractedTriple, RadGraphExtractor
from src.knowledge.ontology_grounding import UMLSExactMatcher

# ---------- Helpers for data prep ----------


def clean_report(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("___", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def choose_best_row(group: pd.DataFrame) -> pd.Series:
    def score(row: pd.Series) -> int:
        vp = str(row.get("ViewPosition", "")).upper()
        v = str(row.get("view", "")).lower()
        s = 0
        if "PA" in vp:
            s += 3
        if "AP" in vp:
            s += 2
        if v == "frontal":
            s += 2
        if "LAT" in vp or v == "lateral":
            s -= 1
        return s

    g = group.copy()
    g["__score"] = g.apply(score, axis=1)
    return g.sort_values("__score", ascending=False).iloc[0]


# ---------- Pipeline steps ----------


def load_local_subset(
    dataset_dir: Path, split: str = "test"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds_findings = load_dataset(
        str(dataset_dir),
        name="findings_section",
        split=split,
        verification_mode="no_checks",
    )
    ds_impressions = load_dataset(
        str(dataset_dir),
        name="impression_section",
        split=split,
        verification_mode="no_checks",
    )
    return ds_findings.to_pandas(), ds_impressions.to_pandas()


def build_canonical_dataframe(df_impressions: pd.DataFrame) -> pd.DataFrame:
    df_impressions = df_impressions.copy()
    df_impressions["impression_clean"] = df_impressions["impression_section"].apply(
        clean_report
    )
    df_impressions["findings_clean"] = df_impressions["findings_section"].apply(
        clean_report
    )

    df_imp = df_impressions[df_impressions["impression_clean"].str.len() > 0].copy()

    canonical = (
        df_imp.groupby("study_id", as_index=False)
        .apply(choose_best_row, include_groups=False)
        .reset_index(drop=True)
    )
    return canonical


def run_radgraph(
    reports: List[str], model_type: str = "modern-radgraph-xl"
) -> Tuple[List[Dict[str, ExtractedEntity]], List[ExtractedTriple]]:
    extractor = RadGraphExtractor(model_type=model_type)
    preds = extractor.infer(reports)
    return extractor.normalize_batch(preds)


def collect_mentions(
    all_entities: List[Dict[str, ExtractedEntity]],
) -> Tuple[List[str], Dict[str, List[str]]]:
    mentions_raw: List[str] = []
    mention_types: Dict[str, List[str]] = {}

    for ents in all_entities:
        for ent in ents.values():
            if ent.text and ent.text.strip():
                t_norm = UMLSExactMatcher.normalize_text(ent.text)
                mentions_raw.append(t_norm)
                mention_types.setdefault(t_norm, [])
                mention_types[t_norm].append(ent.etype)

    # dedupe mention types
    mention_types = {m: sorted(set(types)) for m, types in mention_types.items()}
    return mentions_raw, mention_types


def enrich_entities_with_cui(
    all_entities: List[Dict[str, ExtractedEntity]],
    mention2cui: Dict[str, Any],
) -> List[Dict[str, Any]]:
    enriched_reports: List[Dict[str, Any]] = []
    for ents in all_entities:
        enriched = {}
        for ent_id, ent in ents.items():
            m_norm = UMLSExactMatcher.normalize_text(ent.text)
            info = mention2cui.get(m_norm, {"best_cui": None, "candidates": []})
            enriched[ent_id] = {
                "text": ent.text,
                "etype": ent.etype,
                "start": ent.start,
                "end": ent.end,
                "cui": info.get("best_cui"),
                "cui_candidates": info.get("candidates", []),
            }
        enriched_reports.append(enriched)
    return enriched_reports


def export_outputs(
    output_dir: Path,
    entities_with_cui: List[Dict[str, Any]],
    mention2cui: Dict[str, Any],
    coverage_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "entities_with_cui.pkl", "wb") as f:
        pickle.dump(entities_with_cui, f)
    with open(output_dir / "mention2cui.pkl", "wb") as f:
        pickle.dump(mention2cui, f)
    coverage_df.to_csv(output_dir / "cui_coverage.csv", index=False)


# ---------- CLI runner ----------


def run_pipeline(
    dataset_dir: Path,
    mrconso_path: Path,
    split: str = "test",
    seed: int = 42,
    model_type: str = "modern-radgraph-xl",
    output_dir: Path | None = None,
) -> None:
    output_dir = output_dir or Path("data") / "outputs" / "hybrid_kg"

    df_findings, df_impressions = load_local_subset(dataset_dir, split=split)
    canonical = build_canonical_dataframe(df_impressions)

    ds_batch = canonical.sample(n=len(canonical), random_state=seed)
    reports = [r["impression_clean"] for r in ds_batch.to_dict(orient="records")]

    all_entities, all_triples = run_radgraph(reports, model_type=model_type)

    mentions, mention_types = collect_mentions(all_entities)

    matcher = UMLSExactMatcher(str(mrconso_path))
    mention2cui = matcher.map_mentions_to_cui(mentions, mention_types)
    entities_with_cui = enrich_entities_with_cui(all_entities, mention2cui)

    coverage = []
    for m, info in mention2cui.items():
        cands = info.get("candidates", [])
        coverage.append(
            {
                "mention": m,
                "canonical": matcher.canonicalize_surface(m),
                "types_seen": ",".join(info.get("types_seen", [])),
                "is_measurement": matcher.is_measurement_like(
                    matcher.canonicalize_surface(m)
                ),
                "excluded_reason": info.get("excluded_reason"),
                "mapped": info.get("best_cui") is not None,
                "best_cui": info.get("best_cui"),
                "num_candidates": len(cands),
                "top_sab": (cands[0]["sab"] if cands else None),
                "top_tty": (cands[0]["tty"] if cands else None),
                "top_match_type": (cands[0].get("matched_key_type") if cands else None),
                "top_matched_key": (cands[0].get("matched_key") if cands else None),
            }
        )

    coverage_df = pd.DataFrame(coverage)
    export_outputs(output_dir, entities_with_cui, mention2cui, coverage_df)

    print(f"Saved outputs to {output_dir}")
    print("Triples extracted:", len(all_triples))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Hybrid KG pipeline to mirror the notebook."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to local HF dataset directory (e.g., data/MIMIC-CXR-RRG_small)",
    )
    parser.add_argument(
        "--mrconso", type=Path, required=True, help="Path to MRCONSO.RRF"
    )
    parser.add_argument("--split", default="test", help="Dataset split name")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--model-type", default="modern-radgraph-xl", help="RadGraph model type"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write outputs (pickles + CSV)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_pipeline(
        dataset_dir=args.dataset_dir,
        mrconso_path=args.mrconso,
        split=args.split,
        seed=args.seed,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

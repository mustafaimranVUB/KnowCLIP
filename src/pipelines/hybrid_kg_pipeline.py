"""Phase I — Hybrid Knowledge Graph Construction Pipeline.

Orchestrates:
1. Report loading
2. Entity / relation extraction (RadGraph-XL)
3. UMLS exact-match grounding (MRCONSO + MRSTY + MRREL)
4. Node embedding (SapBERT)
5. PyG graph construction & validation
6. Artifact persistence (``torch.save``)

All Phase I artifacts are stored under ``config.kg_pipeline.output_dir``
as ``.pt`` tensors (never pickle).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.core.config import DataConfig, KGPipelineConfig, ProjectConfig
from src.data.report_loader import ReportLoader
from src.knowledge.extraction import (
    EntityExtractor,
    EntityType,
    ExtractedEntity,
    ExtractedTriple,
)
from src.knowledge.graph_builder import KnowledgeGraphBuilder, validate_graph
from src.knowledge.ontology_grounding import OntologyGrounder

logger = logging.getLogger(__name__)


class HybridKGPipeline:
    """End-to-end Phase I knowledge graph construction.

    Parameters:
        config: Full project configuration (only ``data`` and
            ``kg_pipeline`` sub-configs are used).
        device: Torch device for RadGraph and embedding models.
    """

    def __init__(
        self,
        config: ProjectConfig,
        device: Optional[str] = None,
    ) -> None:
        self.data_cfg: DataConfig = config.data
        self.kg_cfg: KGPipelineConfig = config.kg_pipeline
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-initialised components
        self._extractor: Optional[EntityExtractor] = None
        self._grounder: Optional[OntologyGrounder] = None
        self._embedder: Optional[Any] = None  # NodeEmbedder
        self._report_loader: Optional[ReportLoader] = None

    # ------------------------------------------------------------------
    # Component accessors (lazy init)
    # ------------------------------------------------------------------

    @property
    def extractor(self) -> EntityExtractor:
        if self._extractor is None:
            self._extractor = EntityExtractor(
                model_type=self.kg_cfg.radgraph_model_type,
                device=self.device,
            )
        return self._extractor

    @property
    def grounder(self) -> OntologyGrounder:
        if self._grounder is None:
            # ---- Stage-2 scispaCy fallback (optional) ----------------
            scispacy_grounder = None
            if self.kg_cfg.use_scispacy_fallback:
                from src.knowledge.scispacy_grounding import (
                    ScispaCyGrounder,
                )

                cache_dir = (
                    self.kg_cfg.scispacy_cache_dir
                    if self.kg_cfg.scispacy_cache_dir
                    and str(self.kg_cfg.scispacy_cache_dir) != "."
                    and self.kg_cfg.scispacy_cache_dir != Path(".")
                    else None
                )
                if ScispaCyGrounder.is_available(cache_dir):
                    logger.info(
                        "Initialising scispaCy Stage-2 fallback grounder "
                        "(cache=%s, threshold=%.2f)",
                        cache_dir or "<default>",
                        self.kg_cfg.scispacy_confidence_threshold,
                    )
                    scispacy_grounder = ScispaCyGrounder(
                        cache_dir=cache_dir,
                        confidence_threshold=self.kg_cfg.scispacy_confidence_threshold,
                    )
                else:
                    logger.warning(
                        "use_scispacy_fallback=True but en_core_sci_lg is not "
                        "available (cache=%s). Proceeding with MRCONSO only. "
                        "To install: pip install "
                        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                        "releases/v0.5.5/en_core_sci_lg-0.5.5.tar.gz "
                        "--target $VSC_SCRATCH/scispacy_cache",
                        cache_dir or "<default>",
                    )

            self._grounder = OntologyGrounder(
                mrconso_path=self.kg_cfg.mrconso_path,
                mrsty_path=self.kg_cfg.mrsty_path,
                mrrel_path=self.kg_cfg.mrrel_path,
                top_k=self.kg_cfg.top_k_candidates,
                validate_semantic_types=True,
                scispacy_grounder=scispacy_grounder,
            )
            logger.info("Loading UMLS indices …")
            self._grounder.load_indices()
            logger.info("UMLS indices loaded.")
        return self._grounder

    @property
    def embedder(self):
        if self._embedder is None:
            from src.knowledge.embeddings import NodeEmbedder

            self._embedder = NodeEmbedder(device=self.device)
        return self._embedder

    @property
    def report_loader(self) -> ReportLoader:
        if self._report_loader is None:
            self._report_loader = ReportLoader(
                reports_root=self.data_cfg.reports_root,
                section_preference=self.data_cfg.report_section_preference,
            )
        return self._report_loader

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        study_list: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32,
        skip_extraction: bool = False,
        skip_grounding: bool = False,
        skip_embedding: bool = False,
        max_studies: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the full Phase I pipeline.

        Args:
            study_list: List of dicts with keys ``subject_id`` and
                ``study_id``.  If *None*, the pipeline auto-discovers
                studies by scanning ``data_cfg.reports_root`` (or reads
                ``cxr-record-list.csv`` if present).
            batch_size: Batch size for RadGraph extraction.
            skip_extraction: If True, load extraction artifacts from
                disk instead of re-running RadGraph.
            skip_grounding: If True, load grounding artifacts from disk.
            skip_embedding: If True, load embedding artifacts from disk.
            max_studies: Optional cap on the number of studies to process.
                Useful for development / smoke-test runs.

        Returns:
            Summary dict with coverage stats, graph validation, and
            artifact paths.
        """
        out_dir = Path(self.kg_cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        summary: Dict[str, Any] = {"output_dir": str(out_dir)}

        # ----- 1. Gather study list ---------------------------------
        if study_list is None:
            study_list = self._load_study_list(max_studies=max_studies)
        elif max_studies is not None:
            study_list = study_list[:max_studies]
        logger.info("Phase I: %d studies to process.", len(study_list))
        summary["num_studies"] = len(study_list)

        # ----- 2. Load reports --------------------------------------
        reports, valid_studies = self._load_reports(study_list)
        logger.info("Loaded %d reports (of %d studies).", len(reports), len(study_list))
        summary["num_reports_loaded"] = len(reports)

        # ----- 3. Entity extraction ---------------------------------
        extraction_path = out_dir / "extractions.pt"
        if skip_extraction and extraction_path.exists():
            logger.info("Loading cached extractions from %s", extraction_path)
            extraction_data = torch.load(extraction_path, weights_only=False)
            all_entities = extraction_data["entities"]
            all_triples = extraction_data["triples"]
        else:
            all_entities, all_triples = self._extract_all(reports, batch_size)
            torch.save(
                {"entities": all_entities, "triples": all_triples},
                extraction_path,
            )
            logger.info("Saved extractions to %s", extraction_path)

        total_entities = sum(len(e) for e in all_entities)
        total_triples = sum(len(t) for t in all_triples)
        summary["total_entities"] = total_entities
        summary["total_triples"] = total_triples
        logger.info(
            "Extracted %d entities and %d triples from %d reports.",
            total_entities,
            total_triples,
            len(reports),
        )

        # ----- 4. Filter eligible entities --------------------------
        eligible_entities: List[List[ExtractedEntity]] = []
        for ents in all_entities:
            eligible_entities.append(EntityExtractor.filter_eligible(ents))

        total_eligible = sum(len(e) for e in eligible_entities)
        summary["total_eligible_entities"] = total_eligible

        # ----- 5. Ontology grounding --------------------------------
        grounding_path = out_dir / "grounding.pt"
        if skip_grounding and grounding_path.exists():
            logger.info("Loading cached grounding from %s", grounding_path)
            grounding_data = torch.load(grounding_path, weights_only=False)
            all_groundings = grounding_data["groundings"]
            mention2cui = grounding_data["mention2cui"]
        else:
            all_groundings, mention2cui = self._ground_all(eligible_entities)
            torch.save(
                {"groundings": all_groundings, "mention2cui": mention2cui},
                grounding_path,
            )
            logger.info("Saved grounding to %s", grounding_path)

        # Coverage
        from src.knowledge.ontology_grounding import OntologyGrounder

        flat_groundings = [g for gs in all_groundings for g in gs]
        coverage = OntologyGrounder.compute_coverage(flat_groundings)
        summary["grounding_coverage"] = coverage
        logger.info(
            "Grounding coverage: %.1f%% (%d/%d eligible mapped).",
            coverage.get("coverage_pct", 0),
            coverage.get("mapped", 0),
            coverage.get("total", 0),
        )

        # Save coverage CSV
        self._save_coverage_csv(coverage, mention2cui, out_dir)

        # ----- 6. Compute node embeddings ---------------------------
        embedding_path = out_dir / "embeddings.pt"
        if skip_embedding and embedding_path.exists():
            logger.info("Loading cached embeddings from %s", embedding_path)
            embeddings = torch.load(embedding_path, weights_only=False)
        else:
            embeddings = self._compute_embeddings(all_entities)
            torch.save(embeddings, embedding_path)
            logger.info("Saved %d embeddings to %s", len(embeddings), embedding_path)

        # ----- 7. Build global knowledge graph ----------------------
        builder = KnowledgeGraphBuilder(
            node_dim=768,
            add_self_loops=True,
            bidirectional=False,
        )

        global_graph = builder.build_global_graph(
            all_entities=all_entities,
            all_triples=all_triples,
            all_groundings=all_groundings,
            embeddings=embeddings,
            mrrel_index=self.grounder.mrrel_index,
        )

        graph_path = out_dir / "global_kg.pt"
        torch.save(global_graph, graph_path)
        logger.info("Saved global KG to %s", graph_path)

        # ----- 8. Validate graph ------------------------------------
        validation = validate_graph(global_graph)
        summary["graph_validation"] = validation
        logger.info("Graph validation: %s", validation)

        # ----- 9. Build per-report graphs (optional) ----------------
        report_graphs_path = out_dir / "report_graphs.pt"
        report_graphs: Dict[str, Any] = {}
        for i, study in enumerate(valid_studies):
            study_key = f"{study['subject_id']}_{study['study_id']}"
            rg = builder.build_from_extraction(
                entities=all_entities[i],
                triples=all_triples[i],
                grounding_results=all_groundings[i] if i < len(all_groundings) else None,
                embeddings=embeddings,
                mrrel_index=self.grounder.mrrel_index,
            )
            report_graphs[study_key] = rg

        torch.save(report_graphs, report_graphs_path)
        logger.info(
            "Saved %d per-report graphs to %s",
            len(report_graphs),
            report_graphs_path,
        )

        # ----- 10. Save study-level metadata -----------------------
        meta_path = out_dir / "study_metadata.pt"
        torch.save(
            {
                "valid_studies": valid_studies,
                "mention2cui": mention2cui,
            },
            meta_path,
        )

        elapsed = time.time() - t0
        summary["elapsed_seconds"] = elapsed
        summary["artifact_paths"] = {
            "extractions": str(extraction_path),
            "grounding": str(grounding_path),
            "embeddings": str(embedding_path),
            "global_kg": str(graph_path),
            "report_graphs": str(report_graphs_path),
            "study_metadata": str(meta_path),
        }

        # Save summary JSON
        summary_path = out_dir / "phase1_summary.json"
        with open(summary_path, "w") as fp:
            json.dump(summary, fp, indent=2, default=str)

        logger.info(
            "Phase I complete in %.1f s. Artifacts in %s", elapsed, out_dir
        )
        return summary

    # ------------------------------------------------------------------
    # Internal stage methods
    # ------------------------------------------------------------------

    def _load_study_list(
        self, max_studies: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return unique (subject_id, study_id) pairs.

        Priority:
        1. Scan ``data_cfg.reports_root`` directory tree directly.
        2. Fall back to reading ``cxr-record-list.csv`` under
           ``data_cfg.mimic_root`` (legacy behaviour).
        """
        # --- Primary: scan reports directory -----------------------
        reports_root = self.data_cfg.reports_root
        if reports_root.exists() and reports_root.is_dir():
            logger.info(
                "Scanning reports directory: %s", reports_root
            )
            studies = self.report_loader.scan_all_studies(
                max_studies=max_studies
            )
            if studies:
                return studies
            logger.warning(
                "scan_all_studies returned 0 studies from %s; "
                "falling back to CSV.",
                reports_root,
            )

        # --- Fallback: cxr-record-list.csv -------------------------
        csv_path = self.data_cfg.mimic_root / "cxr-record-list.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No studies found. reports_root={reports_root} is empty or "
                f"missing, and cxr-record-list.csv not found at {csv_path}. "
                "Set data.reports_root or data.mimic_root in config."
            )

        seen: set = set()
        studies = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row["subject_id"])
                stid = int(row["study_id"])
                key = (sid, stid)
                if key not in seen:
                    seen.add(key)
                    studies.append({"subject_id": sid, "study_id": stid})
                    if max_studies is not None and len(studies) >= max_studies:
                        break

        logger.info("Loaded %d unique studies from %s", len(studies), csv_path)
        return studies

    def _load_reports(
        self,
        study_list: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Load report texts, skipping studies with missing/empty reports."""
        reports: List[str] = []
        valid_studies: List[Dict[str, Any]] = []

        for study in study_list:
            text = self.report_loader.load_report(
                subject_id=study["subject_id"],
                study_id=study["study_id"],
            )
            if text:
                reports.append(text)
                valid_studies.append(study)

        return reports, valid_studies

    def _extract_all(
        self,
        reports: List[str],
        batch_size: int,
    ) -> Tuple[List[List[ExtractedEntity]], List[List[ExtractedTriple]]]:
        """Run RadGraph extraction on all reports in batches."""
        all_entities: List[List[ExtractedEntity]] = []
        all_triples: List[List[ExtractedTriple]] = []

        num_batches = (len(reports) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(reports))
            batch = reports[start:end]

            logger.info(
                "Extracting batch %d/%d (%d reports) …",
                batch_idx + 1,
                num_batches,
                len(batch),
            )
            results = self.extractor.extract_batch(batch)

            for entities, triples in results:
                all_entities.append(entities)
                all_triples.append(triples)

        return all_entities, all_triples

    def _ground_all(
        self,
        eligible_entities: List[List[ExtractedEntity]],
    ) -> Tuple[List[list], Dict[str, Dict[str, Any]]]:
        """Ground all eligible entities to UMLS CUIs."""
        all_groundings: List[list] = []
        global_mention2cui: Dict[str, Dict[str, Any]] = {}

        for i, entities in enumerate(eligible_entities):
            if not entities:
                all_groundings.append([])
                continue

            results = self.grounder.ground_entities(entities)
            all_groundings.append(results)

            # Merge into global mention→CUI map
            for res in results:
                if res.mapped and res.normalized not in global_mention2cui:
                    global_mention2cui[res.normalized] = {
                        "best_cui": res.best_cui,
                        "best_name": res.best_name,
                        "mapped": res.mapped,
                        "tuis": list(res.tuis) if res.tuis else [],
                    }

            if (i + 1) % 1000 == 0:
                logger.info("Grounded %d/%d reports.", i + 1, len(eligible_entities))

        return all_groundings, global_mention2cui

    def _compute_embeddings(
        self,
        all_entities: List[List[ExtractedEntity]],
    ) -> Dict[str, torch.Tensor]:
        """Compute SapBERT embeddings for all unique entity surface forms."""
        from src.knowledge.extraction import normalize_text

        # Collect unique normalised texts
        unique_texts: Dict[str, str] = {}
        for entities in all_entities:
            for ent in entities:
                norm = normalize_text(ent.text)
                if norm and norm not in unique_texts:
                    unique_texts[norm] = ent.text

        if not unique_texts:
            return {}

        texts = list(unique_texts.keys())
        logger.info("Computing embeddings for %d unique entity texts …", len(texts))
        # embed_texts returns Dict[str, Tensor] keyed by the input texts
        embeddings = self.embedder.embed_texts(texts)
        return embeddings

    def _save_coverage_csv(
        self,
        coverage: Dict[str, Any],
        mention2cui: Dict[str, Dict[str, Any]],
        out_dir: Path,
    ) -> None:
        """Write grounding coverage analysis to CSV."""
        csv_path = out_dir / "cui_coverage.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "mention",
                "best_cui",
                "best_name",
                "mapped",
            ])
            for mention, info in sorted(mention2cui.items()):
                writer.writerow([
                    mention,
                    info.get("best_cui", ""),
                    info.get("best_name", ""),
                    info.get("mapped", False),
                ])

        logger.info("Wrote coverage CSV to %s (%d entries)", csv_path, len(mention2cui))

        # Also save a summary row
        summary_csv = out_dir / "grounding_summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in coverage.items():
                writer.writerow([k, v])

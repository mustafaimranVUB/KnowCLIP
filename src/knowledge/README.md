# Phase I — Knowledge Graph Construction

This package builds per-study and global knowledge graphs from radiology report
text. It is invoked by the CLI entry point `python -m main phase1`.

## Entry point

```bash
# Local smoke test (CPU, subset of reports)
python -m main --config configs/phase1_kg_local.yaml phase1 --max-studies 100

# Hydra full dataset run (GPU node, ~20 000 s)
sbatch scripts/hydra/run_phase1.sh
```

## Stage order

The pipeline executes four stages in sequence. Each stage writes an intermediate
artifact so a failed run can resume from the last completed stage.

```
extraction.py           RadGraph-XL entity / relation extraction
        ↓  extractions.pt
ontology_grounding.py   UMLS 2025AB CUI lookup + SapBERT re-ranking
        ↓  (+ scispacy_grounding.py fallback for unmatched mentions)
                        grounding.pt, study_metadata.pt, cui_coverage.csv
embeddings.py           SapBERT node embeddings  →  embeddings.pt
        ↓
graph_builder.py        PyG Data objects         →  report_graphs.pt, global_kg.pt
```

### Stage 1 — `extraction.py` (`EntityExtractor`)

Wraps `radgraph` (StanfordAIMI/modern-radgraph-xl) to extract typed entities
and relations from free-text impressions / findings sections.

Entity types: `anatomy`, `observation`, `measurement`
Certainty labels: `definitely_present`, `uncertain`, `definitely_absent`
Relation types: `located_at`, `modify`, `suggestive_of`, `associated_with`

### Stage 2 — `ontology_grounding.py` + `scispacy_grounding.py`

Maps extracted mentions to UMLS 2025AB CUIs. Primary path: UMLS REST API +
SapBERT cosine re-ranking. Fallback path (`scispacy_grounding.py`): scispaCy
`en_core_sci_lg` pipeline for mentions that the primary stage could not ground.

Writes `study_metadata.pt` (per-study mention→CUI lookup table) and
`cui_coverage.csv` (grounding coverage statistics per CUI).

### Stage 3 — `embeddings.py`

Encodes each unique grounded concept using SapBERT
(`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) to produce a fixed-size
node embedding matrix stored in `embeddings.pt`.

### Stage 4 — `graph_builder.py` (`KnowledgeGraphBuilder`)

Assembles per-study PyG `Data` objects from the grounded entities and relations,
accumulates them into the global KG, and writes all output artifacts.

## Output artifacts (`outputs/KG/`)

| File | Content |
|------|---------|
| `extractions.pt` | Raw RadGraph-XL extraction results per study |
| `grounding.pt` | UMLS CUI assignments per extraction |
| `embeddings.pt` | SapBERT node embedding matrix |
| `report_graphs.pt` | Per-study PyG `Data` objects (17 GB; one entry per study) |
| `global_kg.pt` | Merged global knowledge graph (PyG `Data`) |
| `study_metadata.pt` | Per-study mention → CUI metadata |
| `cui_coverage.csv` | Grounding coverage statistics |
| `phase1_summary.json` | Run summary (counts, timing, coverage rate) |

## Validated run statistics (MIMIC-CXR impression build)

- Studies processed: **227,835**
- Grounding coverage: **94.1 %**
- Global KG nodes: **14,587** (13,202 in the giant component)
- Global KG unique weighted edges: **141,991**

## Module reference

| Module | Class / function | Role |
|--------|-----------------|------|
| `extraction.py` | `EntityExtractor` | Stage 1 |
| `ontology_grounding.py` | `OntologyGrounder` | Stage 2 primary |
| `scispacy_grounding.py` | `ScispaCyGrounder` | Stage 2 fallback |
| `embeddings.py` | (module-level functions) | Stage 3 |
| `graph_builder.py` | `KnowledgeGraphBuilder` | Stage 4 |

All components are exported from `src/knowledge/__init__.py`.

## Configuration

Phase I behavior is controlled by a YAML config loaded via
`src/core/config.py:KGPipelineConfig`. Key fields:

- `report_section_preference`: `impression` (production) or `findings` (ablation)
- `max_studies`: cap for development runs
- `grounding_backend`: `umls` | `scispacy`
- `embedding_model`: SapBERT model ID

The production impression config is `configs/phase1_kg_gpu.yaml`.
The local development config is `configs/phase1_kg_local.yaml`.

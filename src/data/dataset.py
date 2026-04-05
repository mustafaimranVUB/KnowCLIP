"""Production MIMIC-CXR dataset for DICOM-based training.

Works directly with PhysioNet DICOM files (no HuggingFace proxy).
Loads images, reports, CheXpert labels, and per-study KG sub-graphs.
"""



from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset

from src.data.dicom_loader import DICOMLoader
from src.data.report_loader import ReportLoader
from src.data.splits import Split, SplitManager

logger = logging.getLogger(__name__)

# Module-level tokenizer (lazily initialised, monkey-patchable in tests).
_REPORT_TOKENIZER: Any = None


def _get_report_tokenizer() -> Any:
    """Return (and cache) a GPT-2 tokenizer for report targets."""
    global _REPORT_TOKENIZER
    if _REPORT_TOKENIZER is None:
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        _REPORT_TOKENIZER = tok
    return _REPORT_TOKENIZER


class MIMICCXRDataset(Dataset):
    """MIMIC-CXR dataset reading DICOM images from disk.

    Accepts the same kwargs used by ``TrainingPipeline.build_dataloaders``
    so that a ``SplitManager`` is constructed internally.

    Parameters:
        split: Dataset split (``'train'``, ``'validate'``, ``'test'``).
        mimic_root: Root directory for DICOM files.
        reports_root: Root directory for report text files.
        split_csv: Path to ``mimic-cxr-2.0.0-split.csv``.
        chexpert_csv: Path to ``mimic-cxr-2.0.0-chexpert.csv``.
        kg_artifacts_dir: Optional directory with Phase I KG artifacts.
        transforms: Optional image transform pipeline.
        dev_subset_frac: Fraction of subjects for dev subset (None = full).
        include_graphs: Load per-study KG sub-graphs from
            ``report_graphs.pt``.
        split_strategy: ``'official'``, ``'subset_hash'``, or ``'auto'``.
        subset_seed: Seed for deterministic hash splitting.
        subset_train_ratio: Train fraction for hash split.
        subset_val_ratio: Validation fraction for hash split.
        subset_test_ratio: Test fraction for hash split.
        auto_min_val_samples: Min val samples before falling back to hash
            split (``auto`` strategy).
        auto_min_test_samples: Min test samples before falling back to
            hash split (``auto`` strategy).
        image_size: Spatial resolution (used only when *transforms* is
            ``None``).
    """

    def __init__(
        self,
        split: Split,
        mimic_root: Path | str,
        reports_root: Path | str,
        split_csv: Path | str,
        chexpert_csv: Path | str,
        kg_artifacts_dir: Optional[Path | str] = None,
        transforms: Optional[Callable] = None,
        dev_subset_frac: Optional[float] = None,
        include_graphs: bool = False,
        split_strategy: Literal["official", "subset_hash", "auto"] = "official",
        subset_seed: int = 42,
        subset_train_ratio: float = 0.8,
        subset_val_ratio: float = 0.1,
        subset_test_ratio: float = 0.1,
        auto_min_val_samples: int = 100,
        auto_min_test_samples: int = 100,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.split = split
        self.mimic_root = Path(mimic_root)
        self.reports_root = Path(reports_root)
        self.image_size = image_size
        self.kg_artifacts_dir = Path(kg_artifacts_dir) if kg_artifacts_dir else None
        self.include_graphs = include_graphs
        self.transform = transforms

        # Cache of DICOM paths that failed to load (avoid repeated warnings).
        self._failed_dicoms: set[str] = set()

        # Build SplitManager internally
        self.split_manager = SplitManager(
            split_csv=split_csv,
            chexpert_csv=chexpert_csv,
        )
        self.dicom_loader = DICOMLoader(image_size=image_size)
        self.report_loader = ReportLoader(reports_root=self.reports_root)

        # Split strategy
        self._split_strategy = split_strategy
        self._subset_seed = subset_seed
        self._subset_train_ratio = subset_train_ratio
        self._subset_val_ratio = subset_val_ratio
        self._subset_test_ratio = subset_test_ratio
        self._auto_min_val = auto_min_val_samples
        self._auto_min_test = auto_min_test_samples

        # Build sample index
        self._build_index(dev_subset_frac=dev_subset_frac, seed=subset_seed)

        # Load KG report graphs if requested
        self._report_graphs: Optional[Dict[str, Any]] = None
        if self.include_graphs:
            self._load_report_graphs()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(
        self,
        dev_subset_frac: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        """Build the sample index, applying split_strategy.

        Each sample maps to one (subject_id, study_id) pair with a
        canonical DICOM image selected by view priority.
        """
        split_df = self.split_manager.splits_df

        strategy = self._split_strategy

        # ``auto``: use official splits unless val/test are too small.
        if strategy == "auto":
            n_val = len(split_df[split_df["split"] == "validate"])
            n_test = len(split_df[split_df["split"] == "test"])
            if n_val < self._auto_min_val or n_test < self._auto_min_test:
                strategy = "subset_hash"
                logger.info(
                    "Auto split: val=%d (min=%d), test=%d (min=%d) → falling back to subset_hash.",
                    n_val, self._auto_min_val, n_test, self._auto_min_test,
                )
            else:
                strategy = "official"

        if strategy == "subset_hash":
            records = self._hash_split(split_df)
        else:
            records = split_df[split_df["split"] == self.split].copy()

        if records.empty:
            logger.warning("No records for split '%s'", self.split)
            self.samples: List[Dict[str, Any]] = []
            return

        # Optional dev subset (sample subjects, not individual records)
        if dev_subset_frac is not None and 0 < dev_subset_frac < 1.0:
            subjects = records["subject_id"].unique()
            n = max(1, int(len(subjects) * dev_subset_frac))
            rng = np.random.default_rng(seed)
            subset = rng.choice(subjects, size=n, replace=False)
            records = records[records["subject_id"].isin(subset)]
            logger.info(
                "Dev subset for '%s': %d subjects, %d records",
                self.split, n, len(records),
            )

        # Load CheXpert labels into a lookup table
        chexpert_lookup: Dict[Tuple[int, int], torch.Tensor] = {}
        try:
            chexpert_df = self.split_manager.chexpert_df
            if chexpert_df is not None:
                label_cols = SplitManager.CHEXPERT_LABELS
                for _, row in chexpert_df.iterrows():
                    sid = int(row["subject_id"])
                    stid = int(row["study_id"])
                    vals = []
                    for col in label_cols:
                        v = row.get(col, float("nan"))
                        if pd.isna(v):
                            vals.append(float("nan"))
                        else:
                            v = float(v)
                            # Map -1 (uncertain) → NaN
                            vals.append(float("nan") if v == -1.0 else v)
                    chexpert_lookup[(sid, stid)] = torch.tensor(vals, dtype=torch.float32)
        except Exception:
            pass  # CheXpert CSV not available

        # Group by study and select canonical image
        self.samples = []
        skipped_missing = 0
        for (subj, study), grp in records.groupby(["subject_id", "study_id"]):
            view_records: List[Tuple[str, Optional[str]]] = []
            dicom_paths: Dict[str, Path] = {}

            for _, row in grp.iterrows():
                did = str(row["dicom_id"])
                if "path" in row.index and pd.notna(row["path"]):
                    dcm_path = self.mimic_root / str(row["path"])
                    if not dcm_path.exists():
                        # Strip leading 'files/' if the layout omits it
                        raw = str(row["path"])
                        if raw.startswith("files/"):
                            dcm_path = self.mimic_root / raw[len("files/"):]
                else:
                    prefix = f"p{str(subj)[:2]}"
                    rel = Path(prefix) / f"p{subj}" / f"s{study}" / f"{did}.dcm"
                    dcm_path = self.mimic_root / "files" / rel
                    if not dcm_path.exists():
                        # Also check without the 'files/' subdirectory
                        dcm_path = self.mimic_root / rel
                # Only include DICOMs that actually exist on disk
                if not dcm_path.exists():
                    continue
                dicom_paths[did] = dcm_path
                vp = row.get("ViewPosition") if "ViewPosition" in row.index else None
                view_records.append((did, vp))

            if not view_records:
                skipped_missing += 1
                continue

            canonical = ReportLoader.select_canonical_image(view_records)
            if canonical is None:
                continue

            labels = chexpert_lookup.get(
                (int(subj), int(study)),
                torch.full((14,), float("nan")),
            )

            self.samples.append({
                "subject_id": int(subj),
                "study_id": int(study),
                "dicom_id": canonical,
                "dicom_path": dicom_paths[canonical],
                "labels": labels,
                "study_key": f"{int(subj)}_{int(study)}",
            })

        if skipped_missing > 0:
            logger.info(
                "Skipped %d studies with no DICOM files on disk.", skipped_missing,
            )
        logger.info(
            "Built index for '%s': %d studies from %d records (strategy=%s)",
            self.split, len(self.samples), len(records), strategy,
        )

    def _hash_split(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic hash-based split that guarantees class coverage.

        First ensures at least one positive sample per class per split
        (val/test), then assigns remaining subjects via hash.
        """
        split_map = {"train": 0, "validate": 1, "test": 2}
        target = split_map.get(self.split)
        if target is None:
            return split_df.iloc[0:0]

        subjects = sorted(split_df["subject_id"].unique())
        train_end = self._subset_train_ratio
        val_end = train_end + self._subset_val_ratio

        # Build per-subject label vectors from CheXpert for coverage enforcement
        subj_labels: Dict[int, torch.Tensor] = {}
        try:
            chex_df = self.split_manager.chexpert_df
            label_cols = SplitManager.CHEXPERT_LABELS
            for _, row in chex_df.iterrows():
                sid = int(row["subject_id"])
                if sid in subj_labels:
                    continue  # one vector per subject is enough
                vals = [max(0.0, float(row.get(c, 0) or 0)) for c in label_cols]
                subj_labels[sid] = torch.tensor(vals)
        except Exception:
            pass

        # Phase 1: guarantee ≥1 positive per class in val and test.
        val_reserved: set = set()
        test_reserved: set = set()
        n_classes = 14
        rng = np.random.default_rng(self._subset_seed)

        if subj_labels:
            # For each class, shuffle subjects with that class positive, assign
            # the first unseen to val and the second to test.
            for c in range(n_classes):
                positives = [s for s in subjects if s in subj_labels and subj_labels[s][c] > 0.5]
                rng.shuffle(positives)
                for s in positives:
                    if s not in val_reserved and s not in test_reserved:
                        val_reserved.add(s)
                        break
                for s in positives:
                    if s not in val_reserved and s not in test_reserved:
                        test_reserved.add(s)
                        break

        # Phase 2: hash remaining subjects into buckets.
        assigned: Dict[int, int] = {}
        for s in val_reserved:
            assigned[s] = 1
        for s in test_reserved:
            assigned[s] = 2

        for s in subjects:
            if s in assigned:
                continue
            h = int(hashlib.md5(f"{s}_{self._subset_seed}".encode()).hexdigest(), 16) % 10000
            frac = h / 10000.0
            if frac < train_end:
                assigned[s] = 0
            elif frac < val_end:
                assigned[s] = 1
            else:
                assigned[s] = 2

        selected = {s for s, b in assigned.items() if b == target}
        return split_df[split_df["subject_id"].isin(selected)].copy()

    # ------------------------------------------------------------------
    # KG artifacts
    # ------------------------------------------------------------------

    def _load_report_graphs(self) -> None:
        """Load per-study PyG graphs from ``report_graphs.pt``."""
        if self.kg_artifacts_dir is None:
            return
        rg_path = self.kg_artifacts_dir / "report_graphs.pt"
        if rg_path.exists():
            try:
                self._report_graphs = torch.load(
                    str(rg_path), map_location="cpu", weights_only=False,
                )
                logger.info(
                    "Loaded %d report graphs from %s",
                    len(self._report_graphs), rg_path,
                )
            except Exception as exc:
                logger.warning("Failed to load report graphs: %s", exc)

    def get_graph(self, study_key: str) -> Optional[Any]:
        """Retrieve a per-study PyG ``Data`` graph."""
        if self._report_graphs is None:
            return None
        return self._report_graphs.get(study_key)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        subject_id = sample["subject_id"]
        study_id = sample["study_id"]
        study_key = sample["study_key"]
        dicom_path = sample["dicom_path"]

        # ----- Image -----
        dicom_key = str(dicom_path)
        if dicom_key in self._failed_dicoms:
            image = None
        else:
            image = self.dicom_loader.load_dicom(dicom_path)
            if image is None:
                self._failed_dicoms.add(dicom_key)

        if image is None:
            image = PILImage.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
            if dicom_key not in self._failed_dicoms:
                logger.warning("Returning zero image for study %s", study_id)

        if self.transform is not None:
            image = self.transform(image)

        # ----- Report -----
        report_text = self.report_loader.load_report(subject_id, study_id)
        if report_text is None:
            report_text = ""

        # ----- Labels -----
        label_tensor = sample["labels"]

        # ----- Per-study graph -----
        graph = self.get_graph(study_key) if self.include_graphs else None

        out: Dict[str, Any] = {
            "image": image,
            "report_text": report_text,
            "labels": label_tensor,
            "subject_id": subject_id,
            "study_id": study_id,
            "study_key": study_key,
        }

        if graph is not None:
            out["graph_x"] = graph.x  # (N, D)
            out["graph_edge_index"] = graph.edge_index  # (2, E)
            out["graph_edge_type"] = graph.edge_type  # (E,)

        return out


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_mimic(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Custom collate that batches images, labels, graphs, and report tokens.

    Returns a dictionary understood by :class:`Trainer`:

    - ``image``  – ``(B, 3, H, W)``
    - ``labels`` – ``(B, 14)``
    - ``graph_x`` / ``graph_edge_index`` / ``graph_edge_type`` /
      ``graph_batch`` – PyG-style batched graph tensors (or ``None``)
    - ``target_ids`` / ``generation_targets`` – tokenised report targets
      ``(B, L-1)`` (or ``None`` if no tokenizer)
    - ``report_text`` – list of raw report strings
    """
    images = torch.stack([s["image"] for s in batch])
    labels = torch.stack([s["labels"] for s in batch])

    collated: Dict[str, Any] = {
        "image": images,
        "labels": labels,
        "report_text": [s.get("report_text", "") for s in batch],
        "subject_id": [s["subject_id"] for s in batch],
        "study_id": [s["study_id"] for s in batch],
        "study_key": [s.get("study_key", "") for s in batch],
    }

    # ----- Graph batching (PyG-style) -----
    has_graph = any("graph_x" in s for s in batch)
    if has_graph:
        xs, edges, etypes, batch_vec = [], [], [], []
        offset = 0
        for i, s in enumerate(batch):
            gx = s.get("graph_x")
            if gx is not None and gx.shape[0] > 0:
                n = gx.shape[0]
                xs.append(gx)
                edges.append(s["graph_edge_index"] + offset)
                etypes.append(s["graph_edge_type"])
                batch_vec.append(torch.full((n,), i, dtype=torch.long))
                offset += n
            else:
                # Placeholder: single zero node for samples without a graph
                xs.append(torch.zeros(1, batch[0].get("graph_x", torch.zeros(1, 768)).shape[-1]))
                edges.append(torch.zeros(2, 0, dtype=torch.long) + offset)
                etypes.append(torch.zeros(0, dtype=torch.long))
                batch_vec.append(torch.full((1,), i, dtype=torch.long))
                offset += 1

        collated["graph_x"] = torch.cat(xs, dim=0)
        collated["graph_edge_index"] = torch.cat(edges, dim=1)
        collated["graph_edge_type"] = torch.cat(etypes, dim=0)
        collated["graph_batch"] = torch.cat(batch_vec, dim=0)

    # ----- Report tokenization -----
    tokenizer = _REPORT_TOKENIZER
    if tokenizer is None:
        try:
            tokenizer = _get_report_tokenizer()
        except Exception:
            pass  # transformers not available or offline

    if tokenizer is not None:
        texts = collated["report_text"]

        # Append EOS so the model learns to predict end-of-sequence.
        eos_str = getattr(tokenizer, "eos_token", None) or ""
        if eos_str:
            texts = [t + eos_str for t in texts]

        # Tokenize with max_length=127 to leave room for prepended BOS.
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=127,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]  # (B, 127)
        attn_mask = enc["attention_mask"]  # (B, 127)

        # Prepend BOS token so training matches auto-regressive inference,
        # which also starts with this token.  GPT-2 uses eos_token_id
        # (50256) as both BOS and EOS.
        bos_id = getattr(tokenizer, "eos_token_id", None)
        if bos_id is None:
            bos_id = 50256  # GPT-2 default
        B = input_ids.shape[0]
        bos_col = torch.full((B, 1), bos_id, dtype=input_ids.dtype)
        bos_attn = torch.ones((B, 1), dtype=attn_mask.dtype)
        input_ids = torch.cat([bos_col, input_ids], dim=1)  # (B, 128)
        attn_mask = torch.cat([bos_attn, attn_mask], dim=1)  # (B, 128)

        # Teacher-forced input: all but last token
        target_ids = input_ids[:, :-1]  # (B, 127)

        # Generation targets: shifted by 1, padding → -100
        gen_targets = input_ids[:, 1:].clone()  # (B, 127)
        # Mask padding positions with -100 (ignored by CrossEntropyLoss)
        pad_mask = attn_mask[:, 1:] == 0
        gen_targets[pad_mask] = -100

        collated["target_ids"] = target_ids
        collated["generation_targets"] = gen_targets

    return collated

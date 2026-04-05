"""Official MIMIC-CXR split management.

Reads the PhysioNet ``mimic-cxr-2.0.0-split.csv`` to ensure patient-level
train / validate / test partitioning with zero leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

Split = Literal["train", "validate", "test"]


class SplitManager:
    """Manage official MIMIC-CXR splits and CheXpert labels.

    Parameters:
        split_csv: Path to ``mimic-cxr-2.0.0-split.csv``.
        chexpert_csv: Path to ``mimic-cxr-2.0.0-chexpert.csv`` (optional).
        record_list_csv: Path to ``cxr-record-list.csv`` (optional).
    """

    CHEXPERT_LABELS: List[str] = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    def __init__(
        self,
        split_csv: Path | str,
        chexpert_csv: Optional[Path | str] = None,
        record_list_csv: Optional[Path | str] = None,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.chexpert_csv = Path(chexpert_csv) if chexpert_csv else None
        self.record_list_csv = Path(record_list_csv) if record_list_csv else None

        # Lazy-loaded DataFrames
        self._splits_df: Optional[pd.DataFrame] = None
        self._chexpert_df: Optional[pd.DataFrame] = None
        self._records_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Split access
    # ------------------------------------------------------------------

    @property
    def splits_df(self) -> pd.DataFrame:
        if self._splits_df is None:
            logger.info("Loading split CSV: %s", self.split_csv)
            self._splits_df = pd.read_csv(self.split_csv)
            logger.info("Split CSV: %d rows", len(self._splits_df))
        return self._splits_df

    def get_study_ids(self, split: Split) -> List[int]:
        """Return study IDs for a given split."""
        df = self.splits_df
        mask = df["split"] == split
        return df.loc[mask, "study_id"].unique().tolist()

    def get_dicom_ids(self, split: Split) -> List[str]:
        """Return DICOM IDs for a given split."""
        df = self.splits_df
        mask = df["split"] == split
        return df.loc[mask, "dicom_id"].unique().tolist()

    def get_subject_ids(self, split: Split) -> Set[int]:
        """Return subject (patient) IDs for a given split."""
        df = self.splits_df
        return set(df.loc[df["split"] == split, "subject_id"].unique())

    def verify_no_leakage(self) -> bool:
        """Assert no patient overlap between train / validate / test.

        Returns:
            True if splits are clean.

        Raises:
            ValueError: If overlap is detected.
        """
        train = self.get_subject_ids("train")
        val = self.get_subject_ids("validate")
        test = self.get_subject_ids("test")

        tv = train & val
        tt = train & test
        vt = val & test

        if tv or tt or vt:
            msg = (
                f"Patient overlap detected! "
                f"train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
            )
            raise ValueError(msg)

        logger.info(
            "No patient overlap: train=%d, val=%d, test=%d subjects",
            len(train),
            len(val),
            len(test),
        )
        return True

    # ------------------------------------------------------------------
    # CheXpert labels
    # ------------------------------------------------------------------

    @property
    def chexpert_df(self) -> pd.DataFrame:
        if self._chexpert_df is None:
            if self.chexpert_csv is None:
                raise FileNotFoundError("CheXpert CSV path not provided")
            logger.info("Loading CheXpert CSV: %s", self.chexpert_csv)
            self._chexpert_df = pd.read_csv(self.chexpert_csv)
            logger.info("CheXpert CSV: %d rows", len(self._chexpert_df))
        return self._chexpert_df

    def get_labels(
        self,
        subject_id: int,
        study_id: int,
        uncertain_strategy: Literal["zeros", "ones", "ignore"] = "zeros",
    ) -> Optional[List[float]]:
        """Retrieve CheXpert labels for a study.

        Args:
            subject_id: Patient identifier.
            study_id: Study identifier.
            uncertain_strategy: How to handle uncertain labels (-1.0).

        Returns:
            List of 14 float values or None if study not found.
        """
        df = self.chexpert_df
        mask = (df["subject_id"] == subject_id) & (df["study_id"] == study_id)
        rows = df.loc[mask, self.CHEXPERT_LABELS]

        if rows.empty:
            return None

        labels = rows.iloc[0].values.tolist()

        # Map uncertain labels
        mapped: List[float] = []
        for v in labels:
            if v == -1.0:
                if uncertain_strategy == "zeros":
                    mapped.append(0.0)
                elif uncertain_strategy == "ones":
                    mapped.append(1.0)
                else:
                    mapped.append(float("nan"))
            elif pd.isna(v):
                mapped.append(0.0)  # NaN → negative
            else:
                mapped.append(float(v))

        return mapped

    # ------------------------------------------------------------------
    # Record list (DICOM → study → subject mapping)
    # ------------------------------------------------------------------

    @property
    def records_df(self) -> pd.DataFrame:
        if self._records_df is None:
            if self.record_list_csv is None:
                raise FileNotFoundError("Record list CSV path not provided")
            logger.info("Loading record list CSV: %s", self.record_list_csv)
            self._records_df = pd.read_csv(self.record_list_csv)
        return self._records_df

    def get_studies_for_split(
        self,
        split: Split,
        dev_subset_frac: Optional[float] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Return a DataFrame of unique studies for *split*.

        Merges the split CSV with the record list to include paths.

        Args:
            split: One of ``'train'``, ``'validate'``, ``'test'``.
            dev_subset_frac: If set, sample this fraction of *subjects*
                deterministically (for dev).
            seed: Random seed for subset sampling.

        Returns:
            DataFrame with columns [subject_id, study_id, dicom_id, path].
        """
        df = self.splits_df.merge(self.records_df, on=["subject_id", "study_id", "dicom_id"])
        df = df[df["split"] == split].copy()

        if dev_subset_frac is not None and 0 < dev_subset_frac < 1.0:
            subjects = df["subject_id"].unique()
            n = max(1, int(len(subjects) * dev_subset_frac))
            rng = pd.np if hasattr(pd, "np") else __import__("numpy")
            gen = rng.random.default_rng(seed)
            subset = gen.choice(subjects, size=n, replace=False)
            df = df[df["subject_id"].isin(subset)]
            logger.info(
                "Dev subset: %d / %d subjects (%.1f%%)",
                n,
                len(subjects),
                dev_subset_frac * 100,
            )

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        """Return counts per split."""
        df = self.splits_df
        counts: Dict[str, int] = {}
        for split in ("train", "validate", "test"):
            counts[f"{split}_studies"] = int(
                df.loc[df["split"] == split, "study_id"].nunique()
            )
            counts[f"{split}_subjects"] = int(
                df.loc[df["split"] == split, "subject_id"].nunique()
            )
        return counts

"""Tests for dataset label ingestion behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.dataset import MIMICCXRDataset


def test_dataset_maps_minus_one_to_nan(tmp_path: Path):
    mimic_root = tmp_path / "mimic"
    reports_root = tmp_path / "reports"
    kg_dir = tmp_path / "kg"

    (mimic_root / "files" / "p10" / "p10000001" / "s50000001").mkdir(parents=True)
    (reports_root / "p10" / "p10000001").mkdir(parents=True)
    kg_dir.mkdir(parents=True)

    Image.fromarray(np.full((16, 16), 127, dtype=np.uint8), mode="L").save(
        mimic_root / "files" / "p10" / "p10000001" / "s50000001" / "abc.jpg"
    )
    (reports_root / "p10" / "p10000001" / "s50000001.txt").write_text("report", encoding="utf-8")

    split_csv = tmp_path / "split.csv"
    split_csv.write_text(
        "subject_id,study_id,dicom_id,split\n10000001,50000001,abc,train\n",
        encoding="utf-8",
    )

    chexpert_cols = [
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
    chexpert_values = ["-1"] + ["0"] * 13
    chexpert_csv = tmp_path / "chexpert.csv"
    chexpert_csv.write_text(
        "subject_id,study_id," + ",".join(chexpert_cols) + "\n"
        + "10000001,50000001," + ",".join(chexpert_values) + "\n",
        encoding="utf-8",
    )

    ds = MIMICCXRDataset(
        split="train",
        mimic_root=mimic_root,
        reports_root=reports_root,
        split_csv=split_csv,
        chexpert_csv=chexpert_csv,
        kg_artifacts_dir=kg_dir,
        include_graphs=False,
        split_strategy="official",
    )

    assert len(ds) == 1
    labels = ds.samples[0]["labels"]
    assert isinstance(labels, torch.Tensor)
    assert bool(torch.isnan(labels[0])) is True
    assert np.isclose(float(labels[1]), 0.0)


def test_subset_hash_enforces_class_coverage(tmp_path: Path):
    mimic_root = tmp_path / "mimic"
    reports_root = tmp_path / "reports"
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir(parents=True)

    chexpert_cols = [
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

    split_lines = ["subject_id,study_id,dicom_id,split"]
    chexpert_lines = ["subject_id,study_id," + ",".join(chexpert_cols)]

    # 42 studies total (3 positives per class) to support val/test coverage.
    for i in range(42):
        subject_id = 10000000 + i
        study_id = 50000000 + i
        dicom_id = f"d{i}"
        class_idx = i % 14

        prefix = f"p{str(subject_id)[:2]}"
        subject_dir = f"p{subject_id}"
        study_dir = mimic_root / "files" / prefix / subject_dir / f"s{study_id}"
        report_dir = reports_root / prefix / subject_dir
        study_dir.mkdir(parents=True)
        report_dir.mkdir(parents=True)
        Image.fromarray(np.full((16, 16), 127, dtype=np.uint8), mode="L").save(
            study_dir / f"{dicom_id}.jpg"
        )
        (report_dir / f"s{study_id}.txt").write_text("report", encoding="utf-8")

        split_lines.append(f"{subject_id},{study_id},{dicom_id},train")

        labels = ["0"] * 14
        labels[class_idx] = "1"
        chexpert_lines.append(f"{subject_id},{study_id}," + ",".join(labels))

    split_csv = tmp_path / "split.csv"
    split_csv.write_text("\n".join(split_lines) + "\n", encoding="utf-8")

    chexpert_csv = tmp_path / "chexpert.csv"
    chexpert_csv.write_text("\n".join(chexpert_lines) + "\n", encoding="utf-8")

    val_ds = MIMICCXRDataset(
        split="validate",
        mimic_root=mimic_root,
        reports_root=reports_root,
        split_csv=split_csv,
        chexpert_csv=chexpert_csv,
        kg_artifacts_dir=kg_dir,
        include_graphs=False,
        split_strategy="subset_hash",
        subset_seed=42,
        subset_train_ratio=0.7,
        subset_val_ratio=0.15,
        subset_test_ratio=0.15,
    )
    test_ds = MIMICCXRDataset(
        split="test",
        mimic_root=mimic_root,
        reports_root=reports_root,
        split_csv=split_csv,
        chexpert_csv=chexpert_csv,
        kg_artifacts_dir=kg_dir,
        include_graphs=False,
        split_strategy="subset_hash",
        subset_seed=42,
        subset_train_ratio=0.7,
        subset_val_ratio=0.15,
        subset_test_ratio=0.15,
    )

    def present_positive_classes(ds: MIMICCXRDataset) -> set[int]:
        present: set[int] = set()
        for sample in ds.samples:
            labels = sample["labels"]
            for idx, value in enumerate(labels.tolist()):
                if value == 1.0:
                    present.add(idx)
        return present

    assert len(present_positive_classes(val_ds)) == 14
    assert len(present_positive_classes(test_ds)) == 14


def test_dataset_auto_resolves_jpg_images(tmp_path: Path):
    mimic_root = tmp_path / "mimic-jpg"
    reports_root = tmp_path / "reports"
    kg_dir = tmp_path / "kg"

    image_dir = mimic_root / "files" / "p10" / "p10000001" / "s50000001"
    image_dir.mkdir(parents=True)
    (reports_root / "p10" / "p10000001").mkdir(parents=True)
    kg_dir.mkdir(parents=True)

    image_path = image_dir / "abc.jpg"
    Image.fromarray(np.full((32, 32), 127, dtype=np.uint8), mode="L").save(image_path)
    (reports_root / "p10" / "p10000001" / "s50000001.txt").write_text(
        "report",
        encoding="utf-8",
    )

    split_csv = tmp_path / "split.csv"
    split_csv.write_text(
        "subject_id,study_id,dicom_id,split\n10000001,50000001,abc,train\n",
        encoding="utf-8",
    )

    chexpert_cols = [
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
    chexpert_csv = tmp_path / "chexpert.csv"
    chexpert_csv.write_text(
        "subject_id,study_id," + ",".join(chexpert_cols) + "\n"
        + "10000001,50000001," + ",".join(["0"] * 14) + "\n",
        encoding="utf-8",
    )

    ds = MIMICCXRDataset(
        split="train",
        mimic_root=mimic_root,
        reports_root=reports_root,
        split_csv=split_csv,
        chexpert_csv=chexpert_csv,
        kg_artifacts_dir=kg_dir,
        include_graphs=False,
        split_strategy="official",
    )

    assert len(ds) == 1
    assert ds.samples[0]["image_path"].suffix == ".jpg"

    item = ds[0]
    assert item["image"].size == (224, 224)

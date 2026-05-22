"""Focused tests for CLI/runtime surfaces added for thesis evaluation workflows."""

from __future__ import annotations

import csv
import json

from main import parse_args
from src.pipelines.comparison_pipeline import ComparisonPipeline


def test_parse_args_supports_new_commands():
    audit_args = parse_args(["audit-data", "--output-dir", "audit_out"])
    assert audit_args.command == "audit-data"
    assert audit_args.output_dir == "audit_out"

    predict_args = parse_args([
        "predict",
        "--checkpoint",
        "best.pt",
        "--image-path",
        "sample.jpg",
        "--save-explainability",
    ])
    assert predict_args.command == "predict"
    assert predict_args.checkpoint == "best.pt"
    assert predict_args.image_path == "sample.jpg"
    assert predict_args.save_explainability is True

    compare_args = parse_args([
        "compare",
        "--eval-a",
        "eval_a",
        "--eval-b",
        "eval_b",
        "--label-a",
        "baseline",
        "--label-b",
        "kg",
    ])
    assert compare_args.command == "compare"
    assert compare_args.label_a == "baseline"
    assert compare_args.label_b == "kg"


def test_parse_args_accepts_global_options_before_or_after_subcommand():
    before = parse_args(["--config", "configs/hydra_phase2_baseline_jpg.yaml", "train", "--seed", "7"])
    after = parse_args(["train", "--config", "configs/hydra_phase2_baseline_jpg.yaml", "--seed", "7"])

    assert before.command == "train"
    assert before.config == "configs/hydra_phase2_baseline_jpg.yaml"
    assert before.seed == 7
    assert after.command == "train"
    assert after.config == "configs/hydra_phase2_baseline_jpg.yaml"
    assert after.seed == 7


def test_comparison_pipeline_writes_outputs(tmp_path):
    eval_a = tmp_path / "eval_a"
    eval_b = tmp_path / "eval_b"
    eval_a.mkdir()
    eval_b.mkdir()

    metrics = {"classification": {"macro": {"f1": 0.5}}, "generation": {"rouge_l": 0.3}}
    (eval_a / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (eval_b / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    for directory, rows in (
        (eval_a, [("s1", 1, 1), ("s2", 0, 0), ("s3", 1, 0), ("s4", 0, 1)]),
        (eval_b, [("s1", 1, 1), ("s2", 0, 1), ("s3", 1, 1), ("s4", 0, 0)]),
    ):
        with (directory / "classification_predictions.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["study_key", "true_atelectasis", "score_atelectasis", "pred_atelectasis"],
            )
            writer.writeheader()
            for study_key, true_value, pred_value in rows:
                writer.writerow(
                    {
                        "study_key": study_key,
                        "true_atelectasis": true_value,
                        "score_atelectasis": 0.9 if pred_value else 0.1,
                        "pred_atelectasis": pred_value,
                    }
                )

    for directory, suffix in ((eval_a, "A"), (eval_b, "B")):
        with (directory / "generation_predictions.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"study_key": "s1", "reference_report": "normal heart", "generated_report": f"normal heart {suffix}"}) + "\n")
            f.write(json.dumps({"study_key": "s2", "reference_report": "no effusion", "generated_report": f"no effusion {suffix}"}) + "\n")

    output_dir = tmp_path / "comparison"
    results = ComparisonPipeline().run(
        eval_dir_a=eval_a,
        eval_dir_b=eval_b,
        label_a="baseline",
        label_b="kg",
        output_dir=output_dir,
    )

    assert "classification" in results
    assert "generation" in results
    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "comparison.md").exists()
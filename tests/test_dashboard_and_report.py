"""Tests for the report builder and the dashboard data helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.dashboard_data import (
    artifact_inventory,
    baseline_rows,
    cross_user_rows,
    dataset_facts,
    dataset_notice,
    experiment_status,
    improvement_rows,
    load_bundle,
    load_prediction_arrays,
    resolve_paths,
    score_kind_note,
    select_test_sample,
)
from src.utils import ArtifactUnavailableError, ProjectPaths, save_json


def _write_minimal_artifacts(paths: ProjectPaths) -> None:
    summary = {
        "source": "uci_har",
        "is_synthetic": False,
        "source_label": "UCI HAR (data nyata)",
        "n_samples": 100,
        "n_features": 10,
        "n_subjects": 4,
        "n_train": 70,
        "n_test": 30,
        "class_distribution": [
            {"code": index, "activity": name, "display": name, "n": 10, "pct": 10.0}
            for index, name in enumerate(["WALKING", "SITTING", "STANDING", "LAYING", "UP", "DOWN"])
        ],
        "subjects": [],
        "integrity": {"n_missing_values": 0, "n_duplicate_rows": 0},
        "metadata": {"device": "Samsung Galaxy S II", "doi": "10.24432/C54S4K"},
        "feature_groups": {"sensor_counts": {}},
    }
    save_json(paths.tables / "dataset_summary.json", summary)

    baseline = {
        "experiment": "A_baseline",
        "protocol": "official_subject_disjoint",
        "status": "complete",
        "split": {"train_subjects": [1, 2], "test_subjects": [3, 4], "n_train": 70, "n_test": 30},
        "models": {
            "random_forest": {
                "display_name": "Random Forest",
                "metrics": {
                    "accuracy": 0.9,
                    "balanced_accuracy": 0.89,
                    "macro_f1": 0.88,
                    "weighted_f1": 0.9,
                    "cohen_kappa": 0.87,
                    "n_samples": 30,
                    "per_class": [
                        {"activity": name, "precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 5}
                        for name in ["WALKING", "SITTING", "STANDING", "LAYING", "UP", "DOWN"]
                    ],
                    "confusion_matrix": [[5, 0, 0, 0, 0, 0]] * 6,
                    "top_confusions": [],
                },
                "fit_time_s": 1.5,
                "predict_time_s": 0.02,
                "score_kind": "probability",
            }
        },
    }
    save_json(paths.results / "baseline.json", baseline)


def test_empty_project_reports_missing_artifacts(config):
    paths = resolve_paths(config, "research")
    bundle = load_bundle(config, "research")
    assert bundle["dataset_summary"] is None
    assert baseline_rows(bundle) == []
    assert cross_user_rows(bundle) == []
    assert improvement_rows(bundle) == []
    statuses = {row["experiment"]: row["status"] for row in experiment_status(bundle)}
    assert statuses["A - Baseline"] == "belum dijalankan"
    inventory = artifact_inventory(paths)
    assert any(artifact.key == "baseline" for artifact in inventory)
    assert all(not artifact.exists for artifact in inventory)


def test_bundle_reads_available_artifacts(config):
    paths = resolve_paths(config, "research").ensure()
    _write_minimal_artifacts(paths)
    bundle = load_bundle(config, "research")
    assert bundle["dataset_summary"] is not None
    rows = baseline_rows(bundle)
    assert len(rows) == 1
    assert rows[0]["model"] == "random_forest"
    assert rows[0]["macro_f1"] == 0.88
    facts = dataset_facts(bundle)
    assert facts["n_samples"] == 100
    assert facts["device"] == "Samsung Galaxy S II"
    assert dataset_notice(bundle) is None
    statuses = {row["experiment"]: row["status"] for row in experiment_status(bundle)}
    assert statuses["A - Baseline"] == "selesai"


def test_research_mode_rejects_synthetic_summary(config):
    paths = resolve_paths(config, "research").ensure()
    save_json(paths.tables / "dataset_summary.json", {"is_synthetic": True, "source": "synthetic_demo"})
    with pytest.raises(ArtifactUnavailableError):
        load_bundle(config, "research")


def test_demo_mode_rejects_non_synthetic_summary(config):
    paths = resolve_paths(config, "demo").ensure()
    save_json(paths.tables / "dataset_summary.json", {"is_synthetic": False, "source": "uci_har"})
    with pytest.raises(ArtifactUnavailableError):
        load_bundle(config, "demo")


def test_demo_mode_uses_separate_folders(config):
    research = resolve_paths(config, "research")
    demo = resolve_paths(config, "demo")
    assert research.results != demo.results
    assert demo.results.as_posix().endswith("reports/demo/results")
    assert demo.models.as_posix().endswith("models/demo")


def test_load_prediction_arrays_missing_file(config):
    paths = resolve_paths(config, "research")
    with pytest.raises(ArtifactUnavailableError):
        load_prediction_arrays(paths, "reports/predictions/does_not_exist.npz")


def test_select_test_sample(demo_dataset):
    x, label, subject, raw = select_test_sample(demo_dataset, 0)
    assert x.shape == (demo_dataset.n_features,)
    assert 0 <= label < len(demo_dataset.activity_names)
    assert subject in demo_dataset.subject_ids
    assert raw.shape == (128, len(demo_dataset.raw_channels))


def test_score_kind_note_is_explicit():
    assert "bukan probabilitas terkalibrasi" in score_kind_note("normalized_decision_score")
    assert "probabilitas" in score_kind_note("probability")


def test_report_builder_creates_summary(config, paths):
    from scripts.generate_report import build_report

    _write_minimal_artifacts(paths)
    save_json(
        paths.results / "cross_device.json",
        {
            "status": "not_performed",
            "reason": "Hanya satu perangkat.",
            "requirements": ["minimal dua perangkat"],
            "documentation": "docs/cross_device_protocol.md",
        },
    )
    manifest = build_report(config, paths, mode="research")
    summary_path = paths.reports / "experiment_summary.md"
    assert summary_path.exists()
    text = summary_path.read_text(encoding="utf-8")
    assert "Eksperimen A" in text
    assert "Eksperimen C" in text
    assert manifest["counts"]["n_samples"] == 100
    assert (paths.tables / "model_comparison.csv").exists()

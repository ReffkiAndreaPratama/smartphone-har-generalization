"""Integration tests for experiments A, B, D and their leakage guarantees."""

from __future__ import annotations

import numpy as np
import pytest

from src.generalization import run_baseline, run_group_cv_cnn, run_improvement, run_loso
from src.utils import read_json


@pytest.fixture
def baseline(demo_dataset, config, paths):
    return run_baseline(
        demo_dataset,
        config,
        paths,
        models=["random_forest"],
        seed=42,
    )


def test_baseline_writes_artifacts(baseline, paths):
    assert baseline["status"] == "complete"
    assert "random_forest" in baseline["models"]
    assert (paths.results / "baseline.json").exists()
    assert (paths.results / "model_config.json").exists()
    model_file = paths.models / "baseline_random_forest.joblib"
    assert model_file.exists()
    metrics = baseline["models"]["random_forest"]["metrics"]
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "per_class", "confusion_matrix"):
        assert key in metrics
    assert len(metrics["per_class"]) == 6


def test_baseline_split_is_subject_disjoint(baseline, demo_dataset):
    train_subjects = set(baseline["split"]["train_subjects"])
    test_subjects = set(baseline["split"]["test_subjects"])
    assert train_subjects.isdisjoint(test_subjects)
    assert train_subjects | test_subjects == set(demo_dataset.subject_ids)


def test_loso_produces_one_fold_per_subject(baseline, demo_dataset, config, paths):
    payload = run_loso(
        demo_dataset,
        config,
        paths,
        models=["random_forest"],
        max_folds=3,
        seed=42,
    )
    entry = payload["models"]["random_forest"]
    assert entry["status"] == "complete"
    assert entry["n_folds"] == 3
    assert len(entry["per_subject"]) == 3
    assert (paths.results / "cross_user.json").exists()
    for fold in entry["folds"]:
        assert set(fold["test_subjects"]).isdisjoint(set(fold["train_subjects"]))
        assert fold["metrics"]["n_samples"] == fold["n_test"]


def test_loso_requires_baseline_first(demo_dataset, config, paths):
    from src.utils import ProjectError

    with pytest.raises(ProjectError):
        run_loso(demo_dataset, config, paths, models=["random_forest"], max_folds=1, seed=42)


def test_group_cv_cnn_uses_subject_disjoint_folds(baseline, demo_dataset, config, paths):
    payload = run_group_cv_cnn(
        demo_dataset,
        config,
        paths,
        n_splits=2,
        seed=42,
        max_folds=2,
    )
    entry = payload["models"]["cnn"]
    assert entry["status"] == "complete"
    assert entry["n_folds"] == 2
    tested = [subject for fold in entry["folds"] for subject in fold["test_subjects"]]
    assert sorted(tested) == demo_dataset.subject_ids
    for fold in entry["folds"]:
        assert set(fold["test_subjects"]).isdisjoint(set(fold["train_subjects"]))


def test_improvement_compares_strategies_on_equal_folds(baseline, demo_dataset, config, paths):
    payload = run_improvement(
        demo_dataset,
        config,
        paths,
        models=["random_forest"],
        n_splits=2,
        strategies=["base", "subject_norm", "calibration"],
        seed=42,
    )
    strategies = payload["strategies"]["random_forest"]
    assert set(strategies) == {"base", "subject_norm", "calibration"}
    for name, record in strategies.items():
        assert record["status"] == "complete"
        assert record["n_folds"] == 2
        assert record["aggregate"]["macro_f1"]["mean"] is not None

    base_eval = [fold["n_eval"] for fold in strategies["base"]["folds"]]
    for name in ("subject_norm", "calibration"):
        assert [fold["n_eval"] for fold in strategies[name]["folds"]] == base_eval

    calibration = strategies["calibration"]
    for fold in calibration["folds"]:
        assert fold["n_calibration"] > 0
        assert fold["n_train"] > fold["n_eval"]

    deltas = payload["deltas_vs_base"]["random_forest"]
    assert "subject_norm" in deltas and "calibration" in deltas
    assert deltas["subject_norm"]["macro_f1"] is not None


def test_results_are_json_serialisable(baseline, paths):
    payload = read_json(paths.results / "baseline.json")
    assert payload["experiment"] == "A_baseline"
    assert payload["dataset"]["is_synthetic"] is True
    assert payload["dataset"]["notice"]

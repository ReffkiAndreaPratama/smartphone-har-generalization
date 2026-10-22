"""Tests for metric computation, per-subject breakdown and aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    aggregate_metrics,
    aggregate_values,
    compute_metrics,
    per_subject_metrics,
    summarize_predictions,
    top_confused_pairs,
)

LABELS = ["A", "B", "C"]


def test_perfect_predictions():
    truth = np.array([0, 1, 2, 0, 1, 2])
    metrics = compute_metrics(truth, truth.copy(), LABELS)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert metrics["top_confusions"] == []


def test_known_confusion_is_reported():
    truth = np.array([0, 0, 1, 2])
    prediction = np.array([0, 1, 1, 2])
    metrics = compute_metrics(truth, prediction, LABELS)
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"][0][1] == 1
    pairs = top_confused_pairs(metrics, limit=1)
    assert pairs[0]["true"] == "A" and pairs[0]["predicted"] == "B"
    assert pairs[0]["count"] == 1
    assert pairs[0]["rate_of_true"] == pytest.approx(0.5)


def test_compute_metrics_validates_inputs():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0]), LABELS)
    with pytest.raises(ValueError):
        compute_metrics(np.array([], dtype=int), np.array([], dtype=int), LABELS)
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 9]), np.array([0, 1]), LABELS)


def test_per_class_metrics_include_support():
    truth = np.array([0, 0, 1, 1, 2])
    prediction = np.array([0, 1, 1, 1, 2])
    metrics = compute_metrics(truth, prediction, LABELS)
    supports = {item["activity"]: item["support"] for item in metrics["per_class"]}
    assert supports == {"A": 2, "B": 2, "C": 1}
    for item in metrics["per_class"]:
        assert 0.0 <= item["precision"] <= 1.0
        assert 0.0 <= item["recall"] <= 1.0


def test_per_subject_metrics():
    truth = np.array([0, 1, 0, 1])
    prediction = np.array([0, 1, 1, 1])
    subjects = np.array([1, 1, 2, 2])
    records = per_subject_metrics(truth, prediction, subjects, LABELS)
    assert [record["subject"] for record in records] == [1, 2]
    assert records[0]["macro_f1"] == 1.0
    assert records[1]["accuracy"] == 0.5
    assert records[1]["n_classes_present"] == 2


def test_aggregate_values_and_metrics():
    aggregate = aggregate_values([0.5, 0.7, 0.9])
    assert aggregate["n"] == 3
    assert aggregate["mean"] == pytest.approx(0.7)
    assert aggregate["std"] == pytest.approx(0.2)
    assert aggregate["min"] == 0.5 and aggregate["max"] == 0.9
    assert len(aggregate["ci95"]) == 2

    records = [
        {"accuracy": 0.8, "balanced_accuracy": 0.78, "macro_f1": 0.76},
        {"accuracy": 0.9, "balanced_accuracy": 0.88, "macro_f1": 0.86},
    ]
    result = aggregate_metrics(records)
    assert result["n_folds"] == 2
    assert result["macro_f1"]["mean"] == pytest.approx(0.81)


def test_summarize_predictions_structure():
    truth = np.array([0, 1, 2, 0])
    prediction = np.array([0, 1, 1, 0])
    subjects = np.array([1, 1, 2, 2])
    payload = summarize_predictions(truth, prediction, subjects, LABELS)
    assert "metrics" in payload and "per_subject" in payload
    assert payload["metrics"]["accuracy"] == 0.75

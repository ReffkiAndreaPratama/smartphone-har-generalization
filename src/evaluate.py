"""Evaluation metrics, per-subject breakdowns and fold aggregation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    y_score: np.ndarray | None = None,
    include_confusion: bool = True,
) -> dict[str, Any]:
    """Compute the metrics reported in the thesis tables."""
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if len(truth) != len(prediction):
        raise ValueError("Panjang y_true dan y_pred tidak sama")
    if len(truth) == 0:
        raise ValueError("Tidak ada sampel untuk dievaluasi")
    class_indices = np.arange(len(labels))
    unknown = sorted((set(np.unique(truth)) | set(np.unique(prediction))) - set(class_indices))
    if unknown:
        raise ValueError(f"Label di luar daftar aktivitas: {unknown}")

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=class_indices,
        zero_division=0,
    )
    matrix = confusion_matrix(truth, prediction, labels=class_indices)
    per_class = [
        {
            "code": int(index),
            "activity": labels[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in class_indices
    ]

    top_confusions = []
    for true_index in class_indices:
        total_true = int(matrix[true_index].sum())
        for predicted_index in class_indices:
            if true_index == predicted_index:
                continue
            count = int(matrix[true_index, predicted_index])
            if count > 0:
                top_confusions.append(
                    {
                        "true": labels[true_index],
                        "predicted": labels[predicted_index],
                        "count": count,
                        "rate_of_true": round(count / total_true, 4) if total_true else 0.0,
                    }
                )
    top_confusions.sort(key=lambda item: -item["count"])

    metrics: dict[str, Any] = {
        "n_samples": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", labels=class_indices)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", labels=class_indices)),
        "cohen_kappa": float(cohen_kappa_score(truth, prediction)),
        "per_class": per_class,
        "top_confusions": top_confusions[:10],
        "class_distribution": {
            labels[index]: int((truth == index).sum()) for index in class_indices
        },
    }
    if include_confusion:
        metrics["confusion_matrix"] = matrix.astype(int).tolist()
    if y_score is not None:
        try:
            metrics["log_loss"] = float(log_loss(truth, y_score, labels=class_indices))
        except ValueError:
            metrics["log_loss"] = None
    return metrics


def per_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject: np.ndarray,
    labels: list[str],
) -> list[dict[str, Any]]:
    """Metrics per held-out subject.

    Macro-F1 is computed over activities actually present for that subject.
    This avoids punishing a subject for an activity that never occurs in their
    evaluation window while keeping the definition explicit in the output.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    subject_values = np.asarray(subject, dtype=np.int64)
    records = []
    for subject_id in sorted(int(value) for value in np.unique(subject_values)):
        mask = subject_values == subject_id
        if not mask.any():
            continue
        present = sorted(int(value) for value in np.unique(truth[mask]))
        records.append(
            {
                "subject": subject_id,
                "n_samples": int(mask.sum()),
                "n_classes_present": len(present),
                "accuracy": float(accuracy_score(truth[mask], prediction[mask])),
                "balanced_accuracy": float(
                    balanced_accuracy_score(truth[mask], prediction[mask])
                ),
                "macro_f1": float(
                    f1_score(
                        truth[mask],
                        prediction[mask],
                        labels=present,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    return records


def _ci95(values: np.ndarray) -> list[float]:
    if len(values) < 2:
        return [float(values[0]), float(values[0])] if len(values) else [0.0, 0.0]
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    critical = float(stats.t.ppf(0.975, df=len(values) - 1))
    mean = float(np.mean(values))
    return [mean - critical * standard_error, mean + critical * standard_error]


def aggregate_values(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None, "ci95": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "ci95": _ci95(array),
    }


def aggregate_metrics(
    records: list[dict[str, Any]], metric_keys: tuple[str, ...] = ("accuracy", "balanced_accuracy", "macro_f1")
) -> dict[str, Any]:
    """Aggregate fold-level metrics (mean, std, 95% CI)."""
    aggregate = {}
    for key in metric_keys:
        values = [record.get(key) for record in records if record.get(key) is not None]
        aggregate[key] = aggregate_values(values)
    aggregate["n_folds"] = len(records)
    return aggregate


def summarize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject: np.ndarray,
    labels: list[str],
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = compute_metrics(y_true, y_pred, labels, y_score=y_score)
    per_subject = per_subject_metrics(y_true, y_pred, subject, labels)
    return {
        "metrics": metrics,
        "per_subject": per_subject,
        "per_subject_aggregate": aggregate_metrics(per_subject),
    }


def top_confused_pairs(metrics: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    return list(metrics.get("top_confusions", []))[:limit]

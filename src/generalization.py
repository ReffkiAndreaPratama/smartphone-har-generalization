"""Generalisation experiments (A, B, D) with leakage guardrails.

Protokol:
* A: split resmi UCI (subjek train dan test terpisah).
* B: Leave-One-Subject-Out untuk Random Forest/SVM; GroupKFold(6) untuk CNN
  karena keterbatasan komputasi CPU (setiap subjek tetap diuji tepat sekali).
* D: GroupKFold(6) yang sama untuk semua strategi dan model; subset evaluasi
  target identik agar perbandingan adil.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data_loader import HARDataset, build_dataset_summary
from .evaluate import aggregate_metrics, compute_metrics, per_subject_metrics
from .models import CNNConfig, MODEL_DISPLAY_NAMES, TorchHARModel
from .preprocessing import (
    RawSignalStandardizer,
    fit_subject_statistics,
    transform_subject_statistics,
)
from .split_strategy import (
    Fold,
    assert_indices_disjoint,
    assert_no_subject_overlap,
    calibration_split,
    folds_to_records,
    group_kfold_folds,
    loso_folds,
    official_split_indices,
)
from .train import (
    load_sklearn_model,
    predict_cnn,
    predict_sklearn,
    save_cnn_model,
    save_sklearn_model,
    select_validation_subjects,
    train_cnn,
    train_sklearn,
)
from .utils import (
    DEMO_NOTICE,
    ProjectError,
    ProjectPaths,
    get_logger,
    now_iso,
    read_json,
    save_json,
)

SKLEARN_MODELS = ("random_forest", "svm")
ALL_MODELS = ("random_forest", "svm", "cnn")

STRATEGY_DESCRIPTIONS = {
    "base": "Model dilatih pada subjek sumber saja; evaluasi pada subset target yang sama.",
    "subject_norm": (
        "Standardisasi per-subjek memakai statistik subjek itu sendiri tanpa label "
        "(transductive; data target tanpa label dipakai)."
    ),
    "calibration": (
        "Sebagian kecil data target berlabel ditambahkan ke data latih; evaluasi pada "
        "sisa data target yang tidak dipakai untuk kalibrasi."
    ),
    "augmentation": (
        "Augmentasi sinyal (jitter, scaling, time shift) hanya pada data latih CNN; "
        "evaluasi pada sinyal bersih."
    ),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _relative(paths: ProjectPaths, path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return target.relative_to(paths.root.resolve()).as_posix()
    except ValueError:
        return target.as_posix()


def _compact_dataset(dataset: HARDataset) -> dict[str, Any]:
    metadata = dataset.metadata
    keep = [
        "dataset_name",
        "dataset_version",
        "repository_url",
        "doi",
        "citation",
        "license",
        "device",
        "device_count",
        "sampling_rate_hz",
        "window_size",
        "window_seconds",
        "overlap",
        "feature_source",
        "feature_note",
        "n_duplicate_feature_names",
        "official_split",
        "fingerprint",
        "seed",
        "notice",
    ]
    return {
        "source": dataset.source,
        "is_synthetic": dataset.is_synthetic,
        "notice": DEMO_NOTICE if dataset.is_synthetic else None,
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "n_subjects": len(dataset.subject_ids),
        "subject_ids": dataset.subject_ids,
        "train_subjects": sorted(int(v) for v in np.unique(dataset.subject[dataset.split == "train"])),
        "test_subjects": sorted(int(v) for v in np.unique(dataset.subject[dataset.split == "test"])),
        "metadata": {key: metadata.get(key) for key in keep if key in metadata},
    }


def _check_partition(dataset: HARDataset, train_idx: np.ndarray, test_idx: np.ndarray, context: str) -> None:
    assert_indices_disjoint(train_idx, test_idx, context)
    assert_no_subject_overlap(
        dataset.subject[train_idx], dataset.subject[test_idx], context
    )


def _metrics_with_timing(
    dataset: HARDataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = compute_metrics(y_true, y_pred, dataset.activity_names, y_score=scores)
    per_subject = per_subject_metrics(
        y_true, y_pred, np.asarray(dataset.subject)[test_idx], dataset.activity_names
    )
    return {
        "metrics": metrics,
        "per_subject": per_subject,
        "per_subject_aggregate": aggregate_metrics(per_subject),
    }


def _save_predictions(
    paths: ProjectPaths,
    prefix: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject: np.ndarray,
    scores: np.ndarray | None,
) -> str:
    paths.predictions.mkdir(parents=True, exist_ok=True)
    destination = paths.predictions / f"{prefix}.npz"
    payload = {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "subject": np.asarray(subject, dtype=np.int64),
    }
    if scores is not None:
        payload["scores"] = np.asarray(scores, dtype=np.float32)
    np.savez_compressed(destination, **payload)
    return _relative(paths, destination)


def _load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def _aggregate_per_class(folds: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    records = []
    for index, label in enumerate(labels):
        precisions = []
        recalls = []
        f1s = []
        support = 0
        for fold in folds:
            per_class = fold.get("metrics", {}).get("per_class", [])
            if index >= len(per_class):
                continue
            precisions.append(per_class[index]["precision"])
            recalls.append(per_class[index]["recall"])
            f1s.append(per_class[index]["f1"])
            support += int(per_class[index]["support"])
        records.append(
            {
                "activity": label,
                "precision_mean": float(np.mean(precisions)) if precisions else None,
                "precision_std": float(np.std(precisions, ddof=1)) if len(precisions) > 1 else 0.0,
                "recall_mean": float(np.mean(recalls)) if recalls else None,
                "recall_std": float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0,
                "f1_mean": float(np.mean(f1s)) if f1s else None,
                "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                "support_total": support,
            }
        )
    return records


def _sum_confusions(folds: list[dict[str, Any]], n_classes: int) -> list[list[int]]:
    total = np.zeros((n_classes, n_classes), dtype=np.int64)
    for fold in folds:
        matrix = fold.get("metrics", {}).get("confusion_matrix")
        if matrix:
            total += np.asarray(matrix, dtype=np.int64)
    return total.tolist()


def _finalize_cross_user_model(
    paths: ProjectPaths,
    name: str,
    entry: dict[str, Any],
    labels: list[str],
    protocol: str,
) -> None:
    folds = [fold for fold in entry.get("folds", []) if fold.get("status", "complete") == "complete"]
    if not folds:
        entry["status"] = "empty"
        return
    oof_y_true = []
    oof_y_pred = []
    oof_subject = []
    oof_scores = []
    for fold in folds:
        prediction_path = fold.get("predictions_path")
        if not prediction_path:
            continue
        arrays = _load_predictions(paths.root / prediction_path)
        oof_y_true.append(arrays["y_true"])
        oof_y_pred.append(arrays["y_pred"])
        oof_subject.append(arrays["subject"])
        if "scores" in arrays:
            oof_scores.append(arrays["scores"])
    if oof_y_true:
        y_true = np.concatenate(oof_y_true)
        y_pred = np.concatenate(oof_y_pred)
        subject = np.concatenate(oof_subject)
        scores = np.concatenate(oof_scores) if len(oof_scores) == len(folds) else None
        oof_metrics = compute_metrics(y_true, y_pred, labels, y_score=scores)
        per_subject = per_subject_metrics(y_true, y_pred, subject, labels)
        entry["per_subject"] = per_subject
        entry["per_subject_aggregate"] = aggregate_metrics(per_subject)
        entry["confusion_matrix"] = oof_metrics["confusion_matrix"]
        entry["top_confusions"] = oof_metrics["top_confusions"]
        entry["oof_metrics"] = oof_metrics
        oof_path = paths.predictions / f"cross_user_{name}_oof.npz"
        payload = {"y_true": y_true, "y_pred": y_pred, "subject": subject}
        if scores is not None:
            payload["scores"] = scores
        np.savez_compressed(oof_path, **payload)
        entry["oof_predictions"] = _relative(paths, oof_path)
    entry["aggregate"] = aggregate_metrics([fold["metrics"] for fold in folds])
    entry["per_class_mean"] = _aggregate_per_class(folds, labels)
    if "confusion_matrix" not in entry:
        entry["confusion_matrix"] = _sum_confusions(folds, len(labels))
    entry["mean_fit_time_s"] = float(np.mean([fold["fit_time_s"] for fold in folds]))
    entry["std_fit_time_s"] = float(np.std([fold["fit_time_s"] for fold in folds], ddof=1)) if len(folds) > 1 else 0.0
    entry["mean_predict_time_s"] = float(np.mean([fold["predict_time_s"] for fold in folds]))
    entry["total_fit_time_s"] = float(np.sum([fold["fit_time_s"] for fold in folds]))
    entry["total_predict_time_s"] = float(np.sum([fold["predict_time_s"] for fold in folds]))
    entry["n_folds"] = len(folds)
    entry["protocol"] = protocol
    entry["status"] = "complete"


def _compute_gap(baseline: dict[str, Any] | None, cross_user: dict[str, Any]) -> dict[str, Any] | None:
    if not baseline:
        return None
    gap = {}
    for name, payload in cross_user.get("models", {}).items():
        baseline_payload = baseline.get("models", {}).get(name)
        if not baseline_payload or payload.get("status") != "complete":
            continue
        base_value = baseline_payload["metrics"]["macro_f1"]
        cross_value = payload["aggregate"]["macro_f1"]["mean"]
        gap[name] = {
            "baseline_macro_f1": base_value,
            "cross_user_macro_f1_mean": cross_value,
            "absolute_gap": base_value - cross_value,
            "relative_drop_pct": (
                (base_value - cross_value) / base_value * 100.0 if base_value else None
            ),
        }
    return {
        "models": gap,
        "note": (
            "Baseline memakai split resmi (9 subjek uji), sedangkan lintas pengguna "
            "merata-ratakan fold dengan subjek yang berbeda. Perbedaan protokol harus "
            "dipertimbangkan saat menafsirkan selisih."
        ),
    }


def _persist_cross_user(
    paths: ProjectPaths, payload: dict[str, Any], baseline: dict[str, Any] | None
) -> Path:
    gap = _compute_gap(baseline, payload)
    if gap is not None:
        payload["gap_vs_baseline"] = gap
    return save_json(paths.results / "cross_user.json", payload)


def _load_baseline(paths: ProjectPaths) -> dict[str, Any] | None:
    return read_json(paths.results / "baseline.json")


def _load_existing_results(
    paths: ProjectPaths, filename: str, dataset: HARDataset, seed: int
) -> dict[str, Any] | None:
    """Load a partial result file only when it belongs to the same dataset/seed."""
    existing = read_json(paths.results / filename)
    if not existing:
        return None
    record = existing.get("dataset", {})
    if (
        record.get("source") != dataset.source
        or record.get("n_samples") != dataset.n_samples
        or existing.get("seed") != int(seed)
    ):
        return None
    return existing


def _load_model_config(paths: ProjectPaths) -> dict[str, Any]:
    config = read_json(paths.results / "model_config.json")
    if not config:
        raise ProjectError(
            "model_config.json belum ada. Jalankan tahap baseline terlebih dahulu "
            "(`python scripts/run_experiments.py --stages baseline`) agar hyperparameter "
            "yang dipakai lintas pengguna berasal dari data validasi, bukan data uji."
        )
    return config


def _init_cross_user(dataset: HARDataset, seed: int) -> dict[str, Any]:
    return {
        "experiment": "B_cross_user",
        "title": "Generalisasi lintas pengguna",
        "seed": int(seed),
        "status": "running",
        "created_at": now_iso(),
        "dataset": _compact_dataset(dataset),
        "protocols": {},
        "models": {},
    }


# ---------------------------------------------------------------------------
# Experiment A - baseline
# ---------------------------------------------------------------------------
def run_baseline(
    dataset: HARDataset,
    config: dict[str, Any],
    paths: ProjectPaths,
    models: Iterable[str] | None = None,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or get_logger("har.baseline")
    paths.ensure()
    model_names = list(models or config.get("experiments", {}).get("baseline", {}).get("models", ALL_MODELS))
    for name in model_names:
        if name not in ALL_MODELS:
            raise ProjectError(f"Model tidak dikenal: {name}")

    train_idx, test_idx = official_split_indices(dataset)
    _check_partition(dataset, train_idx, test_idx, "baseline official split")
    train_subjects = sorted(int(v) for v in np.unique(dataset.subject[train_idx]))
    test_subjects = sorted(int(v) for v in np.unique(dataset.subject[test_idx]))
    X = dataset.X
    y = dataset.y
    subject = dataset.subject
    raw = dataset.raw

    if "cnn" in model_names and raw is None:
        raise ProjectError("Sinyal mentah tidak tersedia; model CNN tidak dapat dilatih")

    results: dict[str, Any] = {
        "experiment": "A_baseline",
        "title": "Evaluasi baseline pada split resmi berbasis subjek",
        "protocol": "official_subject_disjoint",
        "seed": int(seed),
        "created_at": now_iso(),
        "dataset": _compact_dataset(dataset),
        "split": {
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "guardrail": "Tidak ada subjek/indeks yang tumpang tindih.",
        },
        "models": {},
        "model_config": {},
    }

    for model_name in model_names:
        log.info("Baseline %s", MODEL_DISPLAY_NAMES.get(model_name, model_name))
        if model_name in SKLEARN_MODELS:
            training = train_sklearn(
                model_name,
                X[train_idx],
                y[train_idx],
                subject[train_idx],
                config,
                seed=seed,
                select=True,
                logger=log,
            )
            predictions, scores, score_kind, predict_time = predict_sklearn(
                training.model, X[test_idx]
            )
            y_true = y[test_idx]
            metrics_payload = _metrics_with_timing(
                dataset, y_true, predictions, test_idx, scores=scores
            )
            model_path = paths.models / f"baseline_{model_name}.joblib"
            metadata = {
                "model_name": model_name,
                "display_name": MODEL_DISPLAY_NAMES.get(model_name),
                "params": training.params,
                "seed": int(seed),
                "feature_names": dataset.feature_names,
                "activity_names": dataset.activity_names,
                "n_features": dataset.n_features,
                "dataset_source": dataset.source,
                "is_synthetic": dataset.is_synthetic,
                "training_subjects": train_subjects,
                "n_train": int(len(train_idx)),
                "preprocessing": (
                    "StandardScaler di dalam Pipeline; fit hanya pada data latih."
                    if model_name == "svm"
                    else "Tanpa scaling; Random Forest dilatih pada fitur asli."
                ),
                "trained_at": now_iso(),
            }
            save_sklearn_model(model_path, training.model, metadata)
            predictions_path = _save_predictions(
                paths,
                f"baseline_{model_name}",
                y_true,
                predictions,
                subject[test_idx],
                scores,
            )
            results["models"][model_name] = {
                "display_name": MODEL_DISPLAY_NAMES.get(model_name),
                "params": training.params,
                "selection": training.selection,
                "fit_time_s": round(training.fit_time_s, 4),
                "predict_time_s": round(predict_time, 4),
                "total_time_s": round(training.fit_time_s + predict_time, 4),
                "score_kind": score_kind,
                "n_train": int(len(train_idx)),
                "n_train_subjects": len(train_subjects),
                "model_path": _relative(paths, model_path),
                "predictions_path": predictions_path,
                **metrics_payload,
            }
            results["model_config"][model_name] = {
                "params": training.params,
                "selection_method": training.selection.get("method"),
            }
        else:
            cnn_config = CNNConfig.from_config(config, seed=seed)
            cnn = train_cnn(
                raw,
                y,
                subject,
                train_idx,
                cnn_config,
                seed=seed,
                class_names=dataset.activity_names,
                logger=log,
            )
            predictions, probabilities, predict_time = predict_cnn(cnn, raw[test_idx])
            y_true = y[test_idx]
            metrics_payload = _metrics_with_timing(
                dataset, y_true, predictions, test_idx, scores=probabilities
            )
            model_path = paths.models / "baseline_cnn.pt"
            metadata = {
                "model_name": "cnn",
                "dataset_source": dataset.source,
                "is_synthetic": dataset.is_synthetic,
                "training_subjects": train_subjects,
                "preprocessing": "Standardisasi per channel memakai statistik data latih CNN.",
                "feature_names": None,
                "raw_channels": dataset.raw_channels,
                "window_size": int(raw.shape[1]),
            }
            save_cnn_model(cnn, model_path, extra_metadata=metadata)
            predictions_path = _save_predictions(
                paths,
                "baseline_cnn",
                y_true,
                predictions,
                subject[test_idx],
                probabilities,
            )
            results["models"]["cnn"] = {
                "display_name": MODEL_DISPLAY_NAMES.get("cnn"),
                "config": cnn_config.to_dict(),
                "history": cnn.history,
                "best_epoch": cnn.best_epoch,
                "n_parameters": cnn.metadata.get("n_parameters"),
                "val_subjects": cnn.metadata.get("val_subjects"),
                "best_val_macro_f1": cnn.metadata.get("best_val_macro_f1"),
                "fit_time_s": round(float(cnn.metadata.get("fit_time_s", 0.0)), 4),
                "predict_time_s": round(predict_time, 4),
                "total_time_s": round(float(cnn.metadata.get("fit_time_s", 0.0)) + predict_time, 4),
                "score_kind": "probability",
                "n_train": int(len(train_idx)),
                "n_train_subjects": len(train_subjects),
                "model_path": _relative(paths, model_path),
                "predictions_path": predictions_path,
                **metrics_payload,
            }
            results["model_config"]["cnn"] = {"config": cnn_config.to_dict()}

    results["status"] = "complete"
    save_json(paths.results / "baseline.json", results)
    save_json(
        paths.results / "model_config.json",
        {
            "seed": int(seed),
            "dataset_source": dataset.source,
            "is_synthetic": dataset.is_synthetic,
            "created_at": now_iso(),
            "models": results["model_config"],
            "note": (
                "Hyperparameter sklearn dipilih dengan GroupKFold pada subjek data latih "
                "split resmi. Konfigurasi ini dibekukan untuk fold LOSO. Pemilihan "
                "per-fold belum dilakukan karena biaya komputasi; lihat README."
            ),
        },
    )
    log.info("Baseline selesai: %s", ", ".join(results["models"]))
    return results


# ---------------------------------------------------------------------------
# Experiment B - LOSO for sklearn
# ---------------------------------------------------------------------------
def run_loso(
    dataset: HARDataset,
    config: dict[str, Any],
    paths: ProjectPaths,
    models: Iterable[str] | None = None,
    max_folds: int | None = None,
    seed: int = 42,
    resume: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or get_logger("har.loso")
    paths.ensure()
    model_names = [name for name in (models or ("random_forest", "svm")) if name in SKLEARN_MODELS]
    if not model_names:
        raise ProjectError("Tahap LOSO hanya mendukung random_forest dan svm")
    model_config = _load_model_config(paths)
    folds = loso_folds(dataset, max_folds=max_folds)
    baseline = _load_baseline(paths)
    existing = _load_existing_results(paths, "cross_user.json", dataset, seed)
    payload = existing or _init_cross_user(dataset, seed)
    payload["dataset"] = _compact_dataset(dataset)
    payload["protocols"].update({name: "leave_one_subject_out" for name in model_names})
    payload["folds"] = [fold.to_dict() for fold in folds]
    payload["status"] = "running"
    current_fold_ids = {fold.fold for fold in folds}

    X = dataset.X
    y = dataset.y
    subject = dataset.subject
    for model_name in model_names:
        entry = payload["models"].get(model_name, {})
        entry.update(
            {
                "display_name": MODEL_DISPLAY_NAMES.get(model_name),
                "protocol": "leave_one_subject_out",
                "params": model_config["models"].get(model_name, {}).get("params", {}),
                "folds": entry.get("folds", []),
                "status": "running",
            }
        )
        entry["folds"] = [record for record in entry["folds"] if int(record["fold"]) in current_fold_ids]
        done_folds = {int(record["fold"]) for record in entry["folds"]} if resume else set()
        for fold in folds:
            if fold.fold in done_folds:
                log.info("LOSO %s fold %d sudah ada, dilewati", model_name, fold.fold)
                continue
            _check_partition(dataset, fold.train_idx, fold.test_idx, f"LOSO {model_name} fold {fold.fold}")
            log.info(
                "LOSO %s fold %d/%d (subjek uji %s)",
                model_name,
                fold.fold,
                len(folds),
                fold.test_subjects,
            )
            frozen = dict(
                entry.get("params") or model_config["models"][model_name]["params"]
            )
            from .models import make_sklearn_model

            model = make_sklearn_model(
                model_name,
                frozen,
                seed=seed + fold.fold,
                n_jobs=int(config.get("project", {}).get("n_jobs", -1)),
            )
            start = time.perf_counter()
            model.fit(X[fold.train_idx], y[fold.train_idx])
            fit_time = time.perf_counter() - start
            predictions, scores, score_kind, predict_time = predict_sklearn(
                model, X[fold.test_idx]
            )
            metrics_payload = _metrics_with_timing(
                dataset, y[fold.test_idx], predictions, fold.test_idx, scores=scores
            )
            predictions_path = _save_predictions(
                paths,
                f"cross_user_{model_name}_fold{fold.fold:02d}",
                y[fold.test_idx],
                predictions,
                subject[fold.test_idx],
                scores,
            )
            record = {
                "fold": int(fold.fold),
                "test_subjects": fold.test_subjects,
                "train_subjects": fold.train_subjects,
                "n_train": int(len(fold.train_idx)),
                "n_test": int(len(fold.test_idx)),
                "status": "complete",
                "fit_time_s": round(float(fit_time), 4),
                "predict_time_s": round(predict_time, 4),
                "score_kind": score_kind,
                "predictions_path": predictions_path,
                **metrics_payload,
            }
            entry["folds"].append(record)
            entry["folds"].sort(key=lambda item: item["fold"])
            payload["models"][model_name] = entry
            _persist_cross_user(paths, payload, baseline)

        _finalize_cross_user_model(paths, model_name, entry, dataset.activity_names, "leave_one_subject_out")
        payload["models"][model_name] = entry
        payload["protocols"][model_name] = "leave_one_subject_out"
        _persist_cross_user(paths, payload, baseline)
        log.info(
            "LOSO %s selesai: macro-F1 %.4f +/- %.4f",
            model_name,
            entry["aggregate"]["macro_f1"]["mean"],
            entry["aggregate"]["macro_f1"]["std"],
        )

    payload["status"] = "partial"
    if all(
        payload["models"].get(name, {}).get("status") == "complete" for name in payload["models"]
    ):
        payload["status"] = "complete"
    _persist_cross_user(paths, payload, baseline)
    return payload


# ---------------------------------------------------------------------------
# Experiment B - GroupKFold for CNN
# ---------------------------------------------------------------------------
def run_group_cv_cnn(
    dataset: HARDataset,
    config: dict[str, Any],
    paths: ProjectPaths,
    n_splits: int = 6,
    seed: int = 42,
    max_folds: int | None = None,
    resume: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or get_logger("har.cnn_cv")
    if dataset.raw is None:
        raise ProjectError("Sinyal mentah tidak tersedia; CNN tidak dapat dijalankan")
    paths.ensure()
    if max_folds is not None and max_folds > 0:
        n_splits = min(n_splits, max_folds)
    folds = group_kfold_folds(dataset, n_splits=n_splits, seed=seed)
    baseline = _load_baseline(paths)
    existing = _load_existing_results(paths, "cross_user.json", dataset, seed)
    payload = existing or _init_cross_user(dataset, seed)
    payload["dataset"] = _compact_dataset(dataset)
    entry = payload["models"].get("cnn", {})
    cnn_config = CNNConfig.from_config(config, seed=seed)
    protocol_name = f"group_kfold_{n_splits}"
    if entry.get("protocol") != protocol_name:
        entry["folds"] = []
    entry.update(
        {
            "display_name": MODEL_DISPLAY_NAMES.get("cnn"),
            "protocol": protocol_name,
            "config": cnn_config.to_dict(),
            "folds": entry.get("folds", []),
            "status": "running",
            "note": (
                "CNN diuji dengan GroupKFold karena biaya pelatihan CPU; setiap subjek "
                "tetap diuji tepat satu kali dan tidak pernah masuk pelatihan fold-nya."
            ),
        }
    )
    current_fold_ids = {fold.fold for fold in folds}
    entry["folds"] = [record for record in entry["folds"] if int(record["fold"]) in current_fold_ids]
    done_folds = {int(record["fold"]) for record in entry["folds"]} if resume else set()
    for fold in folds:
        if fold.fold in done_folds:
            log.info("CNN GroupKFold fold %d sudah ada, dilewati", fold.fold)
            continue
        _check_partition(dataset, fold.train_idx, fold.test_idx, f"CNN GroupKFold fold {fold.fold}")
        log.info(
            "CNN GroupKFold fold %d/%d (subjek uji %s)",
            fold.fold,
            len(folds),
            fold.test_subjects,
        )
        cnn = train_cnn(
            dataset.raw,
            dataset.y,
            dataset.subject,
            fold.train_idx,
            cnn_config,
            seed=seed + fold.fold,
            class_names=dataset.activity_names,
            logger=log,
        )
        predictions, probabilities, predict_time = predict_cnn(cnn, dataset.raw[fold.test_idx])
        metrics_payload = _metrics_with_timing(
            dataset,
            dataset.y[fold.test_idx],
            predictions,
            fold.test_idx,
            scores=probabilities,
        )
        predictions_path = _save_predictions(
            paths,
            f"cross_user_cnn_fold{fold.fold:02d}",
            dataset.y[fold.test_idx],
            predictions,
            dataset.subject[fold.test_idx],
            probabilities,
        )
        entry["folds"].append(
            {
                "fold": int(fold.fold),
                "test_subjects": fold.test_subjects,
                "train_subjects": fold.train_subjects,
                "n_train": int(len(fold.train_idx)),
                "n_test": int(len(fold.test_idx)),
                "status": "complete",
                "fit_time_s": round(float(cnn.metadata.get("fit_time_s", 0.0)), 4),
                "predict_time_s": round(predict_time, 4),
                "score_kind": "probability",
                "best_epoch": cnn.best_epoch,
                "best_val_macro_f1": cnn.metadata.get("best_val_macro_f1"),
                "predictions_path": predictions_path,
                **metrics_payload,
            }
        )
        entry["folds"].sort(key=lambda item: item["fold"])
        payload["models"]["cnn"] = entry
        payload["protocols"]["cnn"] = f"group_kfold_{n_splits}"
        _persist_cross_user(paths, payload, baseline)

    _finalize_cross_user_model(
        paths, "cnn", entry, dataset.activity_names, f"group_kfold_{n_splits}"
    )
    payload["models"]["cnn"] = entry
    payload["protocols"]["cnn"] = f"group_kfold_{n_splits}"
    _persist_cross_user(paths, payload, baseline)
    payload["status"] = "complete" if all(
        model.get("status") == "complete" for model in payload["models"].values()
    ) else "partial"
    _persist_cross_user(paths, payload, baseline)
    log.info(
        "CNN cross-user selesai: macro-F1 %.4f +/- %.4f",
        entry["aggregate"]["macro_f1"]["mean"],
        entry["aggregate"]["macro_f1"]["std"],
    )
    return payload


# ---------------------------------------------------------------------------
# Experiment D - improvement strategies
# ---------------------------------------------------------------------------
def _normalized_features_for_subject_strategy(
    X: np.ndarray,
    subject: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    target_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_stats = fit_subject_statistics(X[train_idx], subject[train_idx])
    target_stats = fit_subject_statistics(X[target_idx], subject[target_idx])
    X_train = transform_subject_statistics(X[train_idx], subject[train_idx], train_stats)
    X_eval = transform_subject_statistics(
        X[eval_idx],
        subject[eval_idx],
        target_stats,
        fallback=None,
    )
    return X_train, X_eval


def _cnn_normalizer_for_subject_strategy(
    raw: np.ndarray, subject: np.ndarray, train_idx: np.ndarray, target_idx: np.ndarray
) -> RawSignalStandardizer:
    indices = np.concatenate([train_idx, target_idx])
    normalizer = RawSignalStandardizer(mode="per_subject")
    normalizer.fit(raw[indices], subject[indices])
    return normalizer


def run_improvement(
    dataset: HARDataset,
    config: dict[str, Any],
    paths: ProjectPaths,
    models: Iterable[str] | None = None,
    n_splits: int = 6,
    strategies: Iterable[str] | None = None,
    seed: int = 42,
    resume: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or get_logger("har.improvement")
    paths.ensure()
    experiment_cfg = config.get("experiments", {}).get("improvement", {})
    model_names = list(models or experiment_cfg.get("models", ALL_MODELS))
    strategy_names = list(strategies or experiment_cfg.get("strategies", ["base", "subject_norm", "calibration"]))
    if "cnn" in model_names and dataset.raw is None:
        raise ProjectError("Sinyal mentah tidak tersedia; strategi CNN tidak dapat dijalankan")
    fraction = float(experiment_cfg.get("calibration_fraction", 0.1))
    if n_splits > len(dataset.subject_ids):
        n_splits = len(dataset.subject_ids)
    folds = group_kfold_folds(dataset, n_splits=n_splits, seed=seed)
    model_config = _load_model_config(paths)
    cnn_base_config = CNNConfig.from_config(config, seed=seed)
    baseline = _load_baseline(paths)

    existing = _load_existing_results(paths, "improvement.json", dataset, seed)
    if existing and (
        int(existing.get("n_splits", -1)) != int(n_splits)
        or abs(float(existing.get("calibration_fraction", -1)) - fraction) > 1e-9
    ):
        existing = None
    payload = existing or {
        "experiment": "D_improvement",
        "title": "Perbandingan strategi peningkatan generalisasi",
        "seed": int(seed),
        "protocol": f"group_kfold_{n_splits}",
        "n_splits": int(n_splits),
        "calibration_fraction": fraction,
        "created_at": now_iso(),
        "dataset": _compact_dataset(dataset),
        "folds": [fold.to_dict() for fold in folds],
        "strategies": {},
        "notes": {
            "equal_evaluation": (
                "Semua strategi dievaluasi pada subset evaluasi target yang identik; "
                "baris kalibrasi (10%) dikeluarkan dari evaluasi."
            ),
            "transductive_normalisation": STRATEGY_DESCRIPTIONS["subject_norm"],
            "augmentation": "Hanya tersedia untuk CNN karena membutuhkan sinyal mentah.",
        },
    }
    payload["dataset"] = _compact_dataset(dataset)
    payload["folds"] = [fold.to_dict() for fold in folds]

    X = dataset.X
    y = dataset.y
    subject = dataset.subject
    raw = dataset.raw

    for model_name in model_names:
        if model_name not in ALL_MODELS:
            raise ProjectError(f"Model tidak dikenal: {model_name}")
        model_strategies = payload["strategies"].get(model_name, {})
        for strategy in strategy_names:
            if strategy == "augmentation" and model_name != "cnn":
                continue
            if strategy not in {"base", "subject_norm", "calibration", "augmentation"}:
                raise ProjectError(f"Strategi tidak dikenal: {strategy}")
            record = model_strategies.get(
                strategy,
                {
                    "description": STRATEGY_DESCRIPTIONS[strategy],
                    "folds": [],
                    "status": "running",
                },
            )
            done_folds = {int(fold["fold"]) for fold in record["folds"]} if resume else set()
            for fold in folds:
                if fold.fold in done_folds:
                    continue
                cal_idx, eval_idx = calibration_split(
                    fold.test_idx, y, fraction=fraction, seed=seed + fold.fold
                )
                _check_partition(dataset, fold.train_idx, eval_idx, f"Improvement fold {fold.fold}")
                assert_indices_disjoint(cal_idx, eval_idx, "calibration vs evaluation")
                log.info(
                    "Improvement %s | %s | fold %d/%d (eval %d baris)",
                    model_name,
                    strategy,
                    fold.fold,
                    len(folds),
                    len(eval_idx),
                )
                if model_name in SKLEARN_MODELS:
                    frozen = model_config["models"][model_name]["params"]
                    if strategy == "subject_norm":
                        X_train, X_eval = _normalized_features_for_subject_strategy(
                            X, subject, fold.train_idx, eval_idx, fold.test_idx
                        )
                    else:
                        X_train = X[fold.train_idx]
                        X_eval = X[eval_idx]
                    y_train = y[fold.train_idx]
                    train_rows = fold.train_idx
                    if strategy == "calibration":
                        X_train = np.concatenate([X_train, X[cal_idx]], axis=0)
                        y_train = np.concatenate([y_train, y[cal_idx]])
                        train_rows = np.concatenate([train_rows, cal_idx])
                    assert_indices_disjoint(train_rows, eval_idx, "training vs evaluation")
                    from .models import make_sklearn_model

                    model = make_sklearn_model(
                        model_name,
                        frozen,
                        seed=seed + fold.fold,
                        n_jobs=int(config.get("project", {}).get("n_jobs", -1)),
                    )
                    start = time.perf_counter()
                    model.fit(X_train, y_train)
                    fit_time = time.perf_counter() - start
                    predictions, scores, score_kind, predict_time = predict_sklearn(
                        model, X_eval
                    )
                    scores_for_metrics = scores
                else:
                    cnn_config = CNNConfig(**{**cnn_base_config.to_dict(), "seed": seed + fold.fold})
                    train_idx_cnn = fold.train_idx
                    normalizer = None
                    val_idx = None
                    if strategy == "calibration":
                        train_idx_cnn = np.concatenate([fold.train_idx, cal_idx])
                        # Validation must come from source subjects only.
                        _, val_idx, _ = select_validation_subjects(
                            fold.train_idx,
                            subject,
                            fraction=cnn_config.val_subject_fraction,
                            seed=seed + fold.fold,
                        )
                    elif strategy == "subject_norm":
                        normalizer = _cnn_normalizer_for_subject_strategy(
                            raw, subject, fold.train_idx, fold.test_idx
                        )
                        _, val_idx, _ = select_validation_subjects(
                            fold.train_idx,
                            subject,
                            fraction=cnn_config.val_subject_fraction,
                            seed=seed + fold.fold,
                        )
                    if strategy == "augmentation":
                        cnn_config.augment = True
                    assert_indices_disjoint(train_idx_cnn, eval_idx, "CNN training vs evaluation")
                    start = time.perf_counter()
                    cnn = train_cnn(
                        raw,
                        y,
                        subject,
                        train_idx_cnn,
                        cnn_config,
                        seed=seed + fold.fold,
                        val_idx=val_idx,
                        normalizer=normalizer,
                        class_names=dataset.activity_names,
                        logger=log,
                    )
                    fit_time = time.perf_counter() - start
                    predictions, scores, predict_time = predict_cnn(
                        cnn,
                        raw[eval_idx],
                        subjects=subject[eval_idx] if normalizer is not None and normalizer.mode == "per_subject" else None,
                    )
                    scores_for_metrics = scores
                    score_kind = "probability"
                metrics = compute_metrics(
                    y[eval_idx], predictions, dataset.activity_names, y_score=scores_for_metrics
                )
                record["folds"].append(
                    {
                        "fold": int(fold.fold),
                        "test_subjects": fold.test_subjects,
                        "n_train": int(len(train_idx_cnn) if model_name == "cnn" else len(X_train)),
                        "n_eval": int(len(eval_idx)),
                        "n_calibration": int(len(cal_idx)) if strategy == "calibration" else 0,
                        "fit_time_s": round(float(fit_time), 4),
                        "predict_time_s": round(float(predict_time), 4),
                        "score_kind": score_kind,
                        "metrics": metrics,
                    }
                )
                record["folds"].sort(key=lambda item: item["fold"])
                model_strategies[strategy] = record
                payload["strategies"][model_name] = model_strategies
                save_json(paths.results / "improvement.json", payload)

            folds_complete = [f for f in record["folds"] if f.get("status", "complete") == "complete"]
            record["aggregate"] = aggregate_metrics([fold["metrics"] for fold in folds_complete])
            record["n_folds"] = len(folds_complete)
            record["n_eval_total"] = int(sum(fold["n_eval"] for fold in folds_complete))
            record["mean_fit_time_s"] = float(np.mean([f["fit_time_s"] for f in folds_complete])) if folds_complete else None
            record["total_fit_time_s"] = float(np.sum([f["fit_time_s"] for f in folds_complete]))
            record["mean_predict_time_s"] = float(np.mean([f["predict_time_s"] for f in folds_complete])) if folds_complete else None
            record["confusion_matrix"] = _sum_confusions(folds_complete, len(dataset.activity_names))
            record["status"] = "complete" if len(folds_complete) == len(folds) else "running"
            model_strategies[strategy] = record
            payload["strategies"][model_name] = model_strategies
            save_json(paths.results / "improvement.json", payload)

    # Deltas relative to base on identical folds.
    deltas: dict[str, Any] = {}
    for model_name, model_payload in payload["strategies"].items():
        base = model_payload.get("base", {}).get("aggregate")
        if not base:
            continue
        deltas[model_name] = {}
        for strategy, record in model_payload.items():
            aggregate = record.get("aggregate")
            if not aggregate:
                continue
            deltas[model_name][strategy] = {
                metric: (
                    aggregate[metric]["mean"] - base[metric]["mean"]
                    if aggregate.get(metric) and base.get(metric)
                    else None
                )
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
            }
    payload["deltas_vs_base"] = deltas
    payload["status"] = "complete"
    save_json(paths.results / "improvement.json", payload)
    log.info("Improvement selesai untuk %s", ", ".join(payload["strategies"]))
    return payload


# ---------------------------------------------------------------------------
# Utility for report/dashboard
# ---------------------------------------------------------------------------
def load_all_results(paths: ProjectPaths) -> dict[str, Any]:
    return {
        "baseline": read_json(paths.results / "baseline.json"),
        "cross_user": read_json(paths.results / "cross_user.json"),
        "improvement": read_json(paths.results / "improvement.json"),
        "cross_device": read_json(paths.results / "cross_device.json"),
        "dataset_summary": read_json(paths.tables / "dataset_summary.json"),
        "model_config": read_json(paths.results / "model_config.json"),
    }

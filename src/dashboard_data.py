"""Read-only helpers for the Streamlit dashboard.

The dashboard is intentionally a *viewer* for artefacts that already exist on
disk. It never trains models, never fabricates metrics, and never mixes demo
outputs with research outputs. Demo mode reads from ``reports/demo`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data_loader import HARDataset
from .models import MODEL_DISPLAY_NAMES
from .utils import (
    DEMO_NOTICE,
    ArtifactUnavailableError,
    ProjectPaths,
    read_json,
)

RESULT_FILES: dict[str, str] = {
    "baseline": "baseline.json",
    "cross_user": "cross_user.json",
    "improvement": "improvement.json",
    "cross_device": "cross_device.json",
    "model_config": "model_config.json",
    "run_manifest": "run_manifest.json",
}

TABLE_FILES: dict[str, str] = {
    "dataset_summary": "dataset_summary.json",
    "dataset_verification": "dataset_verification.json",
}

FIGURE_CAPTIONS: dict[str, str] = {
    "01_class_distribution.png": "Distribusi kelas aktivitas pada seluruh dataset.",
    "02_samples_per_subject.png": "Jumlah sampel per subjek; tanda * menandai subjek pada split uji resmi.",
    "03_feature_groups.png": "Komposisi 561 fitur UCI HAR menurut kelompok sensor.",
    "04_model_comparison.png": "Perbandingan accuracy, balanced accuracy dan macro-F1 pada split resmi.",
    "05_per_class_f1.png": "F1 per kelas untuk setiap model baseline.",
    "07_baseline_vs_cross_user.png": "Baseline dibandingkan rata-rata evaluasi lintas pengguna.",
    "08_cross_user_per_subject.png": "Macro-F1 untuk setiap subjek yang tidak terlihat saat pelatihan.",
    "09_cross_user_variation.png": "Sebaran variasi performa antarsubjek.",
    "10_improvement_comparison.png": "Perbandingan strategi peningkatan generalisasi.",
    "11_timing.png": "Waktu pelatihan dan inferensi pada CPU (skala log).",
}

MODEL_ORDER = ("random_forest", "svm", "cnn")
STRATEGY_ORDER = ("base", "subject_norm", "calibration", "augmentation")
STRATEGY_LABELS = {
    "base": "Base (tanpa adaptasi)",
    "subject_norm": "Normalisasi per-subjek (tanpa label)",
    "calibration": "Kalibrasi label target",
    "augmentation": "Augmentasi sinyal (CNN)",
}


@dataclass(frozen=True)
class Artifact:
    key: str
    label: str
    path: Path
    exists: bool
    size_bytes: int | None

    @property
    def status_label(self) -> str:
        return "tersedia" if self.exists else "belum ada"


def resolve_paths(config: dict[str, Any], mode: str = "research") -> ProjectPaths:
    if mode not in {"research", "demo"}:
        raise ValueError("mode harus 'research' atau 'demo'")
    return ProjectPaths.from_config(config, mode=mode)


def artifact_inventory(paths: ProjectPaths) -> list[Artifact]:
    """List expected research/demo artefacts without assuming they exist.

    The label is the path relative to the project root, so demo mode clearly
    points at ``reports/demo/...`` instead of the research artefacts.
    """
    def _label(path: Path) -> str:
        try:
            return path.resolve().relative_to(paths.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    entries: list[tuple[str, Path]] = []
    for key, filename in RESULT_FILES.items():
        entries.append((key, paths.results / filename))
    for key, filename in TABLE_FILES.items():
        entries.append((key, paths.tables / filename))
    entries.append(("experiment_summary", paths.reports / "experiment_summary.md"))
    entries.append(("report_manifest", paths.reports / "report_manifest.json"))
    inventory = []
    for key, path in entries:
        exists = path.exists()
        inventory.append(
            Artifact(
                key=key,
                label=_label(path),
                path=path,
                exists=exists,
                size_bytes=path.stat().st_size if exists else None,
            )
        )
    return inventory


def load_bundle(config: dict[str, Any], mode: str = "research") -> dict[str, Any]:
    """Load every available artefact for one mode (research or demo)."""
    paths = resolve_paths(config, mode=mode)
    summary = read_json(paths.tables / "dataset_summary.json")
    if summary is None and mode == "research":
        # Backwards-compatible fallback for runs made before tables/ existed.
        summary = read_json(paths.data_processed / "dataset_summary.json")

    bundle: dict[str, Any] = {
        "mode": mode,
        "paths": paths,
        "is_demo": mode == "demo",
        "dataset_summary": summary,
        "dataset_verification": read_json(paths.tables / "dataset_verification.json"),
        "report_manifest": read_json(paths.reports / "report_manifest.json"),
        "experiment_summary_path": paths.reports / "experiment_summary.md",
        "inventory": artifact_inventory(paths),
    }
    for key, filename in RESULT_FILES.items():
        bundle[key] = read_json(paths.results / filename)

    if mode == "demo" and summary is not None and not summary.get("is_synthetic"):
        raise ArtifactUnavailableError(
            "Folder demo berisi ringkasan data non-sintetis. Hapus reports/demo dan "
            "jalankan ulang mode demo agar hasil tidak tercampur."
        )
    if mode == "research" and summary is not None and summary.get("is_synthetic"):
        raise ArtifactUnavailableError(
            "Ringkasan data penelitian bertanda sintetis. Jalankan ulang pipeline "
            "penelitian pada data UCI HAR."
        )
    return bundle


def has_result(bundle: dict[str, Any], key: str) -> bool:
    payload = bundle.get(key)
    return bool(payload) and isinstance(payload, dict)


def dataset_notice(bundle: dict[str, Any]) -> str | None:
    summary = bundle.get("dataset_summary") or {}
    if summary.get("is_synthetic") or bundle.get("is_demo"):
        return DEMO_NOTICE
    return None


def dataset_facts(bundle: dict[str, Any]) -> dict[str, Any]:
    summary = bundle.get("dataset_summary") or {}
    metadata = summary.get("metadata") or {}
    integrity = summary.get("integrity") or {}
    manifest = bundle.get("run_manifest") or {}
    return {
        "source_label": summary.get("source_label", "Belum tersedia"),
        "n_samples": summary.get("n_samples"),
        "n_features": summary.get("n_features"),
        "n_subjects": summary.get("n_subjects"),
        "n_train": summary.get("n_train"),
        "n_test": summary.get("n_test"),
        "device": metadata.get("device"),
        "sampling_rate_hz": metadata.get("sampling_rate_hz"),
        "window_size": metadata.get("window_size"),
        "window_seconds": metadata.get("window_seconds"),
        "doi": metadata.get("doi"),
        "license": metadata.get("license"),
        "fingerprint": metadata.get("fingerprint"),
        "n_missing_values": integrity.get("n_missing_values"),
        "n_duplicate_rows": integrity.get("n_duplicate_rows"),
        "n_duplicate_feature_names": integrity.get("n_duplicate_feature_names"),
        "raw_signals_available": integrity.get("raw_signals_available"),
        "run_started_at": manifest.get("started_at"),
        "run_finished_at": manifest.get("finished_at"),
        "run_status": manifest.get("status"),
        "seed": manifest.get("seed"),
        "git_commit": (manifest.get("environment") or {}).get("git_commit"),
    }


def experiment_status(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = bundle.get("baseline") or {}
    cross_user = bundle.get("cross_user") or {}
    improvement = bundle.get("improvement") or {}
    cross_device = bundle.get("cross_device") or {}
    cross_models = cross_user.get("models") or {}
    improvement_strategies = improvement.get("strategies") or {}
    return [
        {
            "experiment": "A - Baseline",
            "status": "selesai" if baseline.get("status") == "complete" else "belum dijalankan",
            "detail": ", ".join(
                MODEL_DISPLAY_NAMES.get(name, name) for name in (baseline.get("models") or {})
            )
            or "-",
        },
        {
            "experiment": "B - Lintas pengguna",
            "status": "selesai" if cross_models else "belum dijalankan",
            "detail": ", ".join(
                f"{MODEL_DISPLAY_NAMES.get(name, name)} ({payload.get('protocol', '?')})"
                for name, payload in cross_models.items()
            )
            or "-",
        },
        {
            "experiment": "C - Lintas perangkat",
            "status": cross_device.get("status", "belum dijalankan"),
            "detail": cross_device.get("reason", "-"),
        },
        {
            "experiment": "D - Peningkatan generalisasi",
            "status": "selesai" if improvement_strategies else "belum dijalankan",
            "detail": ", ".join(improvement_strategies) or "-",
        },
    ]


def baseline_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    models = (bundle.get("baseline") or {}).get("models") or {}
    rows = []
    for name in sorted(models, key=lambda value: _model_sort_key(value)):
        payload = models[name]
        metrics = payload.get("metrics") or {}
        rows.append(
            {
                "model": name,
                "model_label": MODEL_DISPLAY_NAMES.get(name, name),
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "cohen_kappa": metrics.get("cohen_kappa"),
                "n_test": metrics.get("n_samples"),
                "fit_time_s": payload.get("fit_time_s"),
                "predict_time_s": payload.get("predict_time_s"),
                "score_kind": payload.get("score_kind"),
            }
        )
    return rows


def baseline_per_class(bundle: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    payload = ((bundle.get("baseline") or {}).get("models") or {}).get(model_name) or {}
    return list((payload.get("metrics") or {}).get("per_class") or [])


def baseline_confusion(bundle: dict[str, Any], model_name: str) -> list[list[int]] | None:
    payload = ((bundle.get("baseline") or {}).get("models") or {}).get(model_name) or {}
    return (payload.get("metrics") or {}).get("confusion_matrix")


def cross_user_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    models = (bundle.get("cross_user") or {}).get("models") or {}
    rows = []
    for name in sorted(models, key=lambda value: _model_sort_key(value)):
        payload = models[name]
        aggregate = payload.get("aggregate") or {}
        per_subject = (payload.get("per_subject_aggregate") or {}).get("macro_f1") or {}
        rows.append(
            {
                "model": name,
                "model_label": MODEL_DISPLAY_NAMES.get(name, name),
                "protocol": payload.get("protocol"),
                "n_folds": payload.get("n_folds"),
                "macro_f1_mean": (aggregate.get("macro_f1") or {}).get("mean"),
                "macro_f1_std": (aggregate.get("macro_f1") or {}).get("std"),
                "balanced_accuracy_mean": (aggregate.get("balanced_accuracy") or {}).get("mean"),
                "accuracy_mean": (aggregate.get("accuracy") or {}).get("mean"),
                "subject_macro_f1_mean": per_subject.get("mean"),
                "subject_macro_f1_std": per_subject.get("std"),
                "subject_macro_f1_min": per_subject.get("min"),
                "subject_macro_f1_max": per_subject.get("max"),
                "mean_fit_time_s": payload.get("mean_fit_time_s"),
            }
        )
    return rows


def cross_user_per_subject(bundle: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    payload = ((bundle.get("cross_user") or {}).get("models") or {}).get(model_name) or {}
    return list(payload.get("per_subject") or [])


def cross_user_folds(bundle: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    payload = ((bundle.get("cross_user") or {}).get("models") or {}).get(model_name) or {}
    return list(payload.get("folds") or [])


def improvement_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    strategies = (bundle.get("improvement") or {}).get("strategies") or {}
    deltas = (bundle.get("improvement") or {}).get("deltas_vs_base") or {}
    rows = []
    for model_name in sorted(strategies, key=lambda value: _model_sort_key(value)):
        model_payload = strategies[model_name]
        for strategy in sorted(
            model_payload, key=lambda value: _strategy_sort_key(value)
        ):
            record = model_payload[strategy]
            aggregate = record.get("aggregate") or {}
            rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                    "macro_f1_mean": (aggregate.get("macro_f1") or {}).get("mean"),
                    "macro_f1_std": (aggregate.get("macro_f1") or {}).get("std"),
                    "balanced_accuracy_mean": (aggregate.get("balanced_accuracy") or {}).get("mean"),
                    "accuracy_mean": (aggregate.get("accuracy") or {}).get("mean"),
                    "delta_macro_f1": (deltas.get(model_name) or {}).get(strategy, {}).get("macro_f1"),
                    "mean_fit_time_s": record.get("mean_fit_time_s"),
                    "n_folds": record.get("n_folds"),
                }
            )
    return rows


def available_figures(bundle: dict[str, Any]) -> list[Path]:
    directory = bundle["paths"].figures
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.png") if path.is_file())


def figure_caption(path: str | Path) -> str:
    name = Path(path).name
    if name in FIGURE_CAPTIONS:
        return FIGURE_CAPTIONS[name]
    if name.startswith("06_confusion_baseline_"):
        model = name.removeprefix("06_confusion_baseline_").removesuffix(".png")
        return f"Confusion matrix baseline untuk {MODEL_DISPLAY_NAMES.get(model, model)}."
    if name.startswith("loso_confusion_"):
        model = name.removeprefix("loso_confusion_").removesuffix(".png")
        return f"Confusion matrix out-of-fold lintas pengguna untuk {MODEL_DISPLAY_NAMES.get(model, model)}."
    return ""


def load_prediction_arrays(paths: ProjectPaths, relative_path: str | Path) -> dict[str, np.ndarray]:
    path = Path(relative_path)
    if not path.is_absolute():
        path = paths.root / path
    if not path.exists():
        raise ArtifactUnavailableError(f"File prediksi tidak ditemukan: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def prediction_path(bundle: dict[str, Any], model_name: str) -> Path | None:
    baseline = (bundle.get("baseline") or {}).get("models") or {}
    model = baseline.get(model_name) or {}
    value = model.get("predictions_path")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else bundle["paths"].root / path


def model_path(bundle: dict[str, Any], model_name: str) -> Path | None:
    baseline = (bundle.get("baseline") or {}).get("models") or {}
    model = baseline.get(model_name) or {}
    value = model.get("model_path")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else bundle["paths"].root / path


def select_test_sample(
    dataset: HARDataset, index: int
) -> tuple[np.ndarray, int, int, str]:
    """Return the feature row, raw window, label and subject for a test row."""
    test_indices = np.flatnonzero(dataset.split == "test")
    if len(test_indices) == 0:
        raise ArtifactUnavailableError("Dataset tidak memiliki partisi test resmi")
    position = int(np.clip(index, 0, len(test_indices) - 1))
    row = int(test_indices[position])
    x = dataset.X[row]
    raw = dataset.raw[row] if dataset.raw is not None else None
    return (
        x,
        int(dataset.y[row]),
        int(dataset.subject[row]),
        raw,
    )


def class_probability_table(
    class_names: list[str], scores: np.ndarray | None, predicted: int
) -> list[dict[str, Any]]:
    if scores is None:
        return []
    values = np.asarray(scores, dtype=float).reshape(-1)
    return [
        {
            "activity": name.replace("_", " ").title(),
            "score": float(values[index]) if index < len(values) else None,
            "predicted": index == int(predicted),
        }
        for index, name in enumerate(class_names)
    ]


def score_kind_note(score_kind: str | None) -> str:
    if score_kind == "normalized_decision_score":
        return (
            "Skor SVM adalah normalized decision score (softmax atas decision function), "
            "bukan probabilitas terkalibrasi."
        )
    if score_kind == "probability":
        return "Skor adalah probabilitas yang dikeluarkan model."
    return "Model tidak menyediakan skor kelas; hanya label prediksi yang tersedia."


def _model_sort_key(name: str) -> tuple[int, str]:
    return (MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER), name)


def _strategy_sort_key(name: str) -> tuple[int, str]:
    return (STRATEGY_ORDER.index(name) if name in STRATEGY_ORDER else len(STRATEGY_ORDER), name)

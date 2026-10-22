"""Publication-ready figures generated from real experiment artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from .models import MODEL_DISPLAY_NAMES  # noqa: E402

MODEL_COLORS = {
    "random_forest": "#2a6f97",
    "svm": "#e07a5f",
    "cnn": "#3a7d44",
}
STRATEGY_COLORS = {
    "base": "#6c757d",
    "subject_norm": "#2a6f97",
    "calibration": "#e09f3e",
    "augmentation": "#3a7d44",
}
STRATEGY_LABELS = {
    "base": "Base (tanpa adaptasi)",
    "subject_norm": "Normalisasi per-subjek (tanpa label)",
    "calibration": "Kalibrasi 10% label target",
    "augmentation": "Augmentasi sinyal (CNN)",
}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.autolayout": False,
        }
    )


def _save(fig: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def _display(name: str) -> str:
    return str(name).replace("_", " ").title()


def _model_label(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)


def _demo_note(summary: dict[str, Any]) -> str:
    return "  [DATA DEMO SINTETIS]" if summary.get("is_synthetic") else ""


def plot_class_distribution(summary: dict[str, Any], path: str | Path) -> Path:
    _setup_style()
    distribution = summary["class_distribution"]
    labels = [_display(item["activity"]) for item in distribution]
    counts = [item["n"] for item in distribution]
    by_split = summary.get("class_distribution_by_split", {})

    fig, axes = plt.subplots(1, 2 if by_split else 1, figsize=(13, 4.8), squeeze=False)
    ax = axes[0, 0]
    bars = ax.bar(labels, counts, color=sns.color_palette("viridis", len(labels)))
    ax.set_title("Distribusi kelas aktivitas" + _demo_note(summary))
    ax.set_ylabel("Jumlah sampel")
    ax.tick_params(axis="x", rotation=25)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if by_split:
        ax2 = axes[0, 1]
        x = np.arange(len(labels))
        width = 0.38
        train_counts = [by_split["train"][item["activity"]] for item in distribution]
        test_counts = [by_split["test"][item["activity"]] for item in distribution]
        ax2.bar(x - width / 2, train_counts, width, label="Train", color="#2a6f97")
        ax2.bar(x + width / 2, test_counts, width, label="Test", color="#e07a5f")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=25)
        ax2.set_title("Distribusi per partisi")
        ax2.set_ylabel("Jumlah sampel")
        ax2.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_samples_per_subject(summary: dict[str, Any], path: str | Path) -> Path:
    _setup_style()
    subjects = summary["subjects"]
    activity_names = [item["activity"] for item in summary["class_distribution"]]
    ids = [f"S{item['subject']}{'*' if item['split'] == 'test' else ''}" for item in subjects]
    palette = sns.color_palette("Set2", len(activity_names))
    fig, ax = plt.subplots(figsize=(13, 5))
    bottoms = np.zeros(len(subjects))
    for index, activity in enumerate(activity_names):
        values = np.asarray([item["activities"][activity] for item in subjects])
        ax.bar(ids, values, bottom=bottoms, label=_display(activity), color=palette[index])
        bottoms += values
    ax.set_title("Jumlah sampel per subjek" + _demo_note(summary))
    ax.set_xlabel("Subjek (* = partisi test resmi)")
    ax.set_ylabel("Jumlah sampel")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_groups(summary: dict[str, Any], path: str | Path) -> Path:
    _setup_style()
    groups = summary.get("feature_groups", {})
    sensor_counts = groups.get("sensor_counts", {}) if isinstance(groups, dict) else {}
    if not sensor_counts:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Ringkasan kelompok fitur tidak tersedia untuk dataset ini.",
            ha="center",
            va="center",
        )
        return _save(fig, path)
    names = list(sensor_counts)
    values = [sensor_counts[name] for name in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh([_display(name) for name in names], values, color="#2a6f97")
    ax.set_title("Komposisi 561 fitur UCI HAR menurut sensor" + _demo_note(summary))
    ax.set_xlabel("Jumlah fitur")
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value}", va="center", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion_matrix(
    confusion: list[list[int]] | np.ndarray,
    labels: list[str],
    path: str | Path,
    title: str,
    normalize: bool = True,
) -> Path:
    _setup_style()
    matrix = np.asarray(confusion, dtype=float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        display = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        fmt = ".2f"
    else:
        display = matrix
        fmt = "d"
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    sns.heatmap(
        display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=normalize,
        xticklabels=[_display(label) for label in labels],
        yticklabels=[_display(label) for label in labels],
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktivitas sebenarnya")
    ax.set_title(title + (" (proporsi per baris)" if normalize else " (jumlah)"))
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return _save(fig, path)


def plot_model_comparison(baseline: dict[str, Any], path: str | Path) -> Path:
    _setup_style()
    models = baseline.get("models", {})
    if not models:
        raise ValueError("Baseline tidak memiliki hasil model")
    names = list(models)
    metric_keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    metric_labels = ["Accuracy", "Balanced accuracy", "Macro-F1"]
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#8ecae6", "#219ebc", "#023047"]
    for index, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = [models[name]["metrics"][key] for name in names]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=colors[index])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(name) for name in names])
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Skor")
    protocol = baseline.get("protocol", "")
    ax.set_title(f"Perbandingan model pada split resmi berbasis subjek{_demo_note(baseline)}\n({protocol})")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    return _save(fig, path)


def plot_per_class_f1(baseline: dict[str, Any], path: str | Path) -> Path:
    _setup_style()
    models = baseline.get("models", {})
    labels = [item["activity"] for item in next(iter(models.values()))["metrics"]["per_class"]]
    rows = []
    for name, payload in models.items():
        rows.append([item["f1"] for item in payload["metrics"]["per_class"]])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.heatmap(
        np.asarray(rows),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        xticklabels=[_display(label) for label in labels],
        yticklabels=[_model_label(name) for name in models],
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("F1 per kelas aktivitas" + _demo_note(baseline))
    ax.set_xlabel("Aktivitas")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save(fig, path)


def plot_cross_user_per_subject(cross_user: dict[str, Any], path: str | Path) -> Path:
    _setup_style()
    models = cross_user.get("models", {})
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, payload in models.items():
        records = payload.get("per_subject", [])
        if not records:
            continue
        records = sorted(records, key=lambda item: item["subject"])
        ax.plot(
            [item["subject"] for item in records],
            [item["macro_f1"] for item in records],
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=_model_label(name),
            color=MODEL_COLORS.get(name, None),
        )
    ax.set_xlabel("Subjek yang diuji (tidak terlihat saat pelatihan)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(sorted({item["subject"] for payload in models.values() for item in payload.get("per_subject", [])}))
    protocol_text = "; ".join(
        f"{_model_label(name)}: {payload.get('protocol', '?')}" for name, payload in models.items()
    )
    ax.set_title(f"Performa lintas pengguna per subjek{_demo_note(cross_user)}")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    ax.text(
        0.0,
        -0.34,
        protocol_text,
        transform=ax.transAxes,
        fontsize=8,
        color="#495057",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_cross_user_variation(cross_user: dict[str, Any], path: str | Path) -> Path:
    _setup_style()
    models = cross_user.get("models", {})
    data = []
    labels = []
    for name, payload in models.items():
        values = [record["macro_f1"] for record in payload.get("per_subject", [])]
        if values:
            data.append(values)
            labels.append(_model_label(name))
    if not data:
        raise ValueError("Data per-subjek tidak tersedia")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.boxplot(data=data, width=0.45, palette="Set2", ax=ax, showfliers=False)
    sns.stripplot(data=data, color="#343a40", size=5, alpha=0.6, ax=ax, jitter=0.08)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-F1 per subjek")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Variasi performa antarsubjek{_demo_note(cross_user)}")
    fig.tight_layout()
    return _save(fig, path)


def plot_baseline_vs_cross_user(
    baseline: dict[str, Any], cross_user: dict[str, Any], path: str | Path
) -> Path:
    _setup_style()
    common = [name for name in baseline.get("models", {}) if name in cross_user.get("models", {})]
    if not common:
        raise ValueError("Tidak ada model yang sama pada baseline dan cross-user")
    baseline_values = [baseline["models"][name]["metrics"]["macro_f1"] for name in common]
    means = [cross_user["models"][name]["aggregate"]["macro_f1"]["mean"] for name in common]
    stds = [cross_user["models"][name]["aggregate"]["macro_f1"]["std"] for name in common]
    x = np.arange(len(common))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, baseline_values, width, label="Baseline (split resmi)", color="#adb5bd")
    bars2 = ax.bar(
        x + width / 2,
        means,
        width,
        yerr=stds,
        capsize=5,
        label="Lintas pengguna (rata-rata fold)",
        color="#2a6f97",
    )
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(name) for name in common])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"Baseline vs generalisasi lintas pengguna{_demo_note(cross_user)}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.tight_layout()
    return _save(fig, path)


def plot_improvement(improvement: dict[str, Any], path: str | Path) -> Path:
    _setup_style()
    strategies = improvement.get("strategies", {})
    models = list(strategies)
    all_strategies = []
    for payload in strategies.values():
        for strategy in payload:
            if strategy not in all_strategies:
                all_strategies.append(strategy)
    x = np.arange(len(models))
    width = 0.8 / max(len(all_strategies), 1)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for index, strategy in enumerate(all_strategies):
        means = []
        stds = []
        positions = []
        for model_index, model in enumerate(models):
            aggregate = strategies[model].get(strategy, {}).get("aggregate", {}).get("macro_f1")
            if aggregate is None:
                continue
            positions.append(model_index + (index - (len(all_strategies) - 1) / 2) * width)
            means.append(aggregate["mean"])
            stds.append(aggregate["std"])
        if not means:
            continue
        bars = ax.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=STRATEGY_LABELS.get(strategy, strategy),
            color=STRATEGY_COLORS.get(strategy),
        )
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(name) for name in models])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro-F1 (rata-rata fold)")
    ax.set_title(
        f"Perbandingan strategi peningkatan generalisasi{_demo_note(improvement)}\n"
        f"GroupKFold({improvement.get('n_splits', '?')}), subset evaluasi sama"
    )
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()
    return _save(fig, path)


def plot_timing(baseline: dict[str, Any], cross_user: dict[str, Any], path: str | Path) -> Path:
    _setup_style()
    baseline_models = baseline.get("models", {})
    cross_models = cross_user.get("models", {})
    common = [name for name in baseline_models if name in cross_models]
    if not common:
        raise ValueError("Tidak ada model untuk grafik waktu")
    baseline_fit = [baseline_models[name]["fit_time_s"] for name in common]
    baseline_predict = [baseline_models[name].get("predict_time_s", 0.0) for name in common]
    cross_fit = [cross_models[name].get("mean_fit_time_s", 0.0) for name in common]
    x = np.arange(len(common))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width, baseline_fit, width, label="Baseline: latih", color="#023047")
    ax.bar(x, baseline_predict, width, label="Baseline: inferensi", color="#8ecae6")
    ax.bar(x + width, cross_fit, width, label="Lintas pengguna: rata-rata latih/fold", color="#e07a5f")
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(name) for name in common])
    ax.set_ylabel("Waktu (detik, skala log)")
    ax.set_yscale("log")
    ax.set_title("Waktu pelatihan dan inferensi" + _demo_note(baseline))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    for model_index, value in enumerate(baseline_fit):
        ax.text(model_index - width, value, f" {value:.1f}s", rotation=0, fontsize=7, va="bottom")
    fig.tight_layout()
    return _save(fig, path)


def plot_cross_user_confusion(
    cross_user: dict[str, Any],
    labels: list[str],
    output_dir: str | Path,
) -> list[Path]:
    paths = []
    for name, payload in cross_user.get("models", {}).items():
        matrix = payload.get("confusion_matrix")
        if not matrix:
            continue
        path = Path(output_dir) / f"loso_confusion_{name}.png"
        paths.append(
            plot_confusion_matrix(
                matrix,
                labels,
                path,
                title=f"Confusion matrix lintas pengguna - {_model_label(name)}",
            )
        )
    return paths

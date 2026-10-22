"""Build report tables, figures and `reports/experiment_summary.md`.

Only artifacts that actually exist are reported. Missing experiments are shown
as "belum dijalankan" with the command needed to run them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.models import MODEL_DISPLAY_NAMES  # noqa: E402
from src.utils import (  # noqa: E402
    DEMO_NOTICE,
    ProjectPaths,
    get_logger,
    load_config,
    now_iso,
    read_json,
    save_json,
)
from src.visualization import (  # noqa: E402
    plot_baseline_vs_cross_user,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_cross_user_confusion,
    plot_cross_user_per_subject,
    plot_cross_user_variation,
    plot_feature_groups,
    plot_improvement,
    plot_model_comparison,
    plot_per_class_f1,
    plot_samples_per_subject,
    plot_timing,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "ya" if value else "tidak"
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _metric(aggregate: dict[str, Any], key: str, kind: str = "mean") -> Any:
    payload = (aggregate or {}).get(key) or {}
    return payload.get(kind)


def _mean_std(aggregate: dict[str, Any], key: str) -> str:
    payload = (aggregate or {}).get(key) or {}
    if payload.get("mean") is None:
        return "—"
    return f"{payload['mean']:.3f} ± {payload.get('std', 0.0):.3f}"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_Tidak ada data._\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _rel_figure(paths: ProjectPaths, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(paths.reports.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _model_label(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)


def _activity_table(metrics: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for item in metrics.get("per_class", []):
        rows.append(
            [
                item["activity"].replace("_", " ").title(),
                f"{item['precision']:.3f}",
                f"{item['recall']:.3f}",
                f"{item['f1']:.3f}",
                item["support"],
            ]
        )
    return rows


def _top_confusions_text(metrics: dict[str, Any], limit: int = 3) -> str:
    pairs = (metrics or {}).get("top_confusions", [])[:limit]
    if not pairs:
        return "Tidak ada kesalahan klasifikasi yang tercatat."
    return "; ".join(
        f"{pair['true'].replace('_', ' ').title()} → {pair['predicted'].replace('_', ' ').title()} "
        f"({pair['count']} kasus, {pair['rate_of_true'] * 100:.1f}% dari kelas asal)"
        for pair in pairs
    )


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------
def build_report(
    config: dict[str, Any],
    paths: ProjectPaths | None = None,
    logger=None,
    mode: str = "research",
) -> dict[str, Any]:
    log = logger or get_logger("har.report")
    paths = paths or ProjectPaths.from_config(config, mode=mode)
    paths.ensure()
    dpi = int(config.get("output", {}).get("figure_dpi", 200))

    summary = read_json(paths.tables / "dataset_summary.json") or read_json(
        paths.data_processed / "dataset_summary.json"
    )
    baseline = read_json(paths.results / "baseline.json")
    cross_user = read_json(paths.results / "cross_user.json")
    improvement = read_json(paths.results / "improvement.json")
    cross_device = read_json(paths.results / "cross_device.json")
    verification = read_json(paths.tables / "dataset_verification.json")
    model_config = read_json(paths.results / "model_config.json")

    if not summary:
        raise FileNotFoundError(
            "dataset_summary.json belum ada. Jalankan `python scripts/prepare_data.py` "
            "atau `python scripts/run_experiments.py --stages baseline` terlebih dahulu."
        )

    figures: list[str] = []
    tables: list[str] = []
    is_demo = bool(summary.get("is_synthetic"))
    labels = [item["activity"] for item in summary.get("class_distribution", [])]

    def figure(path: Path) -> None:
        if path.exists():
            figures.append(_rel_figure(paths, path))

    # Dataset figures
    fig = paths.figures / "01_class_distribution.png"
    if not fig.exists():
        plot_class_distribution(summary, fig)
    figure(fig)
    fig = paths.figures / "02_samples_per_subject.png"
    if not fig.exists():
        plot_samples_per_subject(summary, fig)
    figure(fig)
    fig = paths.figures / "03_feature_groups.png"
    if not fig.exists():
        plot_feature_groups(summary, fig)
    figure(fig)

    # Experiment A figures
    if baseline and baseline.get("models"):
        fig = paths.figures / "04_model_comparison.png"
        plot_model_comparison(baseline, fig)
        figure(fig)
        fig = paths.figures / "05_per_class_f1.png"
        plot_per_class_f1(baseline, fig)
        figure(fig)
        for name, payload in baseline["models"].items():
            fig = paths.figures / f"06_confusion_baseline_{name}.png"
            plot_confusion_matrix(
                payload["metrics"]["confusion_matrix"],
                labels,
                fig,
                title=f"Confusion matrix baseline - {_model_label(name)}",
            )
            figure(fig)

    # Experiment B figures
    if cross_user and cross_user.get("models"):
        if baseline:
            try:
                fig = paths.figures / "07_baseline_vs_cross_user.png"
                plot_baseline_vs_cross_user(baseline, cross_user, fig)
                figure(fig)
            except Exception as exc:  # partial artifacts should not break the report
                log.warning("Grafik baseline vs lintas pengguna dilewati: %s", exc)
        try:
            fig = paths.figures / "08_cross_user_per_subject.png"
            plot_cross_user_per_subject(cross_user, fig)
            figure(fig)
            fig = paths.figures / "09_cross_user_variation.png"
            plot_cross_user_variation(cross_user, fig)
            figure(fig)
            for path in plot_cross_user_confusion(cross_user, labels, paths.figures):
                figure(path)
        except Exception as exc:
            log.warning("Grafik lintas pengguna dilewati: %s", exc)

    # Experiment D figure
    if improvement and improvement.get("strategies"):
        fig = paths.figures / "10_improvement_comparison.png"
        plot_improvement(improvement, fig)
        figure(fig)

    # Timing figure
    if baseline and cross_user:
        try:
            fig = paths.figures / "11_timing.png"
            plot_timing(baseline, cross_user, fig)
            figure(fig)
        except Exception as exc:
            log.warning("Grafik waktu dilewati: %s", exc)

    # ------------------------------------------------------------------
    # CSV tables
    # ------------------------------------------------------------------
    dataset_rows = []
    for subject in summary.get("subjects", []):
        row = {
            "subject": subject["subject"],
            "split": subject["split"],
            "n_samples": subject["n_samples"],
        }
        row.update(subject.get("activities", {}))
        dataset_rows.append(row)
    if dataset_rows:
        path = paths.tables / "dataset_samples_per_subject.csv"
        pd.DataFrame(dataset_rows).to_csv(path, index=False)
        tables.append(str(path))

    path = paths.tables / "dataset_class_distribution.csv"
    pd.DataFrame(summary.get("class_distribution", [])).to_csv(path, index=False)
    tables.append(str(path))

    model_rows = []
    if baseline:
        for name, payload in baseline["models"].items():
            model_rows.append(
                {
                    "model": name,
                    "display_name": _model_label(name),
                    "accuracy": payload["metrics"]["accuracy"],
                    "balanced_accuracy": payload["metrics"]["balanced_accuracy"],
                    "macro_f1": payload["metrics"]["macro_f1"],
                    "weighted_f1": payload["metrics"]["weighted_f1"],
                    "cohen_kappa": payload["metrics"]["cohen_kappa"],
                    "fit_time_s": payload.get("fit_time_s"),
                    "predict_time_s": payload.get("predict_time_s"),
                    "n_train": payload.get("n_train"),
                    "n_test": payload["metrics"]["n_samples"],
                }
            )
    if model_rows:
        path = paths.tables / "model_comparison.csv"
        pd.DataFrame(model_rows).to_csv(path, index=False)
        tables.append(str(path))

    per_class_rows = []
    if baseline:
        for name, payload in baseline["models"].items():
            for item in payload["metrics"]["per_class"]:
                per_class_rows.append({"model": name, **item})
    if per_class_rows:
        path = paths.tables / "per_class_f1.csv"
        pd.DataFrame(per_class_rows).to_csv(path, index=False)
        tables.append(str(path))

    fold_rows = []
    if cross_user:
        for name, payload in cross_user.get("models", {}).items():
            for fold in payload.get("folds", []):
                fold_rows.append(
                    {
                        "model": name,
                        "protocol": payload.get("protocol"),
                        "fold": fold.get("fold"),
                        "test_subjects": ",".join(str(v) for v in fold.get("test_subjects", [])),
                        "n_train": fold.get("n_train"),
                        "n_test": fold.get("n_test"),
                        "accuracy": fold["metrics"]["accuracy"],
                        "balanced_accuracy": fold["metrics"]["balanced_accuracy"],
                        "macro_f1": fold["metrics"]["macro_f1"],
                        "fit_time_s": fold.get("fit_time_s"),
                        "predict_time_s": fold.get("predict_time_s"),
                    }
                )
    if fold_rows:
        path = paths.tables / "cross_user_per_fold.csv"
        pd.DataFrame(fold_rows).to_csv(path, index=False)
        tables.append(str(path))

    per_subject_rows = []
    if cross_user:
        for name, payload in cross_user.get("models", {}).items():
            for record in payload.get("per_subject", []):
                per_subject_rows.append(
                    {
                        "model": name,
                        "protocol": payload.get("protocol"),
                        **record,
                    }
                )
    if per_subject_rows:
        path = paths.tables / "cross_user_per_subject.csv"
        pd.DataFrame(per_subject_rows).to_csv(path, index=False)
        tables.append(str(path))

    improvement_rows = []
    if improvement:
        for name, model_payload in improvement.get("strategies", {}).items():
            base_mean = _metric(
                model_payload.get("base", {}).get("aggregate", {}), "macro_f1", "mean"
            )
            for strategy, record in model_payload.items():
                aggregate = record.get("aggregate", {})
                mean = _metric(aggregate, "macro_f1", "mean")
                improvement_rows.append(
                    {
                        "model": name,
                        "strategy": strategy,
                        "macro_f1_mean": mean,
                        "macro_f1_std": _metric(aggregate, "macro_f1", "std"),
                        "balanced_accuracy_mean": _metric(aggregate, "balanced_accuracy", "mean"),
                        "accuracy_mean": _metric(aggregate, "accuracy", "mean"),
                        "delta_macro_f1_vs_base": (
                            (mean - base_mean) if mean is not None and base_mean is not None else None
                        ),
                        "mean_fit_time_s": record.get("mean_fit_time_s"),
                        "n_folds": record.get("n_folds"),
                    }
                )
    if improvement_rows:
        path = paths.tables / "improvement_comparison.csv"
        pd.DataFrame(improvement_rows).to_csv(path, index=False)
        tables.append(str(path))

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    metadata = summary.get("metadata", {})
    lines: list[str] = []
    lines.append("# Ringkasan Eksperimen\n")
    mode_label = "DEMO SINTETIS" if is_demo else "DATA NYATA (UCI HAR)"
    lines.append(f"**Mode:** {mode_label}  ")
    lines.append(f"**Dibuat:** {now_iso()}  ")
    if is_demo:
        lines.append(f"\n> **PERINGATAN:** {DEMO_NOTICE}\n")
    lines.append(
        f"\nDataset: **{summary.get('source_label', metadata.get('dataset_name', 'unknown'))}** "
        f"dengan {summary['n_samples']:,} sampel, {summary['n_features']} fitur, "
        f"{summary['n_subjects']} subjek.\n".replace(",", ".")
    )

    lines.append("\n## Status eksperimen\n")
    status_rows = [
        [
            "A — Baseline",
            "selesai" if baseline and baseline.get("status") == "complete" else "belum dijalankan",
            ", ".join(_model_label(name) for name in (baseline or {}).get("models", {})) or "—",
        ],
        [
            "B — Lintas pengguna",
            "selesai" if cross_user and cross_user.get("models") else "belum dijalankan",
            ", ".join(
                f"{_model_label(name)} ({payload.get('protocol', '?')})"
                for name, payload in (cross_user or {}).get("models", {}).items()
            )
            or "—",
        ],
        [
            "C — Lintas perangkat",
            (cross_device or {}).get("status", "belum dijalankan"),
            (cross_device or {}).get("reason", "—"),
        ],
        [
            "D — Peningkatan generalisasi",
            "selesai" if improvement and improvement.get("strategies") else "belum dijalankan",
            ", ".join((improvement or {}).get("strategies", {})) or "—",
        ],
    ]
    lines.append(_md_table(["Eksperimen", "Status", "Keterangan"], status_rows))

    lines.append("\n## Dataset dan integritas\n")
    integrity = summary.get("integrity", {})
    dataset_rows = [
        ["Sumber", summary.get("source_label", "—")],
        ["DOI", metadata.get("doi", "—")],
        ["Jumlah sampel", summary["n_samples"]],
        ["Jumlah fitur", summary["n_features"]],
        ["Jumlah subjek", summary["n_subjects"]],
        ["Train / test", f"{summary['n_train']} / {summary['n_test']}"],
        ["Perangkat", metadata.get("device", "—")],
        ["Sampling rate", f"{metadata.get('sampling_rate_hz', '—')} Hz"],
        ["Window", f"{metadata.get('window_size', '—')} sampel ({metadata.get('window_seconds', '—')} s)"],
        ["Overlap", metadata.get("overlap", "—")],
        ["Nilai kosong", integrity.get("n_missing_values", "—")],
        ["Baris duplikat", integrity.get("n_duplicate_rows", "—")],
        ["Nama fitur duplikat (dinamai ulang)", integrity.get("n_duplicate_feature_names", "—")],
        ["Sinyal mentah tersedia", integrity.get("raw_signals_available", "—")],
        ["Lisensi", metadata.get("license", "—")],
    ]
    lines.append(_md_table(["Atribut", "Nilai"], dataset_rows))
    if verification and verification.get("warnings"):
        lines.append("\n**Peringatan verifikasi:**\n")
        for warning in verification["warnings"]:
            lines.append(f"- {warning}")
    lines.append(
        "\n> Fitur 561 dimensi adalah fitur prakomputasi dari pembuat dataset. "
        "Pipeline ini tidak mengekstrak ulang fitur tersebut dari sinyal mentah. "
        "Sinyal mentah 128×9 hanya dipakai oleh 1D-CNN.\n"
    )
    for name in (
        "01_class_distribution.png",
        "02_samples_per_subject.png",
        "03_feature_groups.png",
    ):
        if (paths.figures / name).exists():
            lines.append(f"\n![{name}](figures/{name})\n")

    # Experiment A
    lines.append("\n## Eksperimen A — Baseline pada split resmi\n")
    if baseline:
        split = baseline.get("split", {})
        lines.append(
            f"Protokol: `{baseline.get('protocol')}`. Subjek latih: "
            f"{', '.join(str(v) for v in split.get('train_subjects', []))}; subjek uji: "
            f"{', '.join(str(v) for v in split.get('test_subjects', []))}. "
            "Hyperparameter dipilih dengan GroupKFold pada subjek latih saja.\n"
        )
        rows = []
        for name, payload in baseline["models"].items():
            rows.append(
                [
                    _model_label(name),
                    _fmt(payload["metrics"]["accuracy"]),
                    _fmt(payload["metrics"]["macro_f1"]),
                    _fmt(payload["metrics"]["balanced_accuracy"]),
                    _fmt(payload.get("fit_time_s")),
                    _fmt(payload.get("predict_time_s")),
                ]
            )
        lines.append(
            _md_table(
                ["Model", "Accuracy", "Macro-F1", "Balanced acc.", "Latih (s)", "Inferensi (s)"],
                rows,
            )
        )
        lines.append("\n### F1 per kelas\n")
        header = ["Aktivitas"] + [_model_label(name) for name in baseline["models"]]
        per_class_rows_md = []
        for index, activity in enumerate(labels):
            row = [activity.replace("_", " ").title()]
            for payload in baseline["models"].values():
                row.append(_fmt(payload["metrics"]["per_class"][index]["f1"]))
            per_class_rows_md.append(row)
        lines.append(_md_table(header, per_class_rows_md))
        lines.append("\n### Kesalahan yang paling sering\n")
        for name, payload in baseline["models"].items():
            lines.append(
                f"- **{_model_label(name)}:** {_top_confusions_text(payload['metrics'])}"
            )
        for name in baseline["models"]:
            figure_path = paths.figures / f"06_confusion_baseline_{name}.png"
            if figure_path.exists():
                lines.append(f"\n![Confusion baseline {name}](figures/{figure_path.name})\n")
        if (paths.figures / "04_model_comparison.png").exists():
            lines.append("\n![Perbandingan model](figures/04_model_comparison.png)\n")
        lines.append("\n![F1 per kelas](figures/05_per_class_f1.png)\n")
    else:
        lines.append(
            "_Belum dijalankan._ Jalankan `python scripts/run_experiments.py --stages baseline`.\n"
        )

    # Experiment B
    lines.append("\n## Eksperimen B — Generalisasi lintas pengguna\n")
    if cross_user and cross_user.get("models"):
        rows = []
        for name, payload in cross_user["models"].items():
            aggregate = payload.get("aggregate", {})
            rows.append(
                [
                    _model_label(name),
                    payload.get("protocol", "—"),
                    payload.get("n_folds", "—"),
                    _mean_std(aggregate, "macro_f1"),
                    _mean_std(aggregate, "balanced_accuracy"),
                    _fmt(payload.get("mean_fit_time_s")),
                ]
            )
        lines.append(
            _md_table(
                ["Model", "Protokol", "Fold", "Macro-F1", "Balanced acc.", "Latih/fold (s)"],
                rows,
            )
        )
        lines.append(
            "\n> RF/SVM memakai Leave-One-Subject-Out (30 fold). CNN memakai GroupKFold(6) "
            "karena biaya pelatihan CPU; setiap subjek tetap diuji tepat sekali dan tidak "
            "pernah masuk pelatihan fold-nya. Baseline memakai split resmi 9 subjek uji; "
            "selisih baseline vs lintas pengguna tidak sepenuhnya sebanding karena protokol "
            "partisipan berbeda.\n"
        )
        gap = (cross_user.get("gap_vs_baseline") or {}).get("models", {})
        if gap:
            gap_rows = []
            for name, record in gap.items():
                gap_rows.append(
                    [
                        _model_label(name),
                        _fmt(record["baseline_macro_f1"]),
                        _fmt(record["cross_user_macro_f1_mean"]),
                        _fmt(record["absolute_gap"]),
                        _fmt(record.get("relative_drop_pct"), 1) + "%",
                    ]
                )
            lines.append("\n### Selisih terhadap baseline\n")
            lines.append(
                _md_table(
                    ["Model", "Baseline macro-F1", "Lintas pengguna macro-F1", "Selisih", "Penurunan relatif"],
                    gap_rows,
                )
            )
        if per_subject_rows:
            lines.append("\n### Variasi performa per subjek\n")
            variation_rows = []
            for name, payload in cross_user["models"].items():
                aggregate = payload.get("per_subject_aggregate", {}).get("macro_f1", {})
                variation_rows.append(
                    [
                        _model_label(name),
                        _fmt(aggregate.get("mean")),
                        _fmt(aggregate.get("std")),
                        _fmt(aggregate.get("min")),
                        _fmt(aggregate.get("max")),
                        aggregate.get("n", "—"),
                    ]
                )
            lines.append(
                _md_table(
                    ["Model", "Rata-rata macro-F1", "Std", "Minimum", "Maksimum", "Subjek"],
                    variation_rows,
                )
            )
        lines.append("\n### Kelas yang sering tertukar (out-of-fold)\n")
        for name, payload in cross_user["models"].items():
            lines.append(
                f"- **{_model_label(name)}:** {_top_confusions_text(payload.get('oof_metrics') or payload.get('metrics', {}))}"
            )
        for name in ("08_cross_user_per_subject.png", "09_cross_user_variation.png", "07_baseline_vs_cross_user.png"):
            if (paths.figures / name).exists():
                lines.append(f"\n![{name}](figures/{name})\n")
        for name in cross_user["models"]:
            if (paths.figures / f"loso_confusion_{name}.png").exists():
                lines.append(f"\n![Confusion lintas pengguna {name}](figures/loso_confusion_{name}.png)\n")
    else:
        lines.append(
            "_Belum dijalankan._ Jalankan `python scripts/run_experiments.py --stages loso,group-cv`.\n"
        )

    # Experiment C
    lines.append("\n## Eksperimen C — Generalisasi lintas perangkat\n")
    if cross_device:
        if cross_device.get("status") == "complete":
            lines.append(
                f"Protokol `{cross_device.get('protocol')}` pada data mandiri. "
                "Hasil ini tidak berasal dari UCI HAR.\n"
            )
            rows = []
            for name, payload in cross_device.get("models", {}).items():
                aggregate = payload.get("aggregate", {})
                rows.append(
                    [
                        _model_label(name),
                        _mean_std(aggregate, "macro_f1"),
                        _mean_std(aggregate, "balanced_accuracy"),
                        payload.get("n_folds", len(payload.get("folds", []))),
                    ]
                )
            lines.append(_md_table(["Model", "Macro-F1", "Balanced acc.", "Perangkat"], rows))
        else:
            lines.append(f"**Status: `{cross_device.get('status')}`**\n")
            lines.append(f"Alasan: {cross_device.get('reason')}\n")
            lines.append("Kebutuhan data untuk menjalankan eksperimen ini:\n")
            for requirement in cross_device.get("requirements", []):
                lines.append(f"- {requirement}")
            lines.append(
                f"\nProsedur dan skema manifest: `{cross_device.get('documentation', 'docs/cross_device_protocol.md')}`.\n"
            )
    else:
        lines.append(
            "_Belum dijalankan._ Jalankan `python scripts/run_experiments.py --stages cross-device` "
            "atau sediakan manifest multi-perangkat.\n"
        )

    # Experiment D
    lines.append("\n## Eksperimen D — Strategi peningkatan generalisasi\n")
    if improvement and improvement.get("strategies"):
        lines.append(
            f"Protokol: `{improvement.get('protocol')}`; fraksi kalibrasi "
            f"{improvement.get('calibration_fraction', 0) * 100:.0f}%; semua strategi "
            "dievaluasi pada subset target yang sama.\n"
        )
        rows = []
        for name, model_payload in improvement["strategies"].items():
            for strategy, record in model_payload.items():
                aggregate = record.get("aggregate", {})
                rows.append(
                    [
                        _model_label(name),
                        strategy,
                        _mean_std(aggregate, "macro_f1"),
                        _mean_std(aggregate, "balanced_accuracy"),
                        _fmt(record.get("mean_fit_time_s")),
                    ]
                )
        lines.append(
            _md_table(
                ["Model", "Strategi", "Macro-F1", "Balanced acc.", "Latih/fold (s)"],
                rows,
            )
        )
        delta_rows = []
        for name, deltas in (improvement.get("deltas_vs_base") or {}).items():
            for strategy, values in deltas.items():
                delta_rows.append(
                    [
                        _model_label(name),
                        strategy,
                        _fmt(values.get("macro_f1")),
                        _fmt(values.get("balanced_accuracy")),
                    ]
                )
        if delta_rows:
            lines.append("\n### Perubahan relatif terhadap base\n")
            lines.append(_md_table(["Model", "Strategi", "Δ Macro-F1", "Δ Balanced acc."], delta_rows))
        lines.append(
            "\nCatatan: normalisasi per-subjek memakai data target tanpa label "
            "(transductive). Kalibrasi memakai sebagian label target dan mengevaluasi sisa "
            "data target. Augmentasi hanya tersedia untuk CNN.\n"
        )
        if (paths.figures / "10_improvement_comparison.png").exists():
            lines.append("\n![Perbandingan strategi](figures/10_improvement_comparison.png)\n")
    else:
        lines.append(
            "_Belum dijalankan._ Jalankan `python scripts/run_experiments.py --stages improvement`.\n"
        )

    # Timing
    if baseline:
        lines.append("\n## Waktu komputasi\n")
        timing_rows = []
        for name, payload in baseline["models"].items():
            timing_rows.append(
                [
                    _model_label(name),
                    _fmt(payload.get("fit_time_s")),
                    _fmt(payload.get("predict_time_s")),
                    _fmt((cross_user or {}).get("models", {}).get(name, {}).get("mean_fit_time_s")),
                ]
            )
        lines.append(
            _md_table(
                ["Model", "Latih baseline (s)", "Inferensi baseline (s)", "Rata-rata latih/fold lintas pengguna (s)"],
                timing_rows,
            )
        )
        lines.append(
            "\nWaktu diukur pada CPU dan dapat berbeda antar mesin; angka ini bukan "
            "bagian dari klaim performa model.\n"
        )
        if (paths.figures / "11_timing.png").exists():
            lines.append("\n![Waktu pelatihan dan inferensi](figures/11_timing.png)\n")

    # Limitations
    lines.append("\n## Keterbatasan metodologis\n")
    limitations = [
        "Dataset UCI HAR hanya berisi satu perangkat (Samsung Galaxy S II), sehingga tidak ada uji generalisasi lintas perangkat pada data nyata.",
        "561 fitur UCI adalah fitur prakomputasi; pipeline tidak mengekstrak ulang fitur tersebut dari sinyal mentah. 1D-CNN memakai sinyal mentah 128×9.",
        "Hyperparameter RF/SVM dipilih sekali dengan GroupKFold pada subjek latih split resmi, lalu dibekukan untuk seluruh fold LOSO. Pemilihan nested per fold tidak dilakukan karena biaya komputasi; hal ini dapat memberi sedikit optimisme pada beberapa fold.",
        "CNN dievaluasi dengan GroupKFold(6), bukan 30 fold LOSO, karena keterbatasan CPU. Setiap subjek tetap diuji tepat sekali.",
        "Normalisasi per-subjek dan kalibrasi memerlukan data target (tanpa label untuk normalisasi, sebagian berlabel untuk kalibrasi); ini bukan skenario zero-shot murni.",
        "Hasil demo sintetis tidak boleh dilaporkan sebagai temuan penelitian.",
    ]
    for item in limitations:
        lines.append(f"- {item}")
    lines.append("\n## Cara mereproduksi\n")
    lines.append("```bash\npython scripts/prepare_data.py --download\npython scripts/run_experiments.py\npython scripts/generate_report.py\npytest -q\n```\n")

    summary_path = paths.reports / "experiment_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "generated_at": now_iso(),
        "mode": mode,
        "is_synthetic": is_demo,
        "summary_path": str(summary_path),
        "figures": figures,
        "tables": tables,
        "experiments": {
            "baseline": bool(baseline),
            "cross_user": bool(cross_user),
            "improvement": bool(improvement),
            "cross_device": (cross_device or {}).get("status"),
        },
        "counts": {
            "n_samples": summary["n_samples"],
            "n_features": summary["n_features"],
            "n_subjects": summary["n_subjects"],
        },
    }
    save_json(paths.reports / "report_manifest.json", manifest)
    log.info("Laporan ditulis ke %s (%d figur, %d tabel)", summary_path, len(figures), len(tables))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    mode = "demo" if args.demo else "research"
    paths = ProjectPaths.from_config(config, mode=mode)
    build_report(config, paths, mode=mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

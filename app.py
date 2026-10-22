"""Streamlit dashboard for the smartphone HAR generalisation study.

The dashboard only displays artefacts that already exist in ``reports/``.
Nothing is trained here and nothing is invented: every number comes from the
JSON/CSV files produced by ``scripts/run_experiments.py``.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dashboard_data import (  # noqa: E402
    artifact_inventory,
    baseline_confusion,
    baseline_per_class,
    baseline_rows,
    class_probability_table,
    cross_user_folds,
    cross_user_per_subject,
    cross_user_rows,
    dataset_facts,
    dataset_notice,
    experiment_status,
    figure_caption,
    improvement_rows,
    load_bundle,
    load_prediction_arrays,
    model_path,
    prediction_path,
    resolve_paths,
    score_kind_note,
    select_test_sample,
)
from src.utils import DEMO_NOTICE, ArtifactUnavailableError, load_config  # noqa: E402

st.set_page_config(page_title="Generalisasi HAR Smartphone", page_icon="📱", layout="wide")

PAGE_TITLES = {
    "Ringkasan": "Ringkasan dan status eksperimen",
    "Dataset": "Dataset, eksplorasi dan integritas",
    "Baseline": "Eksperimen A - baseline pada split resmi",
    "Lintas Pengguna": "Eksperimen B - generalisasi lintas pengguna",
    "Strategi": "Eksperimen D & C - strategi peningkatan dan lintas perangkat",
}


@st.cache_data(show_spinner="Memuat artefak...")
def get_bundle(config_path: str, mode: str) -> dict[str, Any]:
    config = load_config(config_path)
    return load_bundle(config, mode=mode)


@st.cache_data(show_spinner="Memuat dataset...")
def get_dataset(config_path: str, mode: str):
    from src.data_loader import load_dataset, load_demo_dataset

    config = load_config(config_path)
    paths = resolve_paths(config, mode=mode)
    if mode == "demo":
        # Demo mode must never touch the real UCI HAR data.
        return load_demo_dataset(paths, config)
    return load_dataset(config, paths=paths, use_cache=True, allow_demo_fallback=False)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "ya" if value else "tidak"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def metric_card(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def render_figure(bundle: dict[str, Any], filename: str) -> None:
    path = bundle["paths"].figures / filename
    if not path.exists():
        return
    caption = figure_caption(filename)
    st.image(str(path), caption=caption if caption else None)


def render_dataframe(rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if not rows:
        st.info("Tidak ada data untuk ditampilkan.")
        return
    frame = pd.DataFrame(rows)
    if columns:
        frame = frame[[column for column in columns if column in frame.columns]]
    st.dataframe(frame, use_container_width=True, hide_index=True)


def missing_state(command: str) -> None:
    st.warning(
        "Artefak untuk bagian ini belum ada. Jalankan perintah berikut terlebih dahulu:\n\n"
        f"```bash\n{command}\n```"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📱 HAR Generalisasi")
    config_path = str(PROJECT_ROOT / "config" / "config.yaml")
    mode_label = st.radio(
        "Sumber data",
        options=["Penelitian (UCI HAR)", "Demo sintetis"],
        index=0,
        help=(
            "Mode demo hanya menampilkan artefak di reports/demo dan diberi penanda "
            "DATA DEMO SINTETIS. Hasil demo bukan temuan penelitian."
        ),
    )
    mode = "demo" if mode_label.startswith("Demo") else "research"
    page = st.radio(
        "Halaman",
        options=list(PAGE_TITLES),
        index=0,
    )
    st.divider()
    st.caption("Python 3.11 · CPU-only · scikit-learn + PyTorch")

try:
    bundle = get_bundle(config_path, mode)
except ArtifactUnavailableError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # pragma: no cover - defensive UI guard
    st.error(f"Gagal memuat artefak: {type(exc).__name__}: {exc}")
    st.stop()

notice = dataset_notice(bundle)
if notice:
    st.warning(notice)

facts = dataset_facts(bundle)
with st.sidebar:
    st.markdown("**Artefak dimuat**")
    inventory = artifact_inventory(bundle["paths"])
    for artifact in inventory:
        icon = "✅" if artifact.exists else "⭕"
        st.markdown(f"- {icon} `{artifact.label}`")


# ---------------------------------------------------------------------------
# Page 1 - Ringkasan
# ---------------------------------------------------------------------------
if page == "Ringkasan":
    st.title(PAGE_TITLES[page])
    st.caption(
        "Semua angka pada halaman ini dibaca dari berkas hasil di folder reports/. "
        "Dashboard tidak menghitung ulang model."
    )

    if not bundle.get("dataset_summary"):
        missing_state("python scripts/prepare_data.py --download")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sampel", f"{facts['n_samples']:,}".replace(",", ".") if facts["n_samples"] else "-")
        col2.metric("Fitur", facts["n_features"] or "-")
        col3.metric("Subjek", facts["n_subjects"] or "-")
        col4.metric("Perangkat", facts["device"] or "-")

        st.subheader("Status eksperimen")
        render_dataframe(experiment_status(bundle))

        st.subheader("Ringkasan performa")
        baseline_table = baseline_rows(bundle)
        if baseline_table:
            render_dataframe(
                [
                    {
                        "Model": row["model_label"],
                        "Accuracy": fmt(row["accuracy"]),
                        "Balanced acc.": fmt(row["balanced_accuracy"]),
                        "Macro-F1": fmt(row["macro_f1"]),
                        "Waktu latih (s)": fmt(row["fit_time_s"], 1),
                    }
                    for row in baseline_table
                ]
            )
            st.caption(
                "Baseline memakai split resmi UCI (subjek latih dan subjek uji berbeda). "
                "Nilai lintas pengguna ada di halaman 'Lintas Pengguna'."
            )
        else:
            missing_state("python scripts/run_experiments.py --stages baseline")

        cross_table = cross_user_rows(bundle)
        if cross_table:
            st.subheader("Ringkasan lintas pengguna")
            render_dataframe(
                [
                    {
                        "Model": row["model_label"],
                        "Protokol": row["protocol"],
                        "Fold": row["n_folds"],
                        "Macro-F1 (rata-rata ± std)": f"{fmt(row['macro_f1_mean'])} ± {fmt(row['macro_f1_std'])}",
                        "Macro-F1 min subjek": fmt(row["subject_macro_f1_min"]),
                        "Macro-F1 max subjek": fmt(row["subject_macro_f1_max"]),
                    }
                    for row in cross_table
                ]
            )

    st.subheader("Provenance eksekusi")
    provenance = {
        "Status run terakhir": facts["run_status"] or "-",
        "Mulai": facts["run_started_at"] or "-",
        "Selesai": facts["run_finished_at"] or "-",
        "Seed": facts["seed"] if facts["seed"] is not None else "-",
        "Git commit": facts["git_commit"] or "- (repository belum di-commit)",
        "Fingerprint dataset": (facts["fingerprint"] or "-")[:16],
        "Mode": "demo sintetis" if mode == "demo" else "penelitian (UCI HAR)",
    }
    render_dataframe([provenance])

    summary_path = bundle["experiment_summary_path"]
    if summary_path.exists():
        st.subheader("Ringkasan laporan")
        st.download_button(
            "Unduh reports/experiment_summary.md",
            data=summary_path.read_text(encoding="utf-8"),
            file_name="experiment_summary.md",
            mime="text/markdown",
        )


# ---------------------------------------------------------------------------
# Page 2 - Dataset
# ---------------------------------------------------------------------------
elif page == "Dataset":
    st.title(PAGE_TITLES[page])
    summary = bundle.get("dataset_summary")
    if not summary:
        missing_state("python scripts/prepare_data.py --download")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Sampel train (resmi)", facts["n_train"] or "-")
        col2.metric("Sampel test (resmi)", facts["n_test"] or "-")
        col3.metric("Sampling rate", f"{facts['sampling_rate_hz']} Hz" if facts["sampling_rate_hz"] else "-")

        st.subheader("Integritas data")
        render_dataframe(
            [
                {"Atribut": "Sumber", "Nilai": facts["source_label"]},
                {"Atribut": "DOI", "Nilai": facts["doi"] or "-"},
                {"Atribut": "Perangkat", "Nilai": facts["device"] or "-"},
                {"Atribut": "Window", "Nilai": f"{facts['window_size']} sampel ({facts['window_seconds']} s)" if facts["window_size"] else "-"},
                {"Atribut": "Nilai kosong", "Nilai": facts["n_missing_values"]},
                {"Atribut": "Baris duplikat", "Nilai": facts["n_duplicate_rows"]},
                {"Atribut": "Nama fitur duplikat (dinamai unik)", "Nilai": facts["n_duplicate_feature_names"]},
                {"Atribut": "Sinyal mentah tersedia", "Nilai": facts["raw_signals_available"]},
                {"Atribut": "Lisensi", "Nilai": facts["license"] or "-"},
            ]
        )

        st.subheader("Distribusi kelas")
        class_rows = []
        for item in summary.get("class_distribution", []):
            class_rows.append(
                {
                    "Aktivitas": item["activity"].replace("_", " ").title(),
                    "Jumlah": item["n"],
                    "Persentase (%)": item["pct"],
                }
            )
        render_dataframe(class_rows)

        st.subheader("Visualisasi")
        for name in ("01_class_distribution.png", "02_samples_per_subject.png", "03_feature_groups.png"):
            render_figure(bundle, name)

        with st.expander("Verifikasi struktur dataset"):
            verification = bundle.get("dataset_verification")
            if verification:
                render_dataframe(
                    [
                        {"Pemeriksaan": key, "Nilai": str(value)}
                        for key, value in (verification.get("checks") or {}).items()
                        if not isinstance(value, (dict, list))
                    ]
                )
                for warning in verification.get("warnings", []):
                    st.warning(warning)
            else:
                st.info("Berkas verifikasi belum tersedia.")


# ---------------------------------------------------------------------------
# Page 3 - Baseline
# ---------------------------------------------------------------------------
elif page == "Baseline":
    st.title(PAGE_TITLES[page])
    baseline = bundle.get("baseline")
    if not baseline or not baseline.get("models"):
        missing_state("python scripts/run_experiments.py --stages baseline")
    else:
        split = baseline.get("split") or {}
        st.caption(
            f"Protokol `{baseline.get('protocol')}` · subjek latih "
            f"{split.get('train_subjects')} · subjek uji {split.get('test_subjects')} · "
            f"{split.get('n_train')} sampel latih vs {split.get('n_test')} sampel uji."
        )

        rows = baseline_rows(bundle)
        cols = st.columns(len(rows))
        for column, row in zip(cols, rows):
            with column:
                metric_card(row["model_label"], fmt(row["macro_f1"]), "Macro-F1 pada split resmi")
                st.caption(
                    f"Accuracy {fmt(row['accuracy'])} · Balanced acc. {fmt(row['balanced_accuracy'])} "
                    f"· latih {fmt(row['fit_time_s'], 1)} s"
                )

        st.subheader("Tabel metrik")
        render_dataframe(
            [
                {
                    "Model": row["model_label"],
                    "Accuracy": fmt(row["accuracy"]),
                    "Balanced acc.": fmt(row["balanced_accuracy"]),
                    "Macro-F1": fmt(row["macro_f1"]),
                    "Weighted F1": fmt(row["weighted_f1"]),
                    "Kappa": fmt(row["cohen_kappa"]),
                    "Sampel uji": row["n_test"],
                    "Latih (s)": fmt(row["fit_time_s"], 2),
                    "Inferensi (s)": fmt(row["predict_time_s"], 3),
                }
                for row in rows
            ]
        )

        render_figure(bundle, "04_model_comparison.png")
        render_figure(bundle, "05_per_class_f1.png")

        model_names = [row["model"] for row in rows]
        label_for = {row["model"]: row["model_label"] for row in rows}
        selected = st.selectbox("Model", model_names, format_func=lambda value: label_for[value])
        st.subheader(f"F1 per kelas - {label_for[selected]}")
        per_class = baseline_per_class(bundle, selected)
        render_dataframe(
            [
                {
                    "Aktivitas": item["activity"].replace("_", " ").title(),
                    "Precision": fmt(item["precision"]),
                    "Recall": fmt(item["recall"]),
                    "F1": fmt(item["f1"]),
                    "Support": item["support"],
                }
                for item in per_class
            ]
        )

        st.subheader("Confusion matrix")
        confusion = baseline_confusion(bundle, selected)
        if confusion:
            class_labels = [
                item["activity"].replace("_", " ").title()
                for item in ((bundle.get("dataset_summary") or {}).get("class_distribution") or [])
            ]
            if not class_labels:
                class_labels = [f"Class {index}" for index in range(len(confusion))]
            frame = pd.DataFrame(confusion, index=class_labels, columns=class_labels)
            st.dataframe(frame, use_container_width=True)
        render_figure(bundle, f"06_confusion_baseline_{selected}.png")

        st.subheader("Uji prediksi pada sampel uji")
        st.caption(
            "Bagian ini memakai model baseline yang sudah dilatih dan satu baris dari "
            "partisi test resmi. Tidak ada model yang dilatih ulang di dashboard."
        )
        try:
            dataset = get_dataset(config_path, mode)
        except Exception as exc:
            st.info(
                "Dataset tidak dapat dimuat untuk pratinjau prediksi "
                f"({type(exc).__name__}: {exc}). Jalankan `python scripts/prepare_data.py --download`."
            )
        else:
            test_indices = int((dataset.split == "test").sum())
            position = st.number_input(
                "Indeks sampel pada partisi test (0-based)",
                min_value=0,
                max_value=max(test_indices - 1, 0),
                value=0,
                step=1,
            )
            x_row, true_label, subject_id, raw_row = select_test_sample(dataset, int(position))
            st.write(
                f"Subjek: **{subject_id}** · Aktivitas sebenarnya: "
                f"**{dataset.activity_names[true_label].replace('_', ' ').title()}**"
            )
            if st.button("Jalankan prediksi", type="primary"):
                for name in model_names:
                    model_file = model_path(bundle, name)
                    if model_file is None or not model_file.exists():
                        st.warning(f"Model {name} tidak tersimpan; lewati.")
                        continue
                    with st.expander(f"Prediksi - {label_for[name]}"):
                        try:
                            if name == "cnn":
                                from src.train import load_cnn_model, predict_cnn

                                wrapper = load_cnn_model(model_file)
                                predictions, probabilities, elapsed = predict_cnn(wrapper, raw_row[None, ...])
                                predicted = int(predictions[0])
                                score_kind = "probability"
                            else:
                                from src.train import load_sklearn_model, predict_sklearn

                                model, _ = load_sklearn_model(model_file)
                                predictions, scores, score_kind, elapsed = predict_sklearn(model, x_row[None, ...])
                                predicted = int(predictions[0])
                                probabilities = scores
                            correct = "benar" if predicted == true_label else "salah"
                            st.write(
                                f"Prediksi: **{dataset.activity_names[predicted].replace('_', ' ').title()}** ({correct}) "
                                f"· inferensi {elapsed:.4f} s"
                            )
                            st.caption(score_kind_note(score_kind))
                            probability_table = class_probability_table(
                                dataset.activity_names, probabilities, predicted
                            )
                            if probability_table:
                                render_dataframe(probability_table)
                        except Exception as exc:
                            st.error(f"Gagal memuat model {name}: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Page 4 - Lintas Pengguna
# ---------------------------------------------------------------------------
elif page == "Lintas Pengguna":
    st.title(PAGE_TITLES[page])
    cross_user = bundle.get("cross_user")
    if not cross_user or not cross_user.get("models"):
        missing_state("python scripts/run_experiments.py --stages loso,group-cv")
    else:
        st.caption(
            "RF/SVM diuji dengan Leave-One-Subject-Out; CNN diuji dengan GroupKFold "
            "karena biaya pelatihan CPU. Pada kedua protokol setiap subjek diuji tepat "
            "sekali dan tidak pernah masuk pelatihan fold-nya."
        )
        rows = cross_user_rows(bundle)
        render_dataframe(
            [
                {
                    "Model": row["model_label"],
                    "Protokol": row["protocol"],
                    "Fold": row["n_folds"],
                    "Macro-F1": f"{fmt(row['macro_f1_mean'])} ± {fmt(row['macro_f1_std'])}",
                    "Balanced acc.": fmt(row["balanced_accuracy_mean"]),
                    "Accuracy": fmt(row["accuracy_mean"]),
                    "Macro-F1 subjek min": fmt(row["subject_macro_f1_min"]),
                    "Macro-F1 subjek max": fmt(row["subject_macro_f1_max"]),
                    "Latih/fold (s)": fmt(row["mean_fit_time_s"], 1),
                }
                for row in rows
            ]
        )

        gap = (cross_user.get("gap_vs_baseline") or {}).get("models") or {}
        if gap:
            st.subheader("Selisih terhadap baseline")
            st.caption(cross_user.get("gap_vs_baseline", {}).get("note", ""))
            render_dataframe(
                [
                    {
                        "Model": name,
                        "Baseline macro-F1": fmt(record["baseline_macro_f1"]),
                        "Lintas pengguna macro-F1": fmt(record["cross_user_macro_f1_mean"]),
                        "Selisih absolut": fmt(record["absolute_gap"]),
                        "Penurunan relatif (%)": fmt(record.get("relative_drop_pct"), 1),
                    }
                    for name, record in gap.items()
                ]
            )

        render_figure(bundle, "07_baseline_vs_cross_user.png")
        render_figure(bundle, "08_cross_user_per_subject.png")
        render_figure(bundle, "09_cross_user_variation.png")

        model_names = [row["model"] for row in rows]
        label_for = {row["model"]: row["model_label"] for row in rows}
        selected = st.selectbox(
            "Model",
            model_names,
            format_func=lambda value: label_for[value],
        )
        st.subheader(f"Variasi per subjek - {label_for[selected]}")
        per_subject = cross_user_per_subject(bundle, selected)
        render_dataframe(
            [
                {
                    "Subjek": record["subject"],
                    "Sampel": record["n_samples"],
                    "Kelas hadir": record["n_classes_present"],
                    "Accuracy": fmt(record["accuracy"]),
                    "Balanced acc.": fmt(record["balanced_accuracy"]),
                    "Macro-F1": fmt(record["macro_f1"]),
                }
                for record in per_subject
            ]
        )

        with st.expander("Rincian per fold"):
            folds = cross_user_folds(bundle, selected)
            render_dataframe(
                [
                    {
                        "Fold": fold.get("fold"),
                        "Subjek uji": ", ".join(str(value) for value in fold.get("test_subjects", [])),
                        "n latih": fold.get("n_train"),
                        "n uji": fold.get("n_test"),
                        "Accuracy": fmt((fold.get("metrics") or {}).get("accuracy")),
                        "Macro-F1": fmt((fold.get("metrics") or {}).get("macro_f1")),
                        "Latih (s)": fmt(fold.get("fit_time_s"), 2),
                    }
                    for fold in folds
                ]
            )

        render_figure(bundle, f"loso_confusion_{selected}.png")

        with st.expander("Pratinjau prediksi out-of-fold"):
            relative = None
            for fold in cross_user_folds(bundle, selected):
                if fold.get("predictions_path"):
                    relative = fold["predictions_path"]
                    break
            if relative:
                arrays = load_prediction_arrays(bundle["paths"], relative)
                frame = pd.DataFrame(
                    {
                        "y_true": arrays["y_true"],
                        "y_pred": arrays["y_pred"],
                        "subjek": arrays["subject"],
                    }
                )
                st.caption(f"Cuplikan fold pertama ({relative}).")
                st.dataframe(frame.head(200), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 5 - Strategi (D) dan lintas perangkat (C)
# ---------------------------------------------------------------------------
elif page == "Strategi":
    st.title(PAGE_TITLES[page])
    improvement = bundle.get("improvement")
    if not improvement or not improvement.get("strategies"):
        missing_state("python scripts/run_experiments.py --stages improvement")
    else:
        st.caption(
            f"Protokol `{improvement.get('protocol')}` · fraksi kalibrasi "
            f"{(improvement.get('calibration_fraction') or 0) * 100:.0f}% · seluruh strategi "
            "dievaluasi pada subset target yang identik."
        )
        rows = improvement_rows(bundle)
        render_dataframe(
            [
                {
                    "Model": row["model_label"],
                    "Strategi": row["strategy_label"],
                    "Macro-F1": f"{fmt(row['macro_f1_mean'])} ± {fmt(row['macro_f1_std'])}",
                    "Balanced acc.": fmt(row["balanced_accuracy_mean"]),
                    "Δ Macro-F1 vs base": fmt(row["delta_macro_f1"]),
                    "Fold": row["n_folds"],
                    "Latih/fold (s)": fmt(row["mean_fit_time_s"], 1),
                }
                for row in rows
            ]
        )
        st.caption(
            "Normalisasi per-subjek memakai data target tanpa label (transductive). "
            "Kalibrasi memakai sebagian label target dan mengevaluasi sisa data target. "
            "Augmentasi hanya tersedia untuk CNN."
        )
        render_figure(bundle, "10_improvement_comparison.png")

    st.divider()
    st.subheader("Eksperimen C - generalisasi lintas perangkat")
    cross_device = bundle.get("cross_device")
    if not cross_device:
        missing_state("python scripts/run_experiments.py --stages cross-device")
    elif cross_device.get("status") == "complete":
        st.caption("Data mandiri multi-perangkat; bukan bagian dari dataset UCI HAR.")
        render_dataframe(
            [
                {
                    "Model": name,
                    "Macro-F1": fmt(
                        ((payload.get("aggregate") or {}).get("macro_f1") or {}).get("mean")
                    ),
                    "Balanced acc.": fmt(
                        ((payload.get("aggregate") or {}).get("balanced_accuracy") or {}).get("mean")
                    ),
                    "Fold": payload.get("n_folds"),
                }
                for name, payload in (cross_device.get("models") or {}).items()
            ]
        )
    else:
        st.info(
            f"Status: **{cross_device.get('status', 'tidak dijalankan')}**\n\n"
            f"Alasan: {cross_device.get('reason', '-')}"
        )
        requirements = cross_device.get("requirements") or []
        if requirements:
            st.markdown("**Kebutuhan data untuk menjalankan eksperimen ini:**")
            for requirement in requirements:
                st.markdown(f"- {requirement}")
        st.markdown(
            f"Prosedur lengkap: `{cross_device.get('documentation', 'docs/cross_device_protocol.md')}`"
        )

    st.divider()
    st.subheader("Semua artefak pada mode ini")
    render_dataframe(
        [
            {
                "Artefak": artifact.label,
                "Status": artifact.status_label,
                "Ukuran (byte)": artifact.size_bytes if artifact.size_bytes is not None else "-",
            }
            for artifact in artifact_inventory(bundle["paths"])
        ]
    )

st.divider()
st.caption(
    "Dashboard ini menampilkan hasil yang benar-benar ada di folder reports/. "
    "Tidak ada metrik yang dihitung ulang atau diasumsikan."
)

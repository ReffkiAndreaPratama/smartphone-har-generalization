"""Fill the README results section from the artefacts in ``reports/results``.

The README never contains hand-written numbers: this script renders the
"Hasil Eksperimen" section from the JSON artefacts produced by
``scripts/run_experiments.py`` and writes it back into ``README.md``.

Usage
-----
    python scripts/update_readme_results.py            # research mode
    python scripts/update_readme_results.py --demo     # demo artefacts (labelled)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ProjectPaths, load_config, read_json  # noqa: E402

START_MARKER = "<!-- BEGIN-HASIL-EKSPERIMEN -->"
END_MARKER = "<!-- END-HASIL-EKSPERIMEN -->"

MODEL_ORDER = ["random_forest", "svm", "cnn"]
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "svm": "SVM (RBF)",
    "cnn": "1D-CNN",
}
STRATEGY_LABELS = {
    "base": "base",
    "subject_norm": "normalisasi per-subjek",
    "calibration": "kalibrasi 10 % label target",
    "augmentation": "augmentasi sinyal",
}
SCORE_KIND = {
    "probability": "probabilitas",
    "normalized_decision_score": "decision score ternormalisasi (bukan probabilitas)",
}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "ya" if value else "tidak"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f} %"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _baseline_section(baseline: dict[str, Any]) -> str:
    if not baseline or not baseline.get("models"):
        return "_Artefak `reports/results/baseline.json` belum ada._"
    split = baseline.get("split", {})
    rows = []
    for name in MODEL_ORDER:
        payload = baseline["models"].get(name)
        if not payload:
            continue
        metrics = payload["metrics"]
        rows.append(
            [
                MODEL_LABELS.get(name, name),
                _fmt(metrics["accuracy"]),
                _fmt(metrics["balanced_accuracy"]),
                _fmt(metrics["macro_f1"]),
                _fmt(metrics["weighted_f1"]),
                _fmt(metrics["cohen_kappa"]),
                _fmt(payload.get("fit_time_s"), 2),
                _fmt(payload.get("predict_time_s"), 3),
                SCORE_KIND.get(payload.get("score_kind", ""), payload.get("score_kind", "-")),
            ]
        )
    table = _table(
        [
            "Model",
            "Accuracy",
            "Balanced acc.",
            "Macro-F1",
            "Weighted F1",
            "Kappa",
            "Latih (s)",
            "Inferensi (s)",
            "Jenis skor",
        ],
        rows,
    )
    caption = (
        f"Protokol `{baseline.get('protocol')}` — latih {split.get('n_train')} sampel dari "
        f"{len(split.get('train_subjects', []))} subjek, uji {split.get('n_test')} sampel dari "
        f"{len(split.get('test_subjects', []))} subjek. "
        f"Guardrail: {split.get('guardrail')}"
    )
    return f"{caption}\n\n{table}"


def _per_class_section(baseline: dict[str, Any]) -> str:
    if not baseline or not baseline.get("models"):
        return ""
    header = ["Aktivitas"] + [
        f"{MODEL_LABELS.get(name, name)} (P / R / F1)"
        for name in MODEL_ORDER
        if name in baseline["models"]
    ]
    activities: list[str] = []
    for name in MODEL_ORDER:
        payload = baseline["models"].get(name)
        if not payload:
            continue
        for item in payload["metrics"].get("per_class", []):
            if item["activity"] not in activities:
                activities.append(item["activity"])
    rows = []
    for activity in activities:
        row = [activity.replace("_", " ").title()]
        for name in MODEL_ORDER:
            payload = baseline["models"].get(name)
            if not payload:
                continue
            item = next(
                (x for x in payload["metrics"].get("per_class", []) if x["activity"] == activity),
                None,
            )
            if item is None:
                row.append("-")
            else:
                row.append(
                    f"{_fmt(item['precision'], 3)} / {_fmt(item['recall'], 3)} / {_fmt(item['f1'], 3)}"
                )
        rows.append(row)
    return _table(header, rows)


def _cross_user_section(cross_user: dict[str, Any]) -> str:
    if not cross_user or not cross_user.get("models"):
        return "_Artefak `reports/results/cross_user.json` belum ada._"
    rows = []
    for name in MODEL_ORDER:
        payload = cross_user["models"].get(name)
        if not payload or payload.get("status") != "complete":
            continue
        aggregate = payload["aggregate"]
        per_subject = payload.get("per_subject_aggregate", {}).get("macro_f1", {})
        rows.append(
            [
                MODEL_LABELS.get(name, name),
                payload.get("protocol"),
                str(payload.get("n_folds")),
                f"{_fmt(aggregate['macro_f1']['mean'])} ± {_fmt(aggregate['macro_f1']['std'])}",
                _fmt(aggregate["accuracy"]["mean"]),
                _fmt(aggregate["balanced_accuracy"]["mean"]),
                _fmt(per_subject.get("min")),
                _fmt(per_subject.get("max")),
                _fmt(payload.get("mean_fit_time_s"), 2),
            ]
        )
    table = _table(
        [
            "Model",
            "Protokol",
            "Fold",
            "Macro-F1 (mean ± std)",
            "Accuracy",
            "Balanced acc.",
            "Macro-F1 subjek min",
            "Macro-F1 subjek max",
            "Latih/fold (s)",
        ],
        rows,
    )
    gap = (cross_user.get("gap_vs_baseline") or {}).get("models") or {}
    gap_rows = []
    for name, record in gap.items():
        gap_rows.append(
            [
                MODEL_LABELS.get(name, name),
                _fmt(record["baseline_macro_f1"]),
                _fmt(record["cross_user_macro_f1_mean"]),
                _fmt(record["absolute_gap"]),
                _pct(record.get("relative_drop_pct")),
            ]
        )
    gap_table = (
        _table(
            [
                "Model",
                "Baseline macro-F1",
                "Lintas pengguna macro-F1",
                "Selisih absolut",
                "Selisih relatif",
            ],
            gap_rows,
        )
        if gap_rows
        else ""
    )
    note = (cross_user.get("gap_vs_baseline") or {}).get("note", "")
    blocks = [table]
    if gap_table:
        blocks.append(
            "**Selisih terhadap baseline**\n\n"
            + gap_table
            + "\n\n> "
            + note
            + "\n> Tanda positif berarti performa lintas pengguna lebih rendah daripada "
            "baseline. Perhatikan bahwa LOSO melatih model pada 29 subjek (bukan 21 subjek "
            "pada split resmi), sehingga selisih ini **bukan** ukuran degradasi murni."
        )
    hardest = []
    for name in MODEL_ORDER:
        payload = cross_user["models"].get(name)
        if not payload or payload.get("status") != "complete":
            continue
        per_subject = sorted(payload.get("per_subject", []), key=lambda item: item["macro_f1"])
        if not per_subject:
            continue
        hardest.append(
            f"- **{MODEL_LABELS.get(name, name)}** — subjek terlemah: "
            f"S{int(per_subject[0]['subject'])} (macro-F1 {_fmt(per_subject[0]['macro_f1'])}); "
            f"terkuat: S{int(per_subject[-1]['subject'])} (macro-F1 {_fmt(per_subject[-1]['macro_f1'])})."
        )
    if hardest:
        blocks.append("**Variasi per subjek (fold lintas pengguna)**\n\n" + "\n".join(hardest))
    return "\n\n".join(blocks)


def _improvement_section(improvement: dict[str, Any]) -> str:
    if not improvement or not improvement.get("strategies"):
        return "_Artefak `reports/results/improvement.json` belum ada._"
    rows = []
    deltas = improvement.get("deltas_vs_base", {})
    for name in MODEL_ORDER:
        strategies = improvement["strategies"].get(name)
        if not strategies:
            continue
        for strategy in ("base", "subject_norm", "calibration", "augmentation"):
            record = strategies.get(strategy)
            if not record or record.get("status") != "complete":
                continue
            aggregate = record["aggregate"]
            delta = (deltas.get(name, {}) or {}).get(strategy, {})
            rows.append(
                [
                    MODEL_LABELS.get(name, name),
                    STRATEGY_LABELS.get(strategy, strategy),
                    f"{_fmt(aggregate['macro_f1']['mean'])} ± {_fmt(aggregate['macro_f1']['std'])}",
                    _fmt(delta.get("macro_f1")),
                    _fmt(aggregate["balanced_accuracy"]["mean"]),
                    _fmt(aggregate["accuracy"]["mean"]),
                    str(record.get("n_folds")),
                    _fmt(record.get("mean_fit_time_s"), 2),
                ]
            )
    table = _table(
        [
            "Model",
            "Strategi",
            "Macro-F1 (mean ± std)",
            "Δ Macro-F1 vs base",
            "Balanced acc.",
            "Accuracy",
            "Fold",
            "Latih/fold (s)",
        ],
        rows,
    )
    notes = improvement.get("notes", {})
    blocks = [table]
    if notes:
        bullets = "\n".join(f"- **{key.replace('_', ' ')}** — {value}" for key, value in notes.items())
        blocks.append("**Catatan protokol**\n\n" + bullets)
    return "\n\n".join(blocks)


def _cross_device_section(cross_device: dict[str, Any] | None) -> str:
    if not cross_device:
        return "_Artefak `reports/results/cross_device.json` belum ada._"
    status = cross_device.get("status")
    if status == "complete":
        rows = []
        for name, payload in (cross_device.get("models") or {}).items():
            aggregate = payload.get("aggregate", {})
            rows.append(
                [
                    MODEL_LABELS.get(name, name),
                    _fmt(aggregate.get("macro_f1", {}).get("mean")),
                    _fmt(aggregate.get("balanced_accuracy", {}).get("mean")),
                    str(payload.get("n_folds")),
                ]
            )
        return "Status: **complete**\n\n" + _table(
            ["Model", "Macro-F1", "Balanced acc.", "Fold"], rows
        )
    lines = [
        "Status: **"
        + str(status)
        + "** (tidak dihitung, bukan angka yang dipalsukan).",
        "",
        f"**Alasan:** {cross_device.get('reason', '-')}",
    ]
    requirements = cross_device.get("requirements") or []
    if requirements:
        lines.append("")
        lines.append("**Kebutuhan data untuk menjalankannya:**")
        lines.extend(f"- {item}" for item in requirements)
    devices = cross_device.get("available_devices")
    if devices:
        lines.append("")
        lines.append(f"**Perangkat yang tersedia pada dataset:** {', '.join(devices)}")
    documentation = cross_device.get("documentation")
    if documentation:
        lines.append("")
        lines.append(f"Prosedur lengkap: [`{documentation}`]({documentation})")
    return "\n".join(lines)


def _interpretation(
    baseline: dict[str, Any] | None,
    cross_user: dict[str, Any] | None,
    improvement: dict[str, Any] | None,
) -> str:
    bullets: list[str] = []
    if baseline and baseline.get("models"):
        best = max(
            (name for name in MODEL_ORDER if name in baseline["models"]),
            key=lambda name: baseline["models"][name]["metrics"]["macro_f1"],
        )
        worst = min(
            (name for name in MODEL_ORDER if name in baseline["models"]),
            key=lambda name: baseline["models"][name]["metrics"]["macro_f1"],
        )
        best_metrics = baseline["models"][best]["metrics"]
        bullets.append(
            f"- **Eksperimen A.** Macro-F1 tertinggi pada split resmi: "
            f"{MODEL_LABELS.get(best, best)} ({_fmt(best_metrics['macro_f1'])}, "
            f"accuracy {_fmt(best_metrics['accuracy'])}); terendah: "
            f"{MODEL_LABELS.get(worst, worst)} "
            f"({_fmt(baseline['models'][worst]['metrics']['macro_f1'])})."
        )
    if cross_user and cross_user.get("models"):
        for name in MODEL_ORDER:
            payload = cross_user["models"].get(name)
            if not payload or payload.get("status") != "complete":
                continue
            aggregate = payload["aggregate"]["macro_f1"]
            per_subject = payload.get("per_subject_aggregate", {}).get("macro_f1", {})
            bullets.append(
                f"- **Eksperimen B ({MODEL_LABELS.get(name, name)}).** Rata-rata macro-F1 "
                f"lintas pengguna {_fmt(aggregate['mean'])} ± {_fmt(aggregate['std'])} "
                f"dengan rentang antar-subjek {_fmt(per_subject.get('min'))}–"
                f"{_fmt(per_subject.get('max'))} pada {payload.get('n_folds')} fold "
                f"({payload.get('protocol')})."
            )
        gap = (cross_user.get("gap_vs_baseline") or {}).get("models") or {}
        for name, record in gap.items():
            bullets.append(
                f"- Selisih baseline → lintas pengguna untuk {MODEL_LABELS.get(name, name)}: "
                f"{_fmt(record['absolute_gap'])} absolut "
                f"({_pct(record.get('relative_drop_pct'))} relatif). Karena LOSO melatih "
                "model pada 29 subjek sedangkan split resmi hanya 21 subjek, selisih ini "
                "perbandingan antar protokol, bukan bukti degradasi generalisasi."
            )
    if improvement and improvement.get("strategies"):
        for name in MODEL_ORDER:
            strategies = improvement["strategies"].get(name)
            if not strategies:
                continue
            deltas = (improvement.get("deltas_vs_base", {}) or {}).get(name, {})
            parts = []
            for strategy in ("subject_norm", "calibration", "augmentation"):
                delta = deltas.get(strategy)
                if delta and delta.get("macro_f1") is not None:
                    parts.append(f"{STRATEGY_LABELS.get(strategy, strategy)} {delta['macro_f1']:+.4f}")
            if parts:
                bullets.append(
                    f"- **Eksperimen D ({MODEL_LABELS.get(name, name)}).** Δ macro-F1 terhadap "
                    f"`base`: {', '.join(parts)}."
                )
    bullets.append(
        "- **Eksperimen C.** Tidak dapat dihitung dari UCI HAR karena hanya ada satu "
        "perangkat; status `not_performed` beserta alasannya tercatat di "
        "`reports/results/cross_device.json`."
    )
    return "\n".join(bullets)


def render_section(paths: ProjectPaths) -> str:
    baseline = read_json(paths.results / "baseline.json")
    cross_user = read_json(paths.results / "cross_user.json")
    improvement = read_json(paths.results / "improvement.json")
    cross_device = read_json(paths.results / "cross_device.json")
    manifest = read_json(paths.results / "run_manifest.json") or {}

    demo_banner = ""
    if (paths.results / "baseline.json").exists():
        payload = baseline or {}
        if payload.get("dataset", {}).get("is_synthetic"):
            demo_banner = (
                "> ⚠️ **DATA DEMO SINTETIS.** Angka pada bagian ini berasal dari data demo "
                "yang dibuat generator deterministik, bukan UCI HAR. Jangan dilaporkan "
                "sebagai hasil penelitian.\n\n"
            )

    environment = manifest.get("environment") or {}
    packages = environment.get("packages") or {}
    commit = environment.get("git_commit")
    provenance_rows = [
        ["Run id", str(manifest.get("run_id", "-"))],
        ["Status run", str(manifest.get("status", "-"))],
        ["Mode", str(manifest.get("mode", "-"))],
        ["Seed", str(manifest.get("seed", "-"))],
        ["Mulai", str(manifest.get("started_at", "-"))],
        ["Selesai", str(manifest.get("finished_at", "-"))],
        ["Durasi total", f"{float(manifest.get('elapsed_s', 0)) / 60:.1f} menit"],
        [
            "Git commit",
            str(commit) if commit else "null (run dieksekusi sebelum repository diinisialisasi)",
        ],
        ["Python", str(environment.get("python", "-"))],
        ["Platform", f"{environment.get('platform', '-')} / {environment.get('processor', '-')}"],
        [
            "Versi pustaka",
            ", ".join(
                f"{name} {packages.get(name, '-')}"
                for name in ("numpy", "scikit-learn", "torch", "streamlit")
                if name in packages
            ),
        ],
    ]
    stage_rows = []
    for stage, record in (manifest.get("stage_results") or {}).items():
        stage_rows.append(
            [
                stage,
                str(record.get("status")),
                f"{float(record.get('duration_s', 0)) / 60:.1f} menit",
            ]
        )

    parts = [
        demo_banner,
        "Angka di bawah ini dibaca langsung dari berkas JSON pada `reports/results/` "
        "yang dihasilkan `python scripts/run_experiments.py`. Laporan lengkap dengan "
        "tautan figur tersedia di [`reports/experiment_summary.md`](reports/experiment_summary.md).",
        "",
        "### Provenance eksekusi",
        "",
        _table(["Properti", "Nilai"], provenance_rows),
        "",
        _table(["Tahap", "Status", "Durasi"], stage_rows),
        "",
        "### Eksperimen A — Baseline pada split resmi",
        "",
        _baseline_section(baseline),
        "",
        "**F1 per kelas pada partisi uji resmi** (precision / recall / F1)",
        "",
        _per_class_section(baseline),
        "",
        "### Eksperimen B — Generalisasi lintas pengguna",
        "",
        _cross_user_section(cross_user),
        "",
        "### Eksperimen D — Strategi peningkatan generalisasi",
        "",
        _improvement_section(improvement),
        "",
        "### Eksperimen C — Generalisasi lintas perangkat",
        "",
        _cross_device_section(cross_device),
        "",
        "### Interpretasi singkat",
        "",
        _interpretation(baseline, cross_user, improvement),
        "",
        "### Figur pendukung",
        "",
        "| Figur | Isi |",
        "| --- | --- |",
        "| `reports/figures/01_class_distribution.png` | Distribusi kelas |",
        "| `reports/figures/02_samples_per_subject.png` | Jumlah sampel per subjek |",
        "| `reports/figures/04_model_comparison.png` | Perbandingan model pada baseline |",
        "| `reports/figures/05_per_class_f1.png` | F1 per kelas |",
        "| `reports/figures/07_baseline_vs_cross_user.png` | Baseline vs lintas pengguna |",
        "| `reports/figures/08_cross_user_per_subject.png` | Performa per subjek |",
        "| `reports/figures/10_improvement_comparison.png` | Perbandingan strategi |",
        "| `reports/figures/11_timing.png` | Waktu komputasi |",
    ]
    return "\n".join(part for part in parts if part is not None).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--demo", action="store_true", help="Isi dari artefak demo")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = ProjectPaths.from_config(config, mode="demo" if args.demo else "research")
    readme_path = Path(args.readme)
    if not readme_path.is_absolute():
        readme_path = PROJECT_ROOT / readme_path
    text = readme_path.read_text(encoding="utf-8")
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise SystemExit(
            "README harus memuat tepat satu penanda BEGIN-HASIL-EKSPERIMEN dan "
            "satu penanda END-HASIL-EKSPERIMEN"
        )
    section = render_section(paths)
    head, rest = text.split(START_MARKER, 1)
    _, tail = rest.split(END_MARKER, 1)
    manifest = read_json(paths.results / "run_manifest.json") or {}
    elapsed = float(manifest.get("elapsed_s", 0) or 0)
    duration = f"{elapsed / 60:.0f} menit"
    section = section.replace("TOTAL_WAKTU", duration)
    tail = tail.replace("TOTAL_WAKTU", duration)
    readme_path.write_text(f"{head}{START_MARKER}\n\n{section}\n{END_MARKER}{tail}", encoding="utf-8")
    print(f"README diperbarui: {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

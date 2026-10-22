"""Run the experiment suite from the command line.

Examples
--------
python scripts/run_experiments.py --download
python scripts/run_experiments.py --stages baseline,loso,group-cv,improvement
python scripts/run_experiments.py --demo --quick
python scripts/run_experiments.py --stages cross-device
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    build_dataset_summary,
    cache_dataset,
    download_uci_har,
    find_dataset_dir,
    load_demo_dataset,
    load_uci_har,
    save_dataset_summary,
    verify_uci_structure,
)
from src.generalization import (  # noqa: E402
    run_baseline,
    run_group_cv_cnn,
    run_improvement,
    run_loso,
)
from src.cross_device import (  # noqa: E402
    CrossDeviceDataUnavailable,
    load_multi_device_dataset,
    run_cross_device_experiment,
    write_not_performed_result,
)
from src.utils import (  # noqa: E402
    DEMO_NOTICE,
    ProjectPaths,
    environment_info,
    get_logger,
    load_config,
    now_iso,
    save_json,
    set_seed,
)

STAGES = ["baseline", "loso", "group-cv", "improvement", "cross-device"]
ARTIFACTS = {
    "baseline": ["reports/results/baseline.json", "reports/results/model_config.json"],
    "loso": ["reports/results/cross_user.json"],
    "group-cv": ["reports/results/cross_user.json"],
    "improvement": ["reports/results/improvement.json"],
    "cross-device": ["reports/results/cross_device.json"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--stages",
        default="all",
        help="Daftar tahap dipisah koma: " + ", ".join(STAGES) + ", atau all",
    )
    parser.add_argument("--models", default=None, help="Model dipisah koma (default dari config)")
    parser.add_argument("--demo", action="store_true", help="Jalankan pada data demo sintetis")
    parser.add_argument("--quick", action="store_true", help="Konfigurasi cepat untuk smoke test")
    parser.add_argument("--download", action="store_true", help="Unduh dataset UCI bila belum ada")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-loso-folds", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--cnn-epochs", type=int, default=None)
    parser.add_argument("--calibration-fraction", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Latih ulang semua fold (timpa hasil parsial)")
    parser.add_argument("--fresh", action="store_true", help="Hapus artefak hasil pada mode ini terlebih dahulu")
    parser.add_argument("--no-report", action="store_true", help="Jangan membuat laporan/figur setelah eksperimen")
    return parser.parse_args()


def _resolve_stages(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(STAGES)
    stages = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in stages if item not in STAGES]
    if unknown:
        raise SystemExit(f"Tahap tidak dikenal: {unknown}. Pilihan: {STAGES}")
    return stages


def _apply_quick(config: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.quick:
        return
    config.setdefault("models", {}).setdefault("random_forest", {})["n_estimators"] = 60
    config["models"].setdefault("cnn", {})["max_epochs"] = 3
    config["models"]["cnn"]["patience"] = 2
    config["models"]["cnn"]["batch_size"] = 64
    config.setdefault("experiments", {}).setdefault("improvement", {})["n_splits"] = 2
    config["experiments"].setdefault("group_cv_cnn", {})["n_splits"] = 2
    config["experiments"]["improvement"]["strategies"] = ["base", "subject_norm", "calibration"]
    if args.max_loso_folds is None:
        args.max_loso_folds = 2
    if args.n_splits is None:
        args.n_splits = 2


def _prepare_dataset(config: dict[str, Any], args: argparse.Namespace, paths: ProjectPaths, logger):
    if args.demo:
        logger.warning(DEMO_NOTICE)
        return load_demo_dataset(paths, config)
    dataset_dir = find_dataset_dir(paths.data_raw)
    if dataset_dir is None and args.download:
        dataset_dir = download_uci_har(paths, config, force=args.fresh, logger=logger)
    if dataset_dir is None:
        raise SystemExit(
            "Dataset UCI HAR belum tersedia. Tambahkan --download, atau jalankan --demo "
            "untuk smoke test (hasil demo tidak boleh dilaporkan sebagai hasil penelitian)."
        )
    verification = verify_uci_structure(
        dataset_dir, expected=config.get("dataset", {}).get("expected", {})
    )
    for warning in verification.get("warnings", []):
        logger.warning(warning)
    save_json(paths.tables / "dataset_verification.json", verification)
    dataset = load_uci_har(dataset_dir, load_raw=True)
    # Keep the processed cache in sync with the fingerprint of the raw files
    # that were just verified. ``load_dataset`` validates this fingerprint.
    cache_dataset(dataset, paths)
    return dataset


def _clear_results(paths: ProjectPaths) -> None:
    for name in ("baseline.json", "cross_user.json", "improvement.json", "model_config.json", "cross_device.json"):
        (paths.results / name).unlink(missing_ok=True)
    if paths.predictions.exists():
        for path in paths.predictions.glob("*.npz"):
            path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    stages = _resolve_stages(args.stages)
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    set_seed(seed)
    if args.quick:
        _apply_quick(config, args)
    mode = "demo" if args.demo else "research"
    paths = ProjectPaths.from_config(config, mode=mode).ensure()
    logger = get_logger("har.run")
    if args.fresh:
        _clear_results(paths)
        logger.info("Artefak hasil lama pada mode %s dihapus", mode)

    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "run_id": now_iso(),
        "mode": mode,
        "seed": seed,
        "quick": bool(args.quick),
        "stages_requested": stages,
        "config_path": config.get("_config_path"),
        "environment": environment_info(),
        "stage_results": {},
        "status": "running",
        "started_at": now_iso(),
    }

    def persist_manifest() -> None:
        manifest["elapsed_s"] = round(time.perf_counter() - started, 3)
        manifest["finished_at"] = now_iso()
        save_json(paths.results / "run_manifest.json", manifest)

    try:
        dataset = _prepare_dataset(config, args, paths, logger)
        if dataset.is_synthetic and not args.demo:
            raise SystemExit("Dataset fallback sintetis tidak diizinkan pada run penelitian.")
        manifest["dataset"] = {
            "source": dataset.source,
            "is_synthetic": dataset.is_synthetic,
            "n_samples": dataset.n_samples,
            "n_subjects": len(dataset.subject_ids),
            "notice": DEMO_NOTICE if dataset.is_synthetic else None,
        }
        summary = build_dataset_summary(
            dataset, expected=config.get("dataset", {}).get("expected", {}) if not dataset.is_synthetic else None
        )
        save_dataset_summary(summary, paths)
        persist_manifest()

        def run_stage(name: str, function) -> None:
            if name not in stages:
                return
            logger.info("=== Tahap %s ===", name)
            stage_start = time.perf_counter()
            try:
                result = function()
                duration = round(time.perf_counter() - stage_start, 3)
                manifest["stage_results"][name] = {
                    "status": "complete",
                    "duration_s": duration,
                    "artifacts": ARTIFACTS.get(name, []),
                }
                logger.info("Tahap %s selesai dalam %.1f detik", name, duration)
                return result
            except Exception as exc:
                manifest["stage_results"][name] = {
                    "status": "failed",
                    "duration_s": round(time.perf_counter() - stage_start, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                manifest["status"] = "failed"
                persist_manifest()
                raise
            finally:
                persist_manifest()

        baseline_models = config.get("experiments", {}).get("baseline", {}).get("models", ["random_forest", "svm", "cnn"])
        loso_models = config.get("experiments", {}).get("loso", {}).get("models", ["random_forest", "svm"])
        improvement_models = config.get("experiments", {}).get("improvement", {}).get("models", ["random_forest", "svm", "cnn"])
        if args.models:
            selected = [item.strip() for item in args.models.split(",") if item.strip()]
            baseline_models = [name for name in baseline_models if name in selected]
            loso_models = [name for name in loso_models if name in selected]
            improvement_models = [name for name in improvement_models if name in selected]

        run_stage(
            "baseline",
            lambda: run_baseline(dataset, config, paths, models=baseline_models, seed=seed, logger=logger),
        )
        run_stage(
            "loso",
            lambda: run_loso(
                dataset,
                config,
                paths,
                models=loso_models,
                max_folds=args.max_loso_folds,
                seed=seed,
                resume=not args.no_resume,
                logger=logger,
            ),
        )
        n_splits = int(
            args.n_splits
            or config.get("experiments", {}).get("group_cv_cnn", {}).get("n_splits", 6)
        )
        run_stage(
            "group-cv",
            lambda: run_group_cv_cnn(
                dataset,
                config,
                paths,
                n_splits=n_splits,
                seed=seed,
                max_folds=args.n_splits,
                resume=not args.no_resume,
                logger=logger,
            ),
        )

        def improvement_stage():
            if args.calibration_fraction is not None:
                config.setdefault("experiments", {}).setdefault("improvement", {})[
                    "calibration_fraction"
                ] = args.calibration_fraction
            return run_improvement(
                dataset,
                config,
                paths,
                models=improvement_models,
                n_splits=n_splits,
                seed=seed,
                resume=not args.no_resume,
                logger=logger,
            )

        run_stage("improvement", improvement_stage)

        def cross_device_stage():
            manifest_path = Path(
                config.get("experiments", {}).get("cross_device", {}).get(
                    "manifest_path", "data/cross_device/manifest.csv"
                )
            )
            if not manifest_path.is_absolute():
                manifest_path = paths.root / manifest_path
            if not manifest_path.exists():
                logger.warning(
                    "Eksperimen lintas perangkat TIDAK dilakukan: manifest tidak tersedia di %s",
                    manifest_path,
                )
                return write_not_performed_result(
                    paths, dataset_source=dataset.source, is_synthetic=dataset.is_synthetic
                )
            logger.info("Menjalankan eksperimen lintas perangkat dari %s", manifest_path)
            multi = load_multi_device_dataset(manifest_path)
            return run_cross_device_experiment(multi, config, paths, seed=seed, logger=logger)

        run_stage("cross-device", cross_device_stage)

        if not args.no_report:
            logger.info("=== Membuat laporan dan figur ===")
            from scripts.generate_report import build_report

            report_start = time.perf_counter()
            report = build_report(config, paths, logger=logger, mode=mode)
            manifest["report"] = {
                "status": "complete",
                "duration_s": round(time.perf_counter() - report_start, 3),
                "summary": report.get("summary_path"),
                "figures": report.get("figures", []),
            }

        manifest["status"] = "complete"
        manifest["finished_at"] = now_iso()
        manifest["elapsed_s"] = round(time.perf_counter() - started, 3)
        persist_manifest()
        logger.info("Semua tahap selesai. Manifest: %s", paths.results / "run_manifest.json")
        if dataset.is_synthetic:
            logger.warning(DEMO_NOTICE)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        persist_manifest()
        logger.error("Eksperimen gagal: %s", exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

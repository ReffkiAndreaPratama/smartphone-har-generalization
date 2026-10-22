"""Prepare, verify and summarise the dataset (and generate exploration figures).

Examples
--------
python scripts/prepare_data.py --download
python scripts/prepare_data.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from src.utils import (  # noqa: E402
    ProjectPaths,
    get_logger,
    load_config,
    save_json,
    set_seed,
)
from src.visualization import (  # noqa: E402
    plot_class_distribution,
    plot_feature_groups,
    plot_samples_per_subject,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--download", action="store_true", help="Unduh dataset bila belum ada")
    parser.add_argument("--demo", action="store_true", help="Gunakan data demo sintetis")
    parser.add_argument("--force", action="store_true", help="Bangun ulang cache dan verifikasi")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    mode = "demo" if args.demo else "research"
    paths = ProjectPaths.from_config(config, mode=mode).ensure()
    logger = get_logger("har.prepare")
    set_seed(int(config.get("project", {}).get("seed", 42)))

    if args.force:
        for name in ("uci_har_cache", "demo_cache"):
            for path in paths.data_processed.glob(f"{name}.*"):
                if path.suffix in {".npz", ".json"}:
                    path.unlink()

    if args.demo:
        logger.info("Membuat dataset demo sintetis (bukan data penelitian)")
        dataset = load_demo_dataset(paths, config)
        verification = None
    else:
        dataset_dir = find_dataset_dir(paths.data_raw)
        if dataset_dir is None and args.download:
            dataset_dir = download_uci_har(paths, config, force=args.force, logger=logger)
        if dataset_dir is None:
            logger.error(
                "Dataset UCI HAR belum tersedia. Jalankan dengan --download atau letakkan "
                "'UCI HAR Dataset' di %s",
                paths.data_raw,
            )
            return 2
        logger.info("Memverifikasi struktur dataset: %s", dataset_dir)
        verification = verify_uci_structure(
            dataset_dir, expected=config.get("dataset", {}).get("expected", {})
        )
        save_json(paths.tables / "dataset_verification.json", verification)
        for warning in verification.get("warnings", []):
            logger.warning(warning)
        dataset = load_uci_har(dataset_dir, load_raw=True)
        cache_dataset(dataset, paths)

    summary = build_dataset_summary(
        dataset,
        expected=config.get("dataset", {}).get("expected", {}) if not args.demo else None,
    )
    save_dataset_summary(summary, paths)
    logger.info(
        "Dataset: %s | %d sampel | %d fitur | %d subjek | %d train / %d test",
        summary["source_label"],
        summary["n_samples"],
        summary["n_features"],
        summary["n_subjects"],
        summary["n_train"],
        summary["n_test"],
    )
    logger.info("Distribusi kelas: %s", summary["class_distribution"])

    if not args.skip_figures:
        figures = paths.figures
        plot_class_distribution(summary, figures / "01_class_distribution.png")
        plot_samples_per_subject(summary, figures / "02_samples_per_subject.png")
        plot_feature_groups(summary, figures / "03_feature_groups.png")
        logger.info("Gambar eksplorasi dataset disimpan di %s", figures)

    save_json(
        paths.tables / "dataset_run_info.json",
        {
            "mode": mode,
            "source": dataset.source,
            "is_synthetic": dataset.is_synthetic,
            "verification": verification,
            "paths": paths.as_dict(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for the optional cross-device module (UCI HAR has a single device)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cross_device import (
    CrossDeviceDataUnavailable,
    load_multi_device_dataset,
    write_not_performed_result,
)
from src.utils import RAW_CHANNELS, DatasetIntegrityError


def test_missing_manifest_raises(paths):
    with pytest.raises(CrossDeviceDataUnavailable):
        load_multi_device_dataset(paths.root / "data" / "cross_device" / "missing.csv")


def test_manifest_requires_two_devices(tmp_path):
    window = np.zeros((128, len(RAW_CHANNELS)), dtype=np.float32)
    path = tmp_path / "window.npy"
    np.save(path, window)
    frame = pd.DataFrame(
        [
            {
                "file_path": str(path),
                "subject_id": "S1",
                "device_id": "D1",
                "activity": "WALKING",
            }
        ]
    )
    manifest = tmp_path / "manifest.csv"
    frame.to_csv(manifest, index=False)
    with pytest.raises(DatasetIntegrityError):
        load_multi_device_dataset(manifest)


def test_not_performed_result_is_explicit(paths):
    payload = write_not_performed_result(paths, dataset_source="uci_har")
    assert payload["status"] == "not_performed"
    assert "satu perangkat" in payload["reason"] or "Samsung" in payload["available_devices"][0]
    assert payload["requirements"]
    assert payload["documentation"] == "docs/cross_device_protocol.md"
    assert (paths.results / "cross_device.json").exists()


def test_not_performed_result_does_not_overwrite_completed_run(paths):
    payload = write_not_performed_result(paths, dataset_source="uci_har")
    payload["status"] = "complete"
    from src.utils import save_json

    save_json(paths.results / "cross_device.json", payload)
    again = write_not_performed_result(paths, dataset_source="uci_har")
    assert again["status"] == "complete"

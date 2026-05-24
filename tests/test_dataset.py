from __future__ import annotations

from src.dataset import WindowedNPPADDataset, build_class_maps, load_trainable_manifest, split_manifest_by_file
from src.preprocessing import TimeSeriesStandardScaler


def test_window_dataset_shape_and_scaler_roundtrip(tmp_path):
    manifest = load_trainable_manifest(max_files_per_class=4, classes=["FLB", "LOCA"])
    class_to_idx, _ = build_class_maps(manifest)
    train_m, _, _ = split_manifest_by_file(manifest)
    scaler = TimeSeriesStandardScaler.fit_from_files(train_m["path"].tolist())
    path = tmp_path / "scaler.pkl"
    scaler.save(path)
    loaded = TimeSeriesStandardScaler.load(path)
    ds = WindowedNPPADDataset(train_m, class_to_idx, loaded, window_size=64, stride=64, early_ratio=0.5, max_windows_per_file=1)
    item = ds[0]
    assert item["x"].shape == (64, len(loaded.sensor_cols))
    assert item["y_class"].ndim == 0
    assert item["y_severity"].ndim == 0
